import functools
import math

import custom_mesh
import jax
import numpy as np
import splash_attn
import torch
import torch_xla2
import train
from jax.experimental.mesh_utils import (
  create_device_mesh,
  create_hybrid_device_mesh,
)
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from llama import model, model_with_collectives, model_with_scan
from torch_xla2 import interop

sharding_map_original = {
  "freqs_cis": (),  #  torch.complex64 (2048, 64)
  "tok_embeddings.weight": (
    "fsdp",
    "tp",
  ),  #  torch.float32 (vocab_size, 4096)
  "layers.*.attention.wo.weight": ("fsdp", "tp"),  #  torch.int8 (4096, 4096)
  "layers.*.attention.wq.weight": ("tp", "fsdp"),  #  torch.int8 (4096, 4096)
  "layers.*.attention.wk.weight": ("tp", "fsdp"),  #  torch.int8 (4096, 4096)
  "layers.*.attention.wv.weight": ("tp", "fsdp"),  #  torch.int8 (4096, 4096)
  "layers.*.feed_forward.w1.weight": (
    "tp",
    "fsdp",
  ),  #  torch.float32 (11008, 4096)
  "layers.*.feed_forward.w2.weight": (
    "fsdp",
    "tp",
  ),  #  torch.float32 (4096, 11008)
  "layers.*.feed_forward.w3.weight": (
    "tp",
    "fsdp",
  ),  #  torch.float32 (11008, 4096)
  "layers.*.attention_norm.weight": ("fsdp",),  #  torch.float32 (4096,)
  "layers.*.ffn_norm.weight": ("fsdp",),  #  torch.float32 (4096,)
  "norm.weight": ("fsdp",),  #  torch.float32 (4096,)
  "output.weight": ("tp", "fsdp"),  #  torch.float32 (vocab_size, 4096)
}

sharding_map_scan = {
  "freqs_cis": (),  #  torch.complex64 (2048, 64)
  # ParallelEmbedding for llama2; VocabParallelEmbedding for 3
  "tok_embeddings.weight": (
    "tp",
    "fsdp",
  ),  #  torch.float32 (vocab_size, 4096)
  "layers.params.attention___wo___weight": (
    None,
    "fsdp",
    "tp",
  ),  #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wq___weight": (
    None,
    "tp",
    "fsdp",
  ),  #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wk___weight": (
    None,
    "tp",
    "fsdp",
  ),  #  torch.int8 (n, 4096, 4096)
  "layers.params.attention___wv___weight": (
    None,
    "tp",
    "fsdp",
  ),  #  torch.int8 (n, 4096, 4096)
  "layers.params.feed_forward___w1___weight": (
    None,
    "tp",
    "fsdp",
  ),  #  torch.float32 (n, 11008, 4096)
  "layers.params.feed_forward___w2___weight": (
    None,
    "fsdp",
    "tp",
  ),  #  torch.float32 (n, 4096, 11008)
  "layers.params.feed_forward___w3___weight": (
    None,
    "tp",
    "fsdp",
  ),  #  torch.float32 (n, 11008, 4096)
  "layers.params.attention_norm___weight": (
    None,
    "fsdp",
  ),  #  torch.float32 (n, 4096,)
  "layers.params.ffn_norm___weight": (
    None,
    "fsdp",
  ),  #  torch.float32 (n, 4096,)
  "norm.weight": ("fsdp",),  #  torch.float32 (4096,)
  "output.weight": ("tp", "fsdp"),  #  torch.float32 (vocab_size, 4096)
}


def _bytes(x: jax.Array | jax.ShapeDtypeStruct) -> int:
  return math.prod(x.shape) * x.dtype.itemsize


def _process_sharding_name(name):
  """Replace integers in param name with *.

  Presumably all layers should have the same sharding.
  """

  def is_integer(t):
    try:
      int(t)
      return True
    # pylint: disable-next=all
    except:  # noqa: E722
      return False

  tokens = name.split(".")
  for i, t in enumerate(tokens):
    if is_integer(t):
      tokens[i] = "*"
  return ".".join(tokens)


def register_attention(fn):
  from torch_xla2.ops import ops_registry

  env = torch_xla2.default_env()
  k = torch.nn.functional.scaled_dot_product_attention
  env._ops[k] = ops_registry.Operator(
    k, fn, is_jax_function=False, is_user_defined=True, needs_env=False
  )


def make_weight_shard(weight_meta, slice_index):
  weight_shard_meta = weight_meta[slice_index]
  with torch_xla2.default_env():
    return interop.jax_view(
      torch.randn(weight_shard_meta.shape, dtype=weight_shard_meta.dtype)
    )


def create_sharded_weights(model, mesh, sharding_map):
  res = {}
  for name, weight_meta in model.state_dict().items():
    sharding_spec = sharding_map.get(_process_sharding_name(name))
    if sharding_spec is None:
      print("Skipping weight:", name)
      continue
    sharding = NamedSharding(mesh, P(*sharding_spec))
    res[name] = jax.make_array_from_callback(
      weight_meta.shape, sharding, functools.partial(make_weight_shard, weight_meta)
    )
  return res


def sharded_device_put(tensor, sharding):
  num_global_devices = jax.device_count()
  num_local_devices = jax.local_device_count()
  if isinstance(tensor, tuple):
    return tuple(sharded_device_put(t, sharding) for t in tensor)

  if num_global_devices == num_local_devices:
    return jax.device_put(tensor, sharding)

  shape = tensor.shape
  x_split = [
    jax.device_put(tensor[i], device)
    for device, i in sharding.addressable_devices_indices_map(shape).items()
  ]
  return jax.make_array_from_single_device_arrays(shape, sharding, x_split)


def main(
  batch_size: int = 64,
  model_type: str = "8B",
  lr: float = 0.001,
  tp: int = 4,
  seqlen: int = 2048,
  model_impl: str = "scan",
  use_custom_mesh: bool = False,
  use_custom_offload: bool = True,
  internal_override_layers: int = -1,
  profile_dir: str = "profile/",
  unroll_layers: int = 1,
):
  print(locals())
  torch.manual_seed(0)
  torch.set_default_dtype(torch.bfloat16)
  torch_xla2.enable_performance_mode()

  print("Local devices:", jax.local_device_count())
  fsdp_size = len(jax.devices()) // tp

  env = torch_xla2.default_env()
  env.config.use_torch_native_for_cpu_tensor = False

  if use_custom_mesh:
    tp = 4
    if len(jax.devices()) == 512:
      dev_array = custom_mesh.create_custom_64x4_device_mesh(
        (64, 4), (2, 1), jax.devices()
      )
    else:
      assert len(jax.devices()) == 256
      dev_array = (
        np.array(jax.devices()).reshape(8, 2, 8, 2).transpose(0, 2, 1, 3).reshape(64, 4)
      )
  else:
    if fsdp_size * tp <= 256:
      dev_array = create_device_mesh((fsdp_size, tp), allow_split_physical_axes=True)
    else:
      num_pod = len(jax.devices()) // 256
      dev_array = create_hybrid_device_mesh(
        (fsdp_size // num_pod, tp), (num_pod, 1), jax.devices()
      )
  mesh = Mesh(dev_array, ("fsdp", "tp"))

  if use_custom_offload:
    policy = jax.checkpoint_policies.save_and_offload_only_these_names(
      names_which_can_be_saved=[],
      names_which_can_be_offloaded=[
        "decoder_layer_input",
        "query_proj",
        "key_proj",
        "value_proj",
        "out_proj",
      ],
      offload_src="device",
      offload_dst="pinned_host",
    )
  else:
    policy = jax.checkpoint_policies.nothing_saveable

  args = model.ModelArgs(**model.transformer_configs[model_type])
  if internal_override_layers > 0:
    args.n_layers = internal_override_layers

  with torch.device("meta"):
    if model_impl == "scan":
      sharding_map = sharding_map_scan
      llama = model_with_scan.Transformer(args)
    elif model_impl == "scan_manual":
      args.tp_size = tp
      sharding_map = sharding_map_scan
      llama = model_with_collectives.Transformer(args, unroll_layers)
    elif model_impl == "orig":
      sharding_map = sharding_map_original
      llama = model.Transformer(args)
    else:
      raise AssertionError("unknown impl: " + model_impl)

  sharded_weights = create_sharded_weights(llama, mesh, sharding_map)
  with torch.device("cpu"):
    freqs_cis = model.precompute_freqs_cis(
      args.dim // args.n_heads,
      args.max_seq_len * 2,
      args.rope_theta,
      args.use_scaled_rope,
    ).numpy()
  sharding = NamedSharding(mesh, P())  # replicated

  env = torch_xla2.default_env()
  freqs_cis = env.j2t_iso(jax.device_put(freqs_cis, sharding))

  # NOTE: overriding attention to capture mesh and sharding info
  partition = P("fsdp", "tp", None, None)
  attention = functools.partial(
    splash_attn.tpu_splash_attention,
    mesh,
    partition,
    (model_impl != "scan_manual"),
  )
  attention = jax.jit(attention)

  def custom_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
  ):
    #  batch, num of head, seq, dim
    jk, jq, jv = interop.jax_view((query, key, value))
    res = attention(jk, jq, jv, None)
    return interop.torch_view(res)

  register_attention(custom_attention)

  with mesh:
    train.train_loop(
      mesh,
      llama,
      sharded_weights,
      None,
      freqs_cis,
      lr,
      seqlen,
      policy,
      batch_size,
      use_shmap=(model_impl == "scan_manual"),
      profile_dir=profile_dir,
    )


if __name__ == "__main__":
  import fire

  fire.Fire(main)
