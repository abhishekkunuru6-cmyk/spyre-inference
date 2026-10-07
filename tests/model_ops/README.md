# Model-ops YAML generation

Generates the torch-spyre op-test YAML (`test_suite_config`, one case per distinct op signature)
for a model loaded through spyre-inference, so the recorded ops are the ones this plugin compiles
and not stock Hugging Face code.

`utils/torchop_yaml.py` is a copy of
[`utils/model_ops/utils/torchop_yaml.py`](https://github.com/torch-spyre/hf-adapters/blob/585c093/utils/model_ops/utils/torchop_yaml.py)
in hf-adapters at commit `585c093`, with two changes:

- `_extract_meta_info` treats `FakeScriptObject` / `ScriptObject` values (the opaque attention
  handle passed to `unified_attention_with_output`) as "no shape" instead of raising.
- `_compile_fx` forwards its arguments to `compile_fx` untouched, so it no longer collides with
  torch-spyre's `compile_fx` wrapper (`got multiple values for argument 'decompositions'`).

## Run

Use the spyre-inference image on a Spyre host (one card is enough at TP=1). Only one process may
use the card at a time.

```bash
source /app/.venv/bin/activate
export HF_TOKEN=<token>   # google/gemma-4-26B-A4B-it is gated
cd tests/model_ops
rm -rf /tmp/torchinductor_* ~/.cache/vllm/torch_compile_cache
python -m models.gemma4-26b-a4b.run_spyre_inference 2>&1 | tee run.log
```

The YAML is written to the current directory as `gemma-4-26B-A4B-it.yaml`. The first run downloads
the weights and compiles a graph per layer type and bucket, so it is slow; the log is large.

The collector needs `PyYAML`, `regex` and `python-dotenv` in the environment.

## Notes

- `gemma-4-26B-A4B-it.yaml` is the raw output of this driver (image `spyre-inference:ci-cd-tech-preview-v3`:
  55 ops traced, 51 with test configs, 1127 cases). It includes ops that the torch-spyre test run skips
  as unregistered (`torch.ops.spyre.*`, `torch_spyre._monkey_patch.*`, `torch.ops.aten.*`,
  `torch._C._autograd.*`, and a few others such as `torch.unflatten` and `torch.index_select`).
- `<model>_spyre.yaml` (the normalized copy) is not written (`supress_spyre=True`).
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` keeps the engine in the collector's process; do not use
  `--enforce-eager`, which compiles nothing.
