# RFBinaryDetect

Standalone binary drone/no-drone pipeline built from ideas in `RFClassification` without modifying that repo.

## CLI

- `rfbd extract`
- `rfbd process-no-drone-batches`
- `rfbd dataset build`
- `rfbd train binary --arch <arch>`
- `rfbd train vgg-binary` (compat alias to `--arch vgg16`)
- `rfbd eval`
- `rfbd export`
- `rfbd hailo prepare`
- `rfbd hailo compile`
- `rfbd hailo validate`

## Data contracts

Raw manifest CSV columns:

- `capture_id,file_path,label,source_domain,center_freq_hz,sample_rate_sps,gain_db,session_id,timestamp,environment`

Feature shard NPZ arrays:

- `feat` shape `(N,224,224)` float32
- `y` shape `(N,)` int64 (`0=no_drone`, `1=drone`)
- `source_domain` shape `(N,)` string
- `capture_id` shape `(N,)` string
- `session_id` shape `(N,)` string

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

rfbd extract --adapter custom-iq --manifest /path/manifest.csv --out-dir /tmp/custom_shards
rfbd dataset build --custom-shards-dir /tmp/custom_shards --out-dir /tmp/dataset
rfbd train binary --dataset-dir /tmp/dataset --out-dir /tmp/models --arch mobilenet_v3_small --model-name mobilenet.pt
rfbd eval --dataset-dir /tmp/dataset --out-dir /tmp/eval --arch mobilenet_v3_small
rfbd export --checkpoint /tmp/models/mobilenet.pt --out-dir /tmp/export --format torchscript --format onnx
```

Supported architectures:

- `vgg16`
- `resnet18`
- `mobilenet_v3_small`
- `shufflenet_v2_x1_0`
- `vgg13`
- `resnet34`
- `resnet50`
- `regnet_x_1_6gf`
- `vgg_small_gap`
- `repvgg_a1`
- `repvgg_a2`
- `repvgg_a1_hmz`
- `repvgg_a2_hmz`

Checkpoint metadata now includes:

- `arch`
- `threshold`
- `config`
- `preprocessing_contract`
- `label_contract`
- optional `pretrained_init` when a RepVGG checkpoint was initialized from Hailo Model Zoo ONNX weights

## Model Selection Workflow

Baseline lock (current VGG behavior + runtime metrics):

```bash
./scripts/run_baseline_lock.sh
```

Candidate suite (train/eval/export/benchmark for all architectures + leaderboard + strict-FAR reruns):

```bash
./scripts/train_candidate_suite.sh
```

The local RepVGG names, `repvgg_a1` and `repvgg_a2`, are trained from the repo's own initialization path. The Hailo Model Zoo-initialized variants use distinct names, `repvgg_a1_hmz` and `repvgg_a2_hmz`, so their checkpoints and Hailo artifacts do not overwrite local RepVGG runs:

```bash
ARCHES="repvgg_a1_hmz repvgg_a2_hmz" bash ./new_models_run.sh
ARCHES="repvgg_a1_hmz repvgg_a2_hmz" bash ./new_models_eval_run.sh
```

The local ONNX files are expected at `data/pretrained/hailo_model_zoo/repvgg/extracted/RepVGG-A1.onnx` and `data/pretrained/hailo_model_zoo/repvgg/extracted/RepVGG-A2.onnx`. These official files are ImageNet classifiers, so RF training imports the RepVGG backbone tensors and keeps the binary classifier head task-specific. The older `--use-repvgg-model-zoo` flag still works for compatibility, but the `*_hmz` names are preferred for new runs.

Key outputs:

- `data/experiments/g2_edge_suite/leaderboard/leaderboard.json`
- `data/experiments/g2_edge_suite/leaderboard/leaderboard.csv`
- `data/experiments/g2_edge_suite/leaderboard/strict_far_report.json`

## Hailo Compilation

Hailo workspace preparation command:

```bash
rfbd hailo prepare
```

This command:

- discovers the current deployable checkpoints from `g2_edge_suite`
- exports static-batch ONNX artifacts for Hailo compilation
- creates mirrored `strict_far/exports/` roots when needed
- writes target-specific Hailo manifest stubs for the requested target; the default is `hailo8`
- writes `data/experiments/g2_edge_suite/hailo_compilation_inventory.json`

HEF compilation command:

```bash
rfbd hailo compile --calibration-samples 256
```

This command:

- reuses or regenerates the static-batch ONNX exports
- builds an NHWC calibration `.npy` from `data/datasets/binary_all_v1`
- runs Hailo `parser`, `optimize`, and `compiler` for each selected target
- writes `.hef`, `.har`, `.log`, and command-record artifacts under each model's `hailo/` directory
- updates each manifest with the real compiled artifact paths and detected Hailo tool versions
- auto-validates compiled artifacts and leaves them non-deployable until quality and runtime gates pass
- treats `vgg13`/`vgg16` Hailo artifacts as legacy reference only; use `vgg_small_gap` for the Hailo-friendly VGG-style path, and `repvgg_a1_hmz`/`repvgg_a2_hmz` as Hailo-supported RepVGG Model Zoo backup candidates

HEF validation command:

```bash
rfbd hailo validate --model-id vgg16_binary --target hailo8 --runtime-metrics-json /path/runtime.json
```

This command:

- builds a deterministic probe corpus with synthetic and real PSD frames
- compares PyTorch, ONNX, and available Hailo-emulation stages on the same probes
- writes per-target `*.validation.json` reports
- marks `deployable` in the manifest only when the artifact passes both quality and runtime gates

G2 VGG16 handoff:

- `g2_vgg16_hailo_fix_handoff.md`

Legacy typoed filename kept as a compatibility pointer:

- `halio_compilation_plan.md`

## Host GPU Run (No-Reboot Recovery + RAM-Safe Dataset)

When CUDA/NVML fails in restricted shells, run the full pipeline from host shell:

```bash
cd /root/RFBinaryDetect
./scripts/run_binary_all_v1_host.sh
```

The script will:

- fail-fast on GPU preflight (`nvidia-smi` + torch CUDA check),
- attempt no-reboot GPU recovery (`nvidia-persistenced`, `nvidia-modprobe`),
- build curated subsets for RAM-safe training,
- build merged dataset (`binary_all_v1`),
- train + eval with `--require-gpu`.

Manual fail-fast flags are also available:

```bash
rfbd train binary ... --require-gpu
rfbd eval ... --require-gpu
```
