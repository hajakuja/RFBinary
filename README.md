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

Checkpoint metadata now includes:

- `arch`
- `threshold`
- `config`
- `preprocessing_contract`
- `label_contract`

## Model Selection Workflow

Baseline lock (current VGG behavior + runtime metrics):

```bash
./scripts/run_baseline_lock.sh
```

Candidate suite (train/eval/export/benchmark for all architectures + leaderboard + strict-FAR reruns):

```bash
./scripts/train_candidate_suite.sh
```

Key outputs:

- `data/experiments/g2_edge_suite/leaderboard/leaderboard.json`
- `data/experiments/g2_edge_suite/leaderboard/leaderboard.csv`
- `data/experiments/g2_edge_suite/leaderboard/strict_far_report.json`

## Hailo Compilation Planning

Canonical Hailo planning document:

- `hailo_compilation_plan.md`

Hailo workspace preparation command:

```bash
rfbd hailo prepare
```

This command:

- discovers the current deployable checkpoints from `g2_edge_suite`
- exports static-batch ONNX artifacts for Hailo compilation
- creates mirrored `strict_far/exports/` roots when needed
- writes target-specific Hailo manifest stubs for `hailo8` and `hailo8l`
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
