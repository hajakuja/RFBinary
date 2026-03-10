# RFBinaryDetect

Standalone binary drone/no-drone pipeline built from ideas in `RFClassification` without modifying that repo.

## CLI

- `rfbd extract`
- `rfbd process-no-drone-batches`
- `rfbd dataset build`
- `rfbd train vgg-binary`
- `rfbd eval`

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
rfbd train vgg-binary --dataset-dir /tmp/dataset --out-dir /tmp/models
rfbd eval --dataset-dir /tmp/dataset --out-dir /tmp/eval
```

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
rfbd train vgg-binary ... --require-gpu
rfbd eval ... --require-gpu
```
