# Inference Guide: `RFBinaryDetect` Binary Drone Detector

This guide explains exactly how to run inference with the trained VGG binary model in another environment, including RF capture settings, sample counts, preprocessing, model loading, and output interpretation.

## 1. Label and Output Contract

- Binary labels are fixed:
  - `0 = no_drone`
  - `1 = drone`
- String mapping is exact:
  - `0 -> "no_drone"`
  - `1 -> "drone"`

At inference time you should produce, per segment:
- `p_drone` (probability for class `1`)
- `pred_id` (`0` or `1`) using model threshold
- `pred_label` (`"no_drone"` or `"drone"`)

## 2. What the Current Model Was Trained On

Model artifact:
- Checkpoint: `/root/RFBinaryDetect/data/models/binary_all_v1/vgg_binary_all_v1.pt`
- Summary: `/root/RFBinaryDetect/data/models/binary_all_v1/vgg_binary_all_v1_summary.json`

Training config stored in checkpoint:
- `batch_size=64`
- `epochs=8`
- `learning_rate=3e-4`
- `weight_decay=1e-4`
- `pretrained=True` (ImageNet VGG16 backbone init)
- `freeze_features=True` (feature extractor frozen during train)
- `target_far=0.05`
- `seed=13`

Learned decision threshold in this checkpoint:
- `threshold = 0.79`

Merged dataset summary used for this model:
- total samples: `34,043`
- label counts: `no_drone=9,152`, `drone=24,891`
- source domains: `dronedetect=20,487`, `dronerf=5,364`, `custom_bg=8,192`

Important domain note:
- The merged contract shards do not retain center-frequency metadata.
- Your custom no-drone runs (`nd_run_01`, `nd_run_02`) were captured at `2.437e9 Hz`, `20,971,520 sps`, gains `35/50/65 dB`.
- Collection protocol includes both `2.437e9` and `5.735e9`, but your currently ingested no-drone features are concentrated at `2.437e9`.

## 3. RF Capture Settings for Best Inference Results

Use these settings to match training/preprocessing assumptions and reduce domain shift.

## 3.1 Frequencies to monitor

Recommended operating bands from protocol:
- `2.437e9 Hz` (2.4 GHz)
- `5.735e9 Hz` (5.8 GHz)

Practical guidance:
- If you can only run one center frequency, prioritize `2.437e9 Hz` for now (most consistent with your current no-drone training captures).
- If you run both, process each band independently and fuse decisions at the application layer.

## 3.2 Sample rate

- Target sample rate: `20_971_520 sps`
- Keep inference sample rate equal to training sample rate whenever possible.
- If you change sample rate, segment length in samples changes and model behavior can shift.

## 3.3 Gain

- Use fixed gain values similar to training captures: `35 dB`, `50 dB`, `65 dB` (or hardware-equivalent low/mid/high).
- Avoid AGC if it causes unstable amplitude statistics across captures.
- Avoid clipping/saturation (for int16 IQ, sustained values near +/-32767 are bad).

## 3.4 Minimum sample counts

Feature extraction is segment-based with fixed segment duration:
- `segment_ms = 20`

Samples per segment:
- `samples_per_segment = int(sample_rate_sps * 0.02)`
- At `20,971,520 sps`: `419,430` complex IQ samples per segment

Raw bytes per segment for interleaved int16 IQ (`I,Q`):
- `419,430 * 2 channels * 2 bytes = 1,677,720 bytes`

Recommended capture duration per inference decision:
- minimum: `1.0 s` (`50` segments)
- preferred: `2.0 to 3.0 s` (`100 to 150` segments)

The pipeline drops incomplete trailing segments, so capture enough data to avoid sparse decisions.

## 4. Raw IQ Input Formats Supported by Current Code

`rfbd` custom IQ loader supports:
- `.s16`, `.bin`, `.iq`: interleaved int16 `[I0,Q0,I1,Q1,...]`
- `.npy`: complex vector or `(N,2)` I/Q float array
- `.npz`: key `iq` preferred, otherwise first array

If your deployment recorder uses another format, convert it into one of the above before extraction.

## 5. Exact Preprocessing Pipeline Before the Model

This is the exact transform path used by `rfbd extract --adapter custom-iq`.

1. Load raw IQ to complex64 vector.
2. Keep real component only:
   - `signal_real = np.real(iq).astype(np.float32)`
3. Split into non-overlapping fixed windows:
   - window length: `samples_per_segment`
   - incomplete remainder at end is discarded
4. For each segment, compute PSD spectrogram:
   - function: `scipy.signal.spectrogram`
   - `window="hann"`
   - `nperseg=min(1024, len(segment))`
   - `noverlap=min(120, nperseg-1)`
   - `scaling="density"`
   - `mode="psd"`
5. Convert to log power:
   - `10 * log10(psd + 1e-12)`
6. Per-segment min-max normalization to `[0,1]`:
   - if nearly constant, output zeros
7. Resize each spectrogram to `224 x 224` with OpenCV bilinear interpolation.
8. Final model tensor per segment:
   - shape `(224,224)`, dtype `float32`

Do not add extra normalization/transforms after this unless you retrain.

## 6. Model Input and Architecture Expectations

Model class: `rfbd.modeling.VGG16Binary`

Input shapes accepted:
- `(B,H,W)` or `(B,1,H,W)` or `(B,3,H,W)`

Internal behavior:
- if channel count is `1`, model repeats to `3` channels before VGG16.
- final classifier head has `2` outputs (`no_drone`, `drone`).

## 7. Loading the Model and Running Inference

Use the same package version family used for training when possible.

## 7.1 Environment setup

```bash
cd /root/RFBinaryDetect
source .venv/bin/activate
pip install -e .
```

## 7.2 Python inference example (segment-level + aggregated decision)

```python
from pathlib import Path
import numpy as np
import torch
from rfbd.modeling import VGG16Binary
from rfbd.labels import ID_TO_LABEL
from rfbd.features import load_custom_iq, split_segments_1d, samples_per_segment, compute_spec_feature
from rfbd.contracts import FeatureConfig

# ---- Configuration: must match training preprocessing ----
SAMPLE_RATE = 20_971_520.0
SEGMENT_MS = 20
CFG = FeatureConfig(
    segment_ms=SEGMENT_MS,
    nfft=1024,
    noverlap=120,
    resize_h=224,
    resize_w=224,
    log_power=True,
    normalize=True,
)

CKPT_PATH = Path("/root/RFBinaryDetect/data/models/binary_all_v1/vgg_binary_all_v1.pt")
IQ_PATH = Path("/path/to/new_capture.s16")

# ---- Load checkpoint ----
ckpt = torch.load(CKPT_PATH, map_location="cpu")
threshold = float(ckpt["threshold"])  # e.g. 0.79

model = VGG16Binary(pretrained=False, freeze_features=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---- Preprocess raw IQ -> feature tensor ----
iq = load_custom_iq(IQ_PATH)                      # complex64
x = np.real(iq).astype(np.float32, copy=False)    # IMPORTANT: real part only
seg_len = samples_per_segment(SAMPLE_RATE, SEGMENT_MS)

feats = []
for seg in split_segments_1d(x, seg_len):
    feats.append(compute_spec_feature(seg, SAMPLE_RATE, CFG))

if not feats:
    raise RuntimeError("No full 20 ms segments available in this capture.")

X = np.stack(feats, axis=0).astype(np.float32)    # (N,224,224)
xb = torch.from_numpy(X).to(device)

with torch.no_grad():
    logits = model(xb)
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # p(drone)

pred_ids = (probs >= threshold).astype(np.int64)
pred_labels = [ID_TO_LABEL[int(v)] for v in pred_ids]

# Segment-level outputs
print("segments:", len(probs))
print("threshold:", threshold)
print("mean_p_drone:", float(np.mean(probs)))
print("drone_segment_ratio:", float(np.mean(pred_ids)))
print("first_10_labels:", pred_labels[:10])

# Optional clip-level aggregation:
# conservative decision = drone if >=20% segments exceed threshold
clip_is_drone = float(np.mean(pred_ids)) >= 0.20
print("clip_label:", "drone" if clip_is_drone else "no_drone")
```

## 8. How to Interpret Outputs Correctly

Per segment:
- `p_drone` near `1.0`: strong drone evidence
- `p_drone` near `0.0`: strong background evidence
- decision uses checkpoint threshold, not always `0.5`

For this trained model:
- threshold `0.79` was selected to target low false alarm behavior.
- This is strict; recall can decrease if drone signal is weak.

For robust system-level decisions, aggregate segment decisions over time:
- use a sliding window (for example 1 to 3 seconds)
- require persistent evidence (ratio or consecutive positives) before alarming

## 9. Deployment Consistency Checklist

Before trusting field predictions, verify:
- same sample rate (`20,971,520 sps`)
- same segment config (`20 ms`, `nfft=1024`, `noverlap=120`, `224x224`)
- same preprocessing switches (`log_power=True`, `normalize=True`)
- input format decoded correctly as IQ interleaved int16 (if `.s16/.bin`)
- no clipping at SDR front-end
- threshold loaded from checkpoint and applied

## 10. Common Failure Modes

- `torch.cuda.is_available() == True` but CUDA tensor allocation fails:
  - verify with strict probe (`torch.zeros(1, device="cuda")`) before runtime.
- Wrong sample rate passed into preprocessing:
  - segments still compute, but spectrogram time/frequency structure shifts.
- Feeding imaginary channel or magnitude instead of real channel:
  - distribution mismatch versus training path.
- Changing preprocessing (different FFT, overlap, scaling, resize):
  - model will degrade without retraining.
- Very short captures:
  - too few segments, unstable decision.

## 11. Recommended Next Improvement for Production

If inference environment differs materially (different SDR chain, new bands, new RF clutter), run periodic recalibration:
- collect labeled data in deployment conditions,
- regenerate features with the exact same pipeline,
- re-train and re-tune threshold.

This is the most reliable way to keep false alarms and misses under control in a new domain.
