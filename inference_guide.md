# Inference Guide: `RFBinaryDetect` Binary Drone Detector

This guide explains how to run `RFBinaryDetect` models in deployment, both with the original PyTorch checkpoints and with compiled Hailo `.hef` artifacts for Raspberry Pi deployments.

## 1. Label and Output Contract

- Binary labels are fixed:
  - `0 = no_drone`
  - `1 = drone`
- String mapping is exact:
  - `0 -> "no_drone"`
  - `1 -> "drone"`

At inference time you should produce, per segment:
- `p_drone` (probability for class `1`)
- `pred_id` (`0` or `1`) using the model threshold
- `pred_label` (`"no_drone"` or `"drone"`)

## 2. Baseline Model and Multi-Arch Compatibility

Baseline model artifacts:
- Checkpoint: `/root/RFBinaryDetect/data/models/binary_all_v1/vgg_binary_all_v1.pt`
- Summary: `/root/RFBinaryDetect/data/models/binary_all_v1/vgg_binary_all_v1_summary.json`

Baseline training config stored in checkpoint:
- `batch_size=64`
- `epochs=8`
- `learning_rate=3e-4`
- `weight_decay=1e-4`
- `pretrained=True`
- `freeze_features=True`
- `target_far=0.05`
- `seed=13`

Baseline learned decision threshold:
- `threshold = 0.79`

Current compatibility and architecture notes:
- New checkpoints include `arch` and can be loaded with `rfbd.modeling.create_binary_model(...)`.
- Supported architectures: `vgg13`, `vgg16`, `resnet18`, `resnet34`, `resnet50`, `mobilenet_v3_small`, `shufflenet_v2_x1_0`, `regnet_x_1_6gf`, `vgg_small_gap`, `repvgg_a1`, `repvgg_a2`, `repvgg_a1_hmz`, `repvgg_a2_hmz`.
- `vgg13` is the supported replacement for the requested `vgg14`; there is no separate custom `vgg14` architecture in this repo.
- `vgg_small_gap` is the Hailo-friendly VGG-style candidate; it uses VGG-like convolution blocks with global average pooling instead of the large torchvision VGG classifier.
- `repvgg_a1` and `repvgg_a2` are local RepVGG backup candidates trained from the repo's own initialization path.
- `repvgg_a1_hmz` and `repvgg_a2_hmz` are the Hailo Model Zoo-initialized RepVGG backup candidates; they import the official ImageNet backbone tensors, keep the RF binary head task-specific, and export to fused deploy-form convolution blocks for ONNX/Hailo.
- Legacy `rfbd train vgg-binary` behavior is preserved; it is a compatibility alias for `rfbd train binary --arch vgg16`.

Current G2 candidate checkpoint snapshot:
- `vgg13_binary`
  - checkpoint: `/root/RFBinaryDetect/data/experiments/g2_edge_suite/models/vgg13/vgg13_binary.pt`
  - threshold: `0.67`
  - Hailo status: historical artifact only; corrected `hailo8` compile failed mapping and the old `.hef` must not be deployed
- `vgg_small_gap_binary`
  - checkpoint: `/root/RFBinaryDetect/data/experiments/g2_edge_suite/models/vgg_small_gap/vgg_small_gap_binary.pt` once trained
  - Hailo status: intended replacement VGG-style candidate; compile/validation required before deployment
- `resnet34_binary`
  - checkpoint: `/root/RFBinaryDetect/data/experiments/g2_edge_suite/models/resnet34/resnet34_binary.pt`
  - threshold: `0.83`
  - Hailo status: `hailo8` `.hef` exists, but validation decision is `rejected_quality`
- `resnet50_binary`
  - checkpoint: `/root/RFBinaryDetect/data/experiments/g2_edge_suite/models/resnet50/resnet50_binary.pt`
  - threshold: `0.57`
  - Hailo status: `hailo8` `.hef` exists, but validation decision is `rejected_quality`
- `regnet_x_1_6gf_binary`
  - checkpoint: `/root/RFBinaryDetect/data/experiments/g2_edge_suite/models/regnet_x_1_6gf/regnet_x_1_6gf_binary.pt`
  - threshold: `0.56`
  - Hailo status: `hailo8` `.hef` exists, but validation decision is `rejected_quality`

Live-deployment recommendation in the current repo snapshot:
- keep using `shufflenet_v2_x1_0_binary` as the fallback live Hailo model until a newer candidate has both `deployable=true` and confirmed hardware latency on the target device

Merged dataset summary used for the original baseline:
- total samples: `34,043`
- label counts: `no_drone=9,152`, `drone=24,891`
- source domains: `dronedetect=20,487`, `dronerf=5,364`, `custom_bg=8,192`

Important domain note:
- The merged contract shards do not retain center-frequency metadata.
- Your custom no-drone runs (`nd_run_01`, `nd_run_02`) were captured at `2.437e9 Hz`, `20,971,520 sps`, gains `35/50/65 dB`.
- Collection protocol includes both `2.437e9` and `5.735e9`, but the currently ingested no-drone features are concentrated at `2.437e9`.

## 3. RF Capture Settings for Best Inference Results

Use these settings to match training/preprocessing assumptions and reduce domain shift.

### 3.1 Frequencies to monitor

Recommended operating bands from protocol:
- `2.437e9 Hz` (2.4 GHz)
- `5.735e9 Hz` (5.8 GHz)

Practical guidance:
- If you can only run one center frequency, prioritize `2.437e9 Hz` for now.
- If you run both, process each band independently and fuse decisions at the application layer.

### 3.2 Sample rate

- Target sample rate: `20_971_520 sps`
- Keep inference sample rate equal to training sample rate whenever possible.
- If you change sample rate, segment length in samples changes and model behavior can shift.

### 3.3 Gain

- Use fixed gain values similar to training captures: `35 dB`, `50 dB`, `65 dB`.
- Avoid AGC if it causes unstable amplitude statistics across captures.
- Avoid clipping or saturation.

### 3.4 Minimum sample counts

Feature extraction is segment-based with fixed segment duration:
- `segment_ms = 20`

Samples per segment:
- `samples_per_segment = int(sample_rate_sps * 0.02)`
- At `20,971,520 sps`: `419,430` complex IQ samples per segment

Raw bytes per segment for interleaved int16 IQ:
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

1. Load raw IQ to a complex64 vector.
2. Keep the real component only:
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
6. Per-segment min-max normalize to `[0,1]`:
   - if nearly constant, output zeros
7. Resize each spectrogram to `224 x 224` with OpenCV bilinear interpolation.
8. Final model tensor per segment:
   - shape `(224,224)`, dtype `float32`

Do not add extra normalization or transforms after this unless you retrain.

## 6. Model Input and Architecture Expectations

Model builder: `rfbd.modeling.create_binary_model`

Input shapes accepted by the PyTorch models:
- `(B,H,W)`
- `(B,1,H,W)`
- `(B,3,H,W)`

Internal behavior:
- if channel count is `1`, the model repeats to `3` channels before backbone inference
- final classifier head always has `2` outputs (`no_drone`, `drone`)

For Hailo deployment, the logical compiled contract recorded in each manifest is:
- input tensor name: `input`
- input shape: `[1, 1, 224, 224]`
- output tensor name: `logits`
- output shape: `[1, 2]`

Important Hailo note:
- Treat that manifest shape as the logical model contract.
- In the Pi-side HailoRT app, always query the actual stream info from the loaded `.hef` and pack buffers in the format HailoRT reports.

## 7. Hailo Artifacts, Size, and What You Actually Need on the Pi

### 7.1 Why the `exports/` directory is so large

The `exports/` tree is a build workspace, not a runtime bundle.

The size is dominated by Hailo compiler intermediates:
- source ONNX exports
- Hailo-augmented ONNX copies
- `native.har`
- `optimized.har`
- `compiled.har`
- parser, optimize, and compiler logs
- `compiler_output/` logs
- command records and manifest files

Current size on this machine:
- `data/experiments/g2_edge_suite/exports`: about `24 GB`
- `data/experiments/g2_edge_suite/strict_far/exports`: about `264 MB`

The biggest contributors right now are the VGG-family workspaces:
- `exports/vgg16`: about `15 GB`
- `exports/vgg13`: about `5.0 GB`

The biggest files are compiler byproducts, not runtime payloads:
- `vgg16_binary.hailo8l.compiled.har`: about `2.95 GB`
- `vgg16_binary.hailo8.compiled.har`: about `2.87 GB`
- `vgg16_binary.hailo8l.optimized.har`: about `2.77 GB`
- `vgg16_binary.hailo8.optimized.har`: about `2.77 GB`
- each `vgg16` ONNX or augmented ONNX copy: about `537 MB`

### 7.2 Files required on the Raspberry Pi

For runtime on the Pi, you do not need the entire `exports/` directory.

Required:
- the target-specific `.hef`
- the matching `.manifest.json`

Recommended:
- keep the manifest beside the `.hef`, because it records:
  - threshold
  - labels
  - preprocessing contract
  - source model id
  - target architecture

Not required on the Pi for inference:
- `.pt`
- `.onnx`
- `.augmented.onnx`
- `.har`
- `.log`
- `.commands.json`
- `compiler_output/`
- calibration `.npy`
- calibration metadata

Practical deployment rule:
- if you are running one Hailo model, ship one `.hef` plus one `.manifest.json`
- for the current G2 candidate sweep, ship `hailo8` artifacts only
- older baseline artifacts still include `hailo8l` outputs, but new candidate evaluation and ranking in this repo are now `hailo8`-first

Current footprint of all compiled HEFs on this machine:
- all `.hef` files together: about `371.09 MB`
- all manifests together: about `68.34 KB`

That is the number to think about for Pi deployment, not the full `24 GB` build tree.

### 7.3 Current compiled HEFs

Compiled HEFs currently present:
- `vgg16_binary.hailo8.hef`
- `vgg16_binary.hailo8l.hef`
- `resnet34_binary.hailo8.hef`
- `resnet50_binary.hailo8.hef`
- `regnet_x_1_6gf_binary.hailo8.hef`
- `resnet18_binary.hailo8.hef`
- `resnet18_binary.hailo8l.hef`
- `mobilenet_v3_small_binary.hailo8.hef`
- `mobilenet_v3_small_binary.hailo8l.hef`
- `shufflenet_v2_x1_0_binary.hailo8.hef`
- `shufflenet_v2_x1_0_binary.hailo8l.hef`
- `mobilenet_v3_small_binary_far003.hailo8.hef`
- `mobilenet_v3_small_binary_far003.hailo8l.hef`

Important caveat:
- a manifest may exist even when the `.hef` does not
- `vgg13_binary` currently has a `hailo8` manifest and intermediate compiler outputs, but no finished `.hef` or validation report yet
- `shufflenet_v2_x1_0_binary_far003` currently has manifest stubs, but no compiled `.hef` yet

## 8. What Was Done to Produce the Hailo Models

The Hailo artifacts were not trained natively in a Hailo-only format. The flow used in this repo is:

1. Train or load a PyTorch checkpoint.
2. Export a static-batch ONNX model with batch size `1`.
3. Build a representative NHWC calibration array from existing feature shards.
4. Run the Hailo Dataflow Compiler flow:
   - `hailo parser onnx`
   - `hailo optimize`
   - `hailo compiler`
5. Validate the compiled target for flat-output, parity, manifest-contract, and latency gates.
6. Save the final `.hef` plus a manifest that records the runtime contract and validation status.

Repo commands used for that flow:

```bash
cd /root/RFBinaryDetect
/root/RFBinaryDetect/.venv/bin/python -m rfbd.cli hailo prepare --target hailo8
/root/RFBinaryDetect/.venv/bin/python -m rfbd.cli hailo compile --target hailo8 --calibration-samples 32 --skip-existing
/root/RFBinaryDetect/.venv/bin/python -m rfbd.cli hailo validate --model-id resnet34_binary --target hailo8
```

Important implementation details:
- ONNX is the compilation interchange format.
- The checkpoints remain the source of threshold, architecture, and preprocessing metadata.
- Calibration data was generated from existing `NPZ` feature shards under the repo dataset tree.
- Host-side preprocessing and host-side postprocessing were kept unchanged.
- New candidate evaluation is `hailo8`-first. Legacy `hailo8l` artifacts still exist for older checkpoints, but they are not the default target for new work.
- The final Hailo path still expects:
  - host preprocessing
  - Hailo execution of the model core
  - host softmax
  - host thresholding

Architecture-specific note:
- `mobilenet_v3_small` needed a Hailo model-script workaround for avgpool quantization during compilation.

## 9. Loading the Original PyTorch Model and Running Inference

Use the same package version family used for training when possible.

### 9.1 Environment setup

```bash
cd /root/RFBinaryDetect
source .venv/bin/activate
pip install -e .
```

### 9.2 Python inference example

```python
from pathlib import Path
import numpy as np
import torch
from rfbd.modeling import create_binary_model
from rfbd.labels import ID_TO_LABEL
from rfbd.features import load_custom_iq, split_segments_1d, samples_per_segment, compute_spec_feature
from rfbd.contracts import FeatureConfig

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

ckpt = torch.load(CKPT_PATH, map_location="cpu")
threshold = float(ckpt["threshold"])
arch = str(ckpt.get("arch", "vgg16"))

model = create_binary_model(arch=arch, pretrained=False, freeze_features=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

iq = load_custom_iq(IQ_PATH)
x = np.real(iq).astype(np.float32, copy=False)
seg_len = samples_per_segment(SAMPLE_RATE, SEGMENT_MS)

feats = []
for seg in split_segments_1d(x, seg_len):
    feats.append(compute_spec_feature(seg, SAMPLE_RATE, CFG))

if not feats:
    raise RuntimeError("No full 20 ms segments available in this capture.")

X = np.stack(feats, axis=0).astype(np.float32)
xb = torch.from_numpy(X).to(device)

with torch.no_grad():
    logits = model(xb)
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

pred_ids = (probs >= threshold).astype(np.int64)
pred_labels = [ID_TO_LABEL[int(v)] for v in pred_ids]

print("segments:", len(probs))
print("arch:", arch)
print("threshold:", threshold)
print("mean_p_drone:", float(np.mean(probs)))
print("drone_segment_ratio:", float(np.mean(pred_ids)))
print("first_10_labels:", pred_labels[:10])

clip_is_drone = float(np.mean(pred_ids)) >= 0.20
print("clip_label:", "drone" if clip_is_drone else "no_drone")
```

## 10. Using the Hailo Models on the Raspberry Pi

This repo currently packages Hailo compile outputs, but it does not yet ship a dedicated `rfbd hailo infer` runtime command. The Pi-side app should therefore use HailoRT directly and follow the same preprocessing and postprocessing contract described here.

### 10.1 Select the correct files

Choose the `.hef` that matches the accelerator target:
- `hailo8` devices should use `*.hailo8.hef`
- `hailo8l` devices should only use older legacy artifacts that were explicitly compiled for `hailo8l`

Carry the matching manifest beside it:
- `*.hailo8.manifest.json`
- `*.hailo8l.manifest.json`

Before selecting a Hailo model, check the manifest:
- `deployable` must be `true` for live use
- if `deployable` is `false`, follow `fallback_recommendation`

In the current repo snapshot:
- `resnet34_binary.hailo8`, `resnet50_binary.hailo8`, and `regnet_x_1_6gf_binary.hailo8` all compile, but remain `deployable=false`
- `vgg13_binary` is `compile_failed` under the corrected contract; `vgg16_binary` is `stale_contract`
- `vgg_small_gap_binary` is the only VGG-style architecture intended for future Hailo-8 runs

The current repo-level fallback for non-deployable Hailo artifacts is:
- `shufflenet_v2_x1_0_binary`

### 10.2 Runtime contract

At runtime on the Pi:
- preprocess raw IQ exactly as in Section 5
- run one PSD frame at a time through the Hailo model
- read the `2` output logits
- apply softmax on the host
- compare `p_drone` to the manifest threshold
- aggregate segment decisions over time exactly as in the PyTorch path

The manifest fields you should actively consume are:
- `hef_path`
- `target_arch`
- `threshold`
- `label_contract`
- `preprocessing_contract`
- `input_name`
- `input_shape`
- `output_name`
- `output_shape`
- `host_input_dtype`
- `host_input_layout`
- `host_input_range`
- `host_input_quantization`
- `hailo_input_normalization`
- `validation_report`
- `deployable`

For Hailo-8 artifacts compiled with the current linear-uint8 contract, including future `vgg_small_gap` artifacts, the intended host-to-HEF contract is:
- host input shape: `224 x 224 x 1`
- host input dtype: `uint8`
- host value range: `[0,255]`
- host layout: `NHWC`
- extra normalization on the Pi: none
- normalization/reconstruction back into the model domain: inside the Hailo graph, as described by the manifest

### 10.3 Minimal Pi-side integration pattern

The exact HailoRT API depends on your Pi application, but the logical flow should be:

```python
import json
import numpy as np

manifest = json.load(open("model.manifest.json", "r", encoding="utf-8"))
threshold = float(manifest["threshold"])

frame = compute_rfbd_feature(raw_iq_segment)      # float32, shape (224, 224)
frame = frame.astype(np.float32, copy=False)

# Query input/output stream info from the loaded HEF with HailoRT.
# Pack the frame in the tensor layout reported by the runtime.
# If the manifest says host_input_dtype=uint8, quantize exactly as the manifest contract describes.
logits = run_hef_inference(frame)                 # returns length-2 logits

exp = np.exp(logits - np.max(logits))
probs = exp / np.sum(exp)
p_drone = float(probs[1])
pred_id = int(p_drone >= threshold)
pred_label = manifest["label_contract"][str(pred_id)]
```

What must stay the same:
- preprocessing
- output interpretation
- thresholding policy
- clip-level aggregation logic

What changes relative to PyTorch inference:
- the model core runs from a `.hef` through HailoRT instead of from a `.pt` through PyTorch

### 10.4 Deployment checklist for the Pi

Before field use, verify:
- the Pi has Hailo runtime support installed
- the `.hef` target matches the hardware, with `hailo8` preferred for the current candidate set
- the manifest matches the `.hef`
- `deployable` is `true`, or you intentionally chose a non-live artifact for lab analysis only
- the app queries stream metadata from the loaded `.hef`
- preprocessing matches Section 5 exactly
- host postprocess uses manifest threshold, not a hard-coded `0.5`
- the first probe frames line up with the repo validation report before trusting live SDR output

For the VGG16 Hailo path, also read:
- [g2_vgg16_hailo_fix_handoff.md](/root/RFBinaryDetect/g2_vgg16_hailo_fix_handoff.md)

## 11. How to Interpret Outputs Correctly

Per segment:
- `p_drone` near `1.0`: strong drone evidence
- `p_drone` near `0.0`: strong background evidence
- decision uses the checkpoint or manifest threshold, not always `0.5`

For low false-alarm deployments:
- the threshold can be intentionally strict
- recall can decrease if the drone signal is weak

For robust system-level decisions, aggregate segment decisions over time:
- use a sliding window such as `1` to `3` seconds
- require persistent evidence before alarming

## 12. Deployment Consistency Checklist

Before trusting field predictions, verify:
- same sample rate (`20,971,520 sps`)
- same segment config (`20 ms`, `nfft=1024`, `noverlap=120`, `224x224`)
- same preprocessing switches (`log_power=True`, `normalize=True`)
- input format decoded correctly as IQ interleaved int16 if using `.s16` or `.bin`
- no clipping at the SDR front-end
- architecture loaded from checkpoint or selected from the correct `.hef`
- threshold loaded from checkpoint or manifest and actually applied

## 13. Common Failure Modes

- using the wrong `.hef` target for the installed accelerator
- using a manifest that does not match the `.hef`
- shipping the full `exports/` tree to the Pi instead of just runtime files
- assuming a manifest implies a compiled `.hef` exists
- assuming a compiled `.hef` implies the model is approved for live deployment
- wrong sample rate passed into preprocessing
- feeding imaginary channel or magnitude instead of real channel
- changing FFT, overlap, scaling, or resize parameters
- very short captures that produce too few segments
- hard-coding a `0.5` threshold instead of using the stored model threshold

## 14. Recommended Next Improvement for Production

If the inference environment differs materially from training:
- collect labeled data in deployment conditions
- regenerate features with the exact same pipeline
- re-train and re-tune the threshold

For the Hailo path specifically, the next practical improvement would be a small Pi-side HailoRT runner in this repo that:
- loads a manifest
- opens the matching `.hef`
- runs one PSD frame or a batch of frames
- returns `p_drone`, `pred_id`, and `pred_label` using the stored threshold

That would remove the last manual integration step between compiled artifacts and field inference.
