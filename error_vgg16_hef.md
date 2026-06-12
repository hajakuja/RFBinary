# VGG16 Hailo HEF Failure Note

## Summary

The compiled `vgg16_binary.hailo8.hef` artifact is not behaving like a valid detector model in `G2`.

Two major problems were observed:

- the model output appears effectively constant across very different inputs
- the runtime is far too slow for the live detector path

Because of that, the current VGG16 Hailo artifact should be treated as invalid for deployment until it is recompiled and revalidated.

## Runtime Environment

This was observed on the current `G2` edge target with:

- Raspberry Pi class ARM64 deployment target
- Hailo-8 accelerator over PCIe
- HailoRT `4.20.0`
- `G2` Python worker using the native `hailort` backend

The detector contract remains:

- input feature shape: `224 x 224`
- single-channel PSD image
- binary output: `no_drone` or `drone`
- live detector cadence target: about `50 Hz`

## What Happened

When the VGG16 Hailo model was selected, the live detector output stopped responding meaningfully to the scene.

Observed symptoms:

- `p_drone` stayed effectively fixed at about `0.76639`
- the class decision did not meaningfully change between drone and no-drone conditions
- live latency was about `196 ms` per frame
- result throughput dropped to about `5 Hz`

That is already enough to reject the model for deployment, but the more important finding is that the artifact itself appears flat.

## Direct HEF Validation Result

To isolate the issue from the SDR pipeline, the compiled `vgg16_binary.hailo8.hef` was run directly through HailoRT on synthetic inputs that should have produced different activations:

- all zeros
- all `255`
- mid-gray
- checkerboard
- random input #1
- random input #2

The HEF returned the same logits every time:

```text
[-0.5822916626930237, 0.6057711839675903]
```

This corresponds to:

```text
p_drone = 0.7663944363594055
```

That strongly indicates the failure is in the compiled model artifact or its compilation path, not in live RF capture or the `G2` postprocessing logic.

## Why This Is Not Just A Threshold Bug

There is a threshold nuance, but it is not the root cause.

- the VGG16 manifest threshold is `0.77`
- the active config override used `decision_threshold = 0.69`

With the constant output around `0.76639`:

- at `0.69`, the model always predicts `drone`
- at `0.77`, the model would almost always predict `no_drone`

Changing the threshold only flips which constant answer you get. It does not fix the fact that the logits are not changing with the input.

## Likely Failure Modes

The most likely explanations are:

- bad Hailo compilation output for this model
- calibration data or quantization setup that collapsed the signal range
- export or graph transformation issue before Hailo compilation
- mismatch between the intended logical input contract and what the compiled graph actually expects
- VGG16 being a poor fit for the chosen edge compilation/runtime path without additional graph or preprocessing adjustments

Less likely causes, because they were tested around directly:

- `G2` SDR capture path
- `G2` softmax or threshold logic
- live-only timing or buffering behavior

## Performance Problem

Even if the outputs were not flat, the current VGG16 Hailo runtime is still too slow for the detector path.

Observed runtime:

- about `196 ms` per frame
- about `5 Hz` detector result rate

Required runtime direction:

- we need to get under `50 ms` per inference

That target matters because the detector path is designed around a live cadence near `50 Hz`, and a model running around `196 ms` introduces backlog, stale decisions, and degraded responsiveness.

## Deployment Requirement Going Forward

The next accelerator-ready model needs to satisfy both of these constraints at the same time:

- inference runtime under `50 ms`
- detection quality comparable to the original VGG-level detector performance

A faster model is not enough if accuracy drops too far. A more accurate model is not enough if it cannot keep up with the live detector path.

In practice, the acceptance bar should be:

- no flat-output behavior
- stable, input-responsive logits
- sub-`50 ms` inference on the target device
- detection behavior that remains close to the VGG baseline in real RF conditions

## Recommended Next Steps

- do not deploy the current `vgg16_binary.hailo8.hef`
- keep using the working ShuffleNet Hailo artifact for live accelerator testing
- revalidate VGG from the source model outward:
  - compare PyTorch output vs ONNX output vs compiled Hailo output on the same saved frames
  - inspect calibration inputs used during compilation
  - confirm the actual compiled input/output tensor contract
  - recompile only after parity is understood

## Bottom Line

The current VGG16 Hailo artifact is failing in two ways:

- it behaves like a constant-output model
- it is too slow for the live detector path

Until both issues are resolved, it should not be considered a valid `G2` deployment model.
