# G2 VGG16 Hailo Fix Handoff

This note is the runtime-side handoff for using rebuilt `vgg16` Hailo artifacts from `RFBinaryDetect`.

## Files G2 Must Consume

For each selected target, load:

- the target-specific `.hef`
- the matching `.manifest.json`

Recommended runtime pair layout:

- `vgg16_binary.hailo8.hef` + `vgg16_binary.hailo8.manifest.json`
- `vgg16_binary.hailo8l.hef` + `vgg16_binary.hailo8l.manifest.json`

Do not infer runtime behavior from filenames alone. The manifest is the authoritative contract.

## Manifest Fields G2 Must Read

Required fields:

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
- `deployment_decision`
- `fallback_recommendation`

If any of these are missing for `vgg16`, treat the artifact as invalid for live use.

## Required Host Input Contract

For rebuilt `vgg16`, the intended host-to-HEF contract is:

- single-channel `224x224`
- host dtype: `uint8`
- host layout: `NHWC`
- host value range: `[0,255]`
- extra normalization on the host: none

Important:

- `G2` must not apply any additional scaling, centering, or normalization beyond what the manifest contract says.
- The Hailo graph is expected to reconstruct the original model-domain scale internally via the normalization layer recorded in `hailo_input_normalization`.

## Runtime Bring-Up Checklist

Before trusting live SDR results:

- query the actual HailoRT stream metadata from the loaded `.hef`
- confirm the runtime stream dtype/layout matches the host buffer you are about to submit
- confirm the model output is still a 2-logit binary head
- confirm `deployable == true`

If `deployable == false`:

- do not enable `vgg16` for the live detector path
- use the manifest `fallback_recommendation`

Current expected fallback:

- `shufflenet_v2_x1_0_binary`

## Probe Comparison Requirement

The first bring-up run on `G2` must be checked against the repo validation report referenced by `validation_report`.

Minimum requirement:

- run the saved probe frames or equivalent first-frame checks through the `G2` HailoRT path
- compare the first few logits and `p_drone` values to the `RFBinaryDetect` validation report
- fail bring-up if outputs collapse to near-identical logits across clearly different probes

The original failure mode for the bad `vgg16` artifact was flat output across:

- zeros
- full-scale input
- checkerboard
- random probes

That exact failure should be treated as a hard stop, not a threshold-tuning problem.

## Latency Requirement

Compile-time estimates are not enough.

`G2` must measure real device latency separately and report it back via the runtime metrics JSON used by:

```bash
cd /root/RFBinaryDetect
/root/RFBinaryDetect/.venv/bin/python -m rfbd.cli hailo validate \
  --model-id vgg16_binary \
  --target hailo8 \
  --runtime-metrics-json /path/to/runtime_metrics.json \
  --hardware-results-json /path/to/hardware_results.json
```

Target requirement:

- live inference latency must remain under `50 ms` per frame

If runtime exceeds that threshold, `vgg16` remains non-deployable even if the logits look correct.

## Recommended Runtime Metrics JSON Shape

Use a simple JSON object with one or more of:

```json
{
  "mean_latency_ms": 41.2,
  "p95_latency_ms": 46.8,
  "latency_ms": 41.2
}
```

The validator uses the worst available reported latency value.

## Recommended Hardware Results JSON Shape

When sending real HEF outputs back for validation, use:

```json
{
  "synthetic": {
    "logits": [[-0.1, 0.2], [-0.3, 0.7]]
  },
  "real": {
    "logits": [[-0.4, 0.9], [-0.2, 0.1]]
  }
}
```

The `synthetic.logits` count must match the synthetic probe count in the validation report.
The `real.logits` count must match the real probe count in the validation report.

## Decision Rule

`G2` should only enable `vgg16` in the live accelerator path when all of the following are true:

- the `.hef` and manifest match
- the runtime buffer contract matches the manifest
- the first probe comparisons match the validation report closely enough
- `deployable == true`
- measured runtime is under `50 ms`

If any of those checks fail, keep using the fallback model instead of forcing `vgg16` into production.
