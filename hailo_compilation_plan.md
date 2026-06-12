# Multi-Model Hailo Compilation Plan for `RFBinaryDetect`

## Summary

This document defines the repo-level plan for compiling all currently trained deployable `RFBinaryDetect` checkpoints into Hailo deployable artifacts for `Hailo-8` and `Hailo-8L`.

Scope for this plan:
- base checkpoints from `data/experiments/g2_edge_suite/models/`
- strict-FAR checkpoints from `data/experiments/g2_edge_suite/strict_far/models/`
- ONNX as the canonical interchange artifact for Hailo compilation
- preprocessing, softmax, and thresholding remain on the host

Primary outcome:
- produce validated `.hef` artifacts and manifests for all 6 currently trained deployable checkpoints

Defaults for this plan:
- target hardware: compile for both `hailo8` and `hailo8l` unless deployment hardware is fixed in advance
- preferred compile environment: `x86_64` Linux host with Hailo Dataflow Compiler installed
- runtime environment: HailoRT on the target device
- calibration policy: shared calibration dataset is acceptable because all checkpoints use the same preprocessing contract and input size

## Model Inventory

The plan covers these 6 checkpoints.

Base architecture checkpoints:
- `data/experiments/g2_edge_suite/models/vgg16/vgg16_binary.pt`
- `data/experiments/g2_edge_suite/models/resnet18/resnet18_binary.pt`
- `data/experiments/g2_edge_suite/models/mobilenet_v3_small/mobilenet_v3_small_binary.pt`
- `data/experiments/g2_edge_suite/models/shufflenet_v2_x1_0/shufflenet_v2_x1_0_binary.pt`

Strict-FAR checkpoints:
- `data/experiments/g2_edge_suite/strict_far/models/mobilenet_v3_small/mobilenet_v3_small_binary_far003.pt`
- `data/experiments/g2_edge_suite/strict_far/models/shufflenet_v2_x1_0/shufflenet_v2_x1_0_binary_far003.pt`

Compilation policy by architecture:
- parse and translate once per unique architecture: `vgg16`, `resnet18`, `mobilenet_v3_small`, `shufflenet_v2_x1_0`
- resolve export or graph-compatibility issues at the architecture level first
- compile every checkpoint variant separately only after its architecture has proven Hailo-compatible

## Canonical Contracts

Compiled-model input contract:
- input tensor name: `input`
- input tensor shape: `[1, 1, 224, 224]`
- input semantic: single-channel float32 PSD frame produced by existing `RFBinaryDetect` preprocessing

Compiled-model output contract:
- output tensor name: `logits`
- output tensor shape: `[1, 2]`
- output semantic: binary logits for `0=no_drone`, `1=drone`

Runtime behavior that must remain unchanged:
- RF preprocessing stays on the host
- postprocessing stays on the host
- host computes softmax and `p_drone`
- host applies the checkpoint threshold to produce `pred_id` and `pred_label`

Important export constraint:
- the current `rfbd export` ONNX path uses dynamic batch axes
- Hailo compilation should use a static batch-1 source graph
- if the default ONNX export is dynamic, the Hailo flow must include a Hailo-specific static export or a freeze/simplify step before parse/translation begins

## Artifact Layout

This plan aligns artifact locations with the existing repo layout instead of using external host-specific paths.

Base export roots:
- checkpoints: `data/experiments/g2_edge_suite/models/<arch>/`
- ONNX exports: `data/experiments/g2_edge_suite/exports/<arch>/`
- Hailo outputs: `data/experiments/g2_edge_suite/exports/<arch>/hailo/`

Strict-FAR export roots:
- checkpoints: `data/experiments/g2_edge_suite/strict_far/models/<arch>/`
- ONNX exports: `data/experiments/g2_edge_suite/strict_far/exports/<arch>/`
- Hailo outputs: `data/experiments/g2_edge_suite/strict_far/exports/<arch>/hailo/`

Shared calibration root:
- `data/experiments/g2_edge_suite/hailo_calibration/`

Recommended source ONNX naming follows the current exporter:
- base example: `data/experiments/g2_edge_suite/exports/shufflenet_v2_x1_0/shufflenet_v2_x1_0_binary_shufflenet_v2_x1_0.onnx`
- strict-FAR example: `data/experiments/g2_edge_suite/strict_far/exports/mobilenet_v3_small/mobilenet_v3_small_binary_far003_mobilenet_v3_small.onnx`

Recommended HEF naming:
- `<checkpoint_stem>.hailo8.hef`
- `<checkpoint_stem>.hailo8l.hef`

Examples:
- `shufflenet_v2_x1_0_binary.hailo8.hef`
- `shufflenet_v2_x1_0_binary.hailo8l.hef`
- `mobilenet_v3_small_binary_far003.hailo8.hef`

Default target matrix:
- `6 checkpoints x 2 targets = 12 HEF artifacts`

## Workflow

The Hailo compilation workflow is:

`checkpoint (.pt) -> export ONNX -> parse/translate -> optimize/calibrate -> compile -> .hef`

### 1. Checkpoint inventory and metadata capture

For each checkpoint:
- record `arch`, `threshold`, `preprocessing_contract`, and `label_contract` from the checkpoint payload or export summary
- assign a stable `model_id`
- classify the checkpoint as either `base` or `strict_far`

Checkpoint metadata remains the source of truth for:
- model architecture
- decision threshold
- preprocessing contract
- label contract

Checkpoint metadata is not the source of truth for compilation topology:
- ONNX is the Hailo compilation source of truth

### 2. Export canonical ONNX for each checkpoint

For each checkpoint:
- produce an ONNX artifact under the matching export root
- verify the graph still represents one logical input and one logical output
- confirm the logical names remain `input` and `logits`
- convert the graph to a static batch-1 shape if the exporter emitted dynamic batch axes

Fail fast if any checkpoint cannot produce a usable ONNX export.

### 3. Run parse/translation once per architecture

Before compiling every variant:
- run Hailo parse or translation on one representative ONNX per architecture
- verify operator support, preserved tensor semantics, and accepted input layout
- capture the exact command line, full parse log, and short issue summary

Questions this stage must answer:
- are all graph operators supported
- does Hailo accept the single-channel graph as exported
- does the graph need simplification or static-shape freezing first
- do output names or layouts change during translation

### 4. Resolve architecture-level graph issues

Decision order:
1. first choice: change export settings or simplify the ONNX graph while preserving current behavior
2. second choice: keep unsupported work on CPU and compile only the supported inference core if the external model contract remains unchanged
3. last choice: reject the architecture for Hailo deployment if the graph cannot be made compatible without changing the RF pipeline contract

If one checkpoint reveals an architecture-level issue, fix it before compiling other checkpoints that share that architecture.

### 5. Prepare shared calibration inputs

Calibration inputs must:
- match the existing preprocessing contract
- have shape `224x224`
- be single-channel PSD frames
- reflect real or representative `RFBinaryDetect` preprocessing output

The calibration set should include:
- typical background-noise frames
- positive drone frames when available
- enough diversity to avoid overfitting quantization to one domain

Recommended contents under `data/experiments/g2_edge_suite/hailo_calibration/`:
- calibration frames in `.npy` format, or
- a reproducible script that exports calibration frames from existing feature data
- a small metadata note describing frame sources and generation date

### 6. Compile every checkpoint variant for both targets

After architecture-level validation:
- compile each checkpoint separately for `hailo8`
- compile each checkpoint separately for `hailo8l`
- preserve target-specific outputs and logs under the corresponding `hailo/` directory

Each compilation output set should include:
- the target `.hef`
- compiler logs
- parse or translation report
- version capture for compiler and HailoRT
- exact command line used
- a manifest JSON for that compiled artifact

### 7. Capture reproducibility data

For every compiled artifact, record:
- compile host type
- Hailo Dataflow Compiler version
- HailoRT version used for smoke validation
- source checkpoint path
- source ONNX path
- calibration asset source
- target hardware architecture
- any graph simplification or export overrides required

The process should be reproducible by another engineer on a clean machine with the same Hailo toolchain.

## Manifest Schema

Each compiled artifact must have one colocated manifest JSON with at least these fields:

```json
{
  "model_id": "shufflenet_v2_x1_0_binary",
  "arch": "shufflenet_v2_x1_0",
  "variant": "base",
  "checkpoint_path": "data/experiments/g2_edge_suite/models/shufflenet_v2_x1_0/shufflenet_v2_x1_0_binary.pt",
  "source_onnx": "data/experiments/g2_edge_suite/exports/shufflenet_v2_x1_0/shufflenet_v2_x1_0_binary_shufflenet_v2_x1_0.onnx",
  "hef_path": "data/experiments/g2_edge_suite/exports/shufflenet_v2_x1_0/hailo/shufflenet_v2_x1_0_binary.hailo8.hef",
  "target_arch": "hailo8",
  "input_name": "input",
  "input_shape": [1, 1, 224, 224],
  "output_name": "logits",
  "output_shape": [1, 2],
  "threshold": 0.69,
  "preprocessing_contract": {
    "segment_ms": 20,
    "nfft": 1024,
    "noverlap": 120,
    "resize_h": 224,
    "resize_w": 224,
    "log_power": true,
    "normalize": true
  },
  "label_contract": {
    "0": "no_drone",
    "1": "drone"
  },
  "calibration_source": "data/experiments/g2_edge_suite/hailo_calibration/",
  "compiler_version": "record_exact_version",
  "hailort_version": "record_exact_version",
  "postprocess": "host_softmax_threshold"
}
```

The future Hailo runtime integration should consume `hef_path` and manifest metadata explicitly.

It should not:
- guess a `.hef` beside the `.pt` checkpoint
- infer output meaning from positional assumptions alone
- silently select a target-specific artifact based on undocumented device guessing

## Validation and Acceptance

Compilation work is accepted only when all of the following are true.

### Export verification
- all 6 checkpoints export ONNX successfully
- each exported graph preserves the expected logical `input` and `logits`
- Hailo source graphs are static batch-1 after any required freeze or simplify step

### Framework parity
- PyTorch vs ONNX Runtime parity is checked on representative `224x224` single-channel inputs for all 6 checkpoints
- logit drift remains within the agreed tolerance for deployment signoff

### Architecture feasibility
- parse or translation succeeds for all 4 unique architectures
- unsupported operators, layout mismatches, or single-channel issues are documented immediately

### Compilation success
- both `hailo8` and `hailo8l` compilation succeeds for every in-scope checkpoint unless the deployment SKU is fixed in advance and explicitly documented

### Runtime smoke validation
- each `.hef` loads through HailoRT on hardware
- each smoke inference returns one logical output that can be mapped back to the binary-logit contract

### Hailo parity check
- a small representative frame set is run through both ONNX Runtime CPU and Hailo
- any quantization drift is recorded and judged acceptable before deployment

## Failure Scenarios To Document

- ONNX export succeeds but static batch-1 conversion fails
- parse succeeds for one checkpoint but reveals architecture-level incompatibility for sibling variants
- single-channel input is rejected and requires graph adaptation
- graph parse succeeds but full compilation fails
- calibration data is missing, low-quality, or not representative
- output tensor names or layouts differ from the expected `logits` contract
- one target SKU compiles successfully and the other does not

## Assumptions

- This plan covers all 6 currently trained deployable checkpoints, not only the 4 base architectures.
- The existing RF preprocessing contract remains unchanged.
- Threshold loading, softmax, and binary decision logic remain on the host.
- Hailo compilation is an offline build step, not an on-device training or runtime export step.
- The repo is not yet adding Hailo runtime execution code; this document only defines the compilation and handoff workflow.
