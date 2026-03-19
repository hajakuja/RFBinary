# Model Instructions for Future `G2` Deployments

## Purpose and Audience
This document is for the person training, selecting, or adapting the next model to deploy in `G2`.

Its purpose is to describe the deployment constraints that the model must satisfy on the edge target. It is not a training recipe and it is not the source of truth for the exact preprocessing contract.


## Target Runtime Environment
The deployed system is a CPU-only edge pipeline running on:

- `aarch64` Linux
- 4-core `Cortex-A76`
- roughly `15 GiB` system RAM on the box

Important context:

- Inference does not run in isolation. It shares the machine with SDR capture, preprocessing, IPC, logging, and the rest of the long-lived daemon pipeline.
- The current `G2` pipeline runs segment-level inference on fixed `20 ms` RF segments.
- The current input contract is fixed at `20_971_520 sps`, grayscale `224x224` feature frames, and binary output `drone | no_drone`.

## Current Performance Problem
The current binary VGG-style model is too slow for the real deployment workload on this target.

Observed live behavior on the target system:

- capture rate is about `45-49 Hz`
- result rate is about `1 Hz`
- model `latency_us` is roughly `1.1-1.3 s`
- `frame_age_ms` is also roughly `1.2-1.4 s`

This means the system can keep producing detections, but the detections are too stale and too slow for the intended real-time use. Runtime/backend changes alone have not reduced the live latency enough.

## What the Next Model Needs
The next model must be designed for this deployment target first, not for a desktop or GPU workflow.

At a high level, that means:

- substantially lower compute cost on ARM CPU
- materially lower live latency under concurrent system load
- a model/runtime combination that is straightforward to export and run on CPU
- a deployment shape that does not depend on heavy runtime complexity to become usable

The main issue is not only isolated inference speed. The real issue is end-to-end model cost inside a shared edge workload.

## What Must Be Preserved or Explicitly Reworked
Unless a coordinated pipeline change is planned, the next model should preserve the current external contract:

- binary label contract: `no_drone | drone`
- current preprocessing assumptions
- current segment-oriented pipeline behavior

If a future model needs different input size, different preprocessing, different segment duration, or a different aggregation scheme, that must be treated as a full interface change. It should not be treated as a drop-in replacement.

## How Future Models Should Be Evaluated
Future model candidates should be evaluated on the actual target device, not only on a development machine.

Evaluation should include both:

- unloaded per-frame inference timing
- live daemon behavior under real capture workload

An offline benchmark improvement is not enough by itself. A candidate is only successful if it also improves live latency and freshness while keeping detection quality acceptable.

## Out of Scope
This document does not choose a specific architecture.

