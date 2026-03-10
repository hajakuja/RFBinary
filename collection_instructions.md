# No-Drone Collection Protocol

This document defines how to collect new **no-drone** raw I/Q data for the `RFBinaryDetect` binary model (`drone` vs `no_drone`).

## 1. Goal

Collect negative examples (`no_drone`) that are representative of deployment conditions and diverse enough to reduce false alarms.

## 2. Required Labels

Use this label contract everywhere:

- `0 = no_drone`
- `1 = drone`

All captures in this protocol are for `label = no_drone`.

## 3. RF Capture Settings

## 3.1 Center Frequencies (fixed dual-band)

- `2.437e9` Hz (2.4 GHz band)
- `5.735e9` Hz (5.8 GHz band)

## 3.2 Sample Rate

- Use your **deployment SDR sample rate** as primary choice.
- If not finalized, use default fallback: `20_971_520` samples/sec.

## 3.3 Gain Sweep

Collect each frequency at 3 gain levels:

- `low`
- `mid`
- `high`

Map these to your SDR hardware gain values (for example `35/50/65 dB` equivalent).

## 3.4 Segment Target for Feature Extraction

The feature pipeline assumes:

- segment length: `20 ms`
- spectrogram `nfft = 1024`
- `noverlap = 120`
- resize to `224 x 224`

You do not need to segment manually during capture. Just record continuous raw I/Q files.

## 4. Session Design

Minimum recommended data collection:

- at least `10` sessions
- each session records **both frequencies** and **all 3 gains**
- each condition records at least `60 seconds`

Coverage requirements:

- indoor and outdoor
- low RF occupancy and high RF occupancy
- different times of day
- stationary and moving receiver placement when possible

Recommended total no-drone data: `>= 3 hours`.

## 5. Quality Rules

- No known active drone transmitter during capture.
- If uncertain activity occurs, mark interval and exclude from training.
- Keep notes for abnormal events (microwave bursts, strong nearby emitters, SDR clipping).
- Avoid saturated captures; reduce gain if clipping is observed.

## 6. File Naming

Use deterministic names that encode key metadata:

`{capture_id}_{timestamp}_{center_freq_hz}Hz_{sample_rate_sps}sps_{gain_db}g_{session_id}.<ext>`

Example:

`bg00017_20260305T213010Z_5735000000Hz_20971520sps_50g_s03.s16`

## 7. Manifest Requirement (mandatory)

Each captured file must have one manifest row in CSV with these columns:

- `capture_id`
- `file_path`
- `label`
- `source_domain`
- `center_freq_hz`
- `sample_rate_sps`
- `gain_db`
- `session_id`
- `timestamp`
- `environment`

Recommended values for this protocol:

- `label = no_drone`
- `source_domain = custom_bg`
- `environment` examples: `indoor_quiet`, `indoor_busy`, `outdoor_quiet`, `outdoor_busy`

## 8. Manifest Example

```csv
capture_id,file_path,label,source_domain,center_freq_hz,sample_rate_sps,gain_db,session_id,timestamp,environment
bg00017,/data/rf/bg00017_20260305T213010Z_5735000000Hz_20971520sps_50g_s03.s16,no_drone,custom_bg,5735000000,20971520,50,s03,2026-03-05T21:30:10Z,outdoor_busy
bg00018,/data/rf/bg00018_20260305T213210Z_2437000000Hz_20971520sps_35g_s03.s16,no_drone,custom_bg,2437000000,20971520,35,s03,2026-03-05T21:32:10Z,outdoor_busy
```

## 9. Operator Checklist

Before recording:

- confirm center frequency and sample rate
- confirm gain setting
- confirm clock/time sync on host
- confirm storage space for raw I/Q

During recording:

- monitor clipping/saturation
- log unexpected emitters or uncertain events

After recording:

- verify file integrity and duration
- append manifest row for every file
- back up raw files and manifest

## 10. Notes for Drone-Positive Collections

If you later collect `drone` data with same hardware, keep the same:

- sample rate
- segment settings
- file naming and manifest schema

This keeps domain shift low and improves binary model robustness.
