import os
import time

import numpy as np
import pytest
from scipy.ndimage import zoom

from rfbd.features import compute_spec_feature
from rfbd.contracts import FeatureConfig


@pytest.mark.skipif(os.getenv("RFBD_RUN_PERF", "0") != "1", reason="Set RFBD_RUN_PERF=1 to run optional perf test")
def test_fast_resize_path_is_3x_faster_than_naive_zoom():
    cfg = FeatureConfig(segment_ms=20, nfft=1024, noverlap=120, resize_h=224, resize_w=224)
    x = np.random.randn(800_000).astype(np.float32)

    t0 = time.time()
    _ = compute_spec_feature(x, sample_rate_sps=40_000_000.0, cfg=cfg)
    fast = time.time() - t0

    from scipy import signal

    _, _, sxx = signal.spectrogram(
        x,
        fs=40_000_000.0,
        window="hann",
        nperseg=cfg.nfft,
        noverlap=cfg.noverlap,
        scaling="density",
        mode="psd",
    )
    t1 = time.time()
    _ = zoom(sxx.astype(np.float32), (224 / sxx.shape[0], 224 / sxx.shape[1]), order=1)
    slow = time.time() - t1

    assert slow / max(fast, 1e-9) >= 3.0
