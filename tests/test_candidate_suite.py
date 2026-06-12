import csv
import json
from pathlib import Path

from rfbd.candidate_suite import DEFAULT_BASE_ARCHES, LEGACY_REFERENCE_ARCHES, write_hailo8_leaderboard


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _eval_report(*, far: float, recall: float) -> dict:
    return {
        "splits": {
            "custom_bg": {
                "test_metrics": {
                    "far": far,
                    "recall": recall,
                }
            }
        }
    }


def _bench_report(*, loaded_p95_ms: float) -> dict:
    return {
        "rows": [
            {
                "runtime": "pytorch",
                "load_state": "loaded",
                "latency_ms_p50": loaded_p95_ms * 0.9,
                "latency_ms_p95": loaded_p95_ms,
                "throughput_fps": 1000.0 / loaded_p95_ms,
            }
        ]
    }


def _manifest(*, compiled: bool = True) -> dict:
    return {
        "compiled": compiled,
    }


def _validation(
    *,
    deployable: bool,
    flat_output_pass: bool,
    ordering_pass: bool,
    manifest_contract: bool,
    latency_ms: float,
    latency_source: str = "compiler_estimate",
) -> dict:
    return {
        "deployable": deployable,
        "deployment_decision": "accepted_provisional_hailo" if deployable else "rejected_quality",
        "fallback_recommendation": "shufflenet_v2_x1_0_binary",
        "performance": {
            "selected_latency_ms": latency_ms,
            "latency_source": latency_source,
            "latency_is_provisional": latency_source != "hardware_runtime_metrics",
            "hardware_latency_confirmed": latency_source == "hardware_runtime_metrics",
        },
        "gates": {
            "manifest_contract": manifest_contract,
            "flat_output": flat_output_pass,
            "ordering": ordering_pass,
            "runtime_pass": latency_ms < 50.0,
            "overall_pass": deployable,
        },
    }


def test_write_hailo8_leaderboard_prefers_hailo_results_over_cpu_speed(tmp_path: Path):
    suite_root = tmp_path / "suite"
    arches = ("vgg16", "mobilenet_v3_small", "shufflenet_v2_x1_0", "resnet50")

    _write_json(suite_root / "eval/vgg16/domain_holdout_report.json", _eval_report(far=0.20, recall=0.60))
    _write_json(suite_root / "eval/mobilenet_v3_small/domain_holdout_report.json", _eval_report(far=0.20, recall=0.63))
    _write_json(suite_root / "eval/shufflenet_v2_x1_0/domain_holdout_report.json", _eval_report(far=0.20, recall=0.62))
    _write_json(suite_root / "eval/resnet50/domain_holdout_report.json", _eval_report(far=0.20, recall=0.65))

    _write_json(suite_root / "bench/vgg16_runtime.json", _bench_report(loaded_p95_ms=500.0))
    _write_json(suite_root / "bench/mobilenet_v3_small_runtime.json", _bench_report(loaded_p95_ms=5.0))
    _write_json(suite_root / "bench/shufflenet_v2_x1_0_runtime.json", _bench_report(loaded_p95_ms=30.0))
    _write_json(suite_root / "bench/resnet50_runtime.json", _bench_report(loaded_p95_ms=70.0))

    _write_json(suite_root / "exports/vgg16/hailo/vgg16_binary.hailo8.manifest.json", _manifest(compiled=True))
    _write_json(
        suite_root / "exports/vgg16/hailo/vgg16_binary.hailo8.validation.json",
        _validation(
            deployable=False,
            flat_output_pass=False,
            ordering_pass=True,
            manifest_contract=True,
            latency_ms=196.0,
        ),
    )
    _write_json(suite_root / "exports/mobilenet_v3_small/hailo/mobilenet_v3_small_binary.hailo8.manifest.json", _manifest(compiled=True))
    _write_json(
        suite_root / "exports/mobilenet_v3_small/hailo/mobilenet_v3_small_binary.hailo8.validation.json",
        _validation(
            deployable=False,
            flat_output_pass=False,
            ordering_pass=True,
            manifest_contract=True,
            latency_ms=20.0,
        ),
    )
    _write_json(suite_root / "exports/shufflenet_v2_x1_0/hailo/shufflenet_v2_x1_0_binary.hailo8.manifest.json", _manifest(compiled=True))
    _write_json(
        suite_root / "exports/shufflenet_v2_x1_0/hailo/shufflenet_v2_x1_0_binary.hailo8.validation.json",
        _validation(
            deployable=True,
            flat_output_pass=True,
            ordering_pass=True,
            manifest_contract=True,
            latency_ms=35.0,
        ),
    )
    _write_json(suite_root / "exports/resnet50/hailo/resnet50_binary.hailo8.manifest.json", _manifest(compiled=True))
    _write_json(
        suite_root / "exports/resnet50/hailo/resnet50_binary.hailo8.validation.json",
        _validation(
            deployable=True,
            flat_output_pass=True,
            ordering_pass=True,
            manifest_contract=True,
            latency_ms=45.0,
        ),
    )

    leaderboard = write_hailo8_leaderboard(suite_root=suite_root, arches=arches)

    assert leaderboard["strict_candidates"] == ["resnet50", "shufflenet_v2_x1_0"]
    assert [row["arch"] for row in leaderboard["records"][:2]] == ["resnet50", "shufflenet_v2_x1_0"]

    csv_path = suite_root / "leaderboard/leaderboard.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["arch"] == "resnet50"
    assert "hailo_compile_success" in rows[0]
    assert "hailo_validation_success" in rows[0]
    assert "hailo_flat_output_pass" in rows[0]
    assert "hailo_latency_ms" in rows[0]
    assert "hailo_deployable" in rows[0]

    strict_candidates_path = suite_root / "leaderboard/strict_far_candidates.txt"
    assert strict_candidates_path.read_text(encoding="utf-8").splitlines() == ["resnet50", "shufflenet_v2_x1_0"]


def test_default_hailo8_roster_uses_vgg_small_gap_not_legacy_vggs(tmp_path: Path):
    suite_root = tmp_path / "suite"

    assert DEFAULT_BASE_ARCHES == (
        "shufflenet_v2_x1_0",
        "resnet34",
        "resnet50",
        "regnet_x_1_6gf",
        "vgg_small_gap",
        "repvgg_a1_hmz",
        "repvgg_a2_hmz",
    )
    assert "vgg13" in LEGACY_REFERENCE_ARCHES
    assert "vgg16" in LEGACY_REFERENCE_ARCHES

    _write_json(suite_root / "eval/vgg16/domain_holdout_report.json", _eval_report(far=0.20, recall=0.60))
    for arch, recall, latency in (
        ("shufflenet_v2_x1_0", 0.61, 35.0),
        ("resnet34", 0.62, 30.0),
        ("resnet50", 0.63, 45.0),
        ("regnet_x_1_6gf", 0.64, 20.0),
        ("vgg_small_gap", 0.66, 25.0),
        ("repvgg_a1_hmz", 0.65, 18.0),
        ("repvgg_a2_hmz", 0.67, 32.0),
    ):
        _write_json(suite_root / f"eval/{arch}/domain_holdout_report.json", _eval_report(far=0.20, recall=recall))
        _write_json(suite_root / f"bench/{arch}_runtime.json", _bench_report(loaded_p95_ms=200.0))
        _write_json(suite_root / f"exports/{arch}/hailo/{arch}_binary.hailo8.manifest.json", _manifest(compiled=True))
        _write_json(
            suite_root / f"exports/{arch}/hailo/{arch}_binary.hailo8.validation.json",
            _validation(
                deployable=True,
                flat_output_pass=True,
                ordering_pass=True,
                manifest_contract=True,
                latency_ms=latency,
            ),
        )

    leaderboard = write_hailo8_leaderboard(suite_root=suite_root)
    ranked_arches = [row["arch"] for row in leaderboard["records"]]
    vgg_small_gap_row = next(row for row in leaderboard["records"] if row["arch"] == "vgg_small_gap")

    assert "vgg13" not in ranked_arches
    assert "vgg16" not in ranked_arches
    assert "repvgg_a1_hmz" in ranked_arches
    assert "repvgg_a2_hmz" in ranked_arches
    assert vgg_small_gap_row["vgg_family_stop"] is False
    assert leaderboard["strict_candidates"][0] == "repvgg_a2_hmz"
