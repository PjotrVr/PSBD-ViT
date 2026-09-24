"""The multi-source TaCT job generator, checked without a cluster or a dataset.

k must be the smallest source pool holding 5 times the poison, the emitted
commands must parse under their own CLIs, every downstream job must wait on its
own training job, and the gate must refuse exactly what the coverage ledger
would keep off the panel.
"""

import json

import pytest

import pbs.generate_tact_multisource_jobs as generator
from attacks import apply_config_overrides, default_config
from cli.train_backdoor import parse_attack_overrides


def test_k_is_the_smallest_pool_holding_5_times_the_poison():
    uniform = [label for label in range(100) for _ in range(500)]  # CIFAR-100 shape
    assert generator.source_class_count(uniform, 0.01, target_label=0) == 5
    assert generator.source_class_count(uniform, 0.05, target_label=0) == 25

    # Unequal classes overshoot rather than fall short.
    unequal = [0] * 10 + [1] * 30 + [2] * 30 + [3] * 30
    assert generator.source_class_count(unequal, 0.1, target_label=0) == 2


def test_folder_and_override_follow_the_template():
    assert generator.run_folder("cifar100", 0.01, 5) == "vit_cifar100_tact_0_01_src5"
    assert generator.run_folder("cifar10", 0.1, 5) == "vit_cifar10_tact_0_1_src5"

    override = generator.source_override(3)
    config = apply_config_overrides(
        default_config("tact"),
        parse_attack_overrides([f"source_classes={override}"]),
    )
    assert config.source_classes == (1, 2, 3)


def test_emitted_training_command_parses_and_a_bad_flag_is_refused():
    run = {
        "dataset": "tiny",
        "poison_rate": 0.05,
        "source_classes": generator.source_override(50),
        "folder": "vit_tiny_tact_0_05_src50",
    }
    body = generator.train_body(run)

    assert len(generator.command_blocks(body, "cli.train_backdoor")) == 1
    generator.verify_script(body)
    with pytest.raises(SystemExit):
        generator.verify_script(body.replace("--seed", "--sed"))


def test_each_downstream_job_waits_on_its_own_training_job():
    script = generator.submit_script(
        [("tms_t1", "/a/tms_t1.pbs"), ("tms_t2", "/a/tms_t2.pbs")],
        [
            ("tms_d1", "/a/tms_d1.pbs", "tms_t1"),
            ("tms_d2", "/a/tms_d2.pbs", "tms_t2"),
            ("tms_d3", "/a/tms_d3.pbs", None),
        ],
    )

    assert "TMS_T1=$(qsub /a/tms_t1.pbs)" in script
    assert "qsub -W depend=afterok:${TMS_T1} /a/tms_d1.pbs" in script
    assert "qsub -W depend=afterok:${TMS_T2} /a/tms_d2.pbs" in script
    assert "$(qsub /a/tms_d3.pbs)" in script


def write_sidecar(root, folder: str, sidecar: dict) -> None:
    (root / folder).mkdir()
    (root / folder / "args.json").write_text(json.dumps(sidecar))


@pytest.mark.parametrize(
    ("asr", "clean_accuracy", "refused"),
    [(0.99, 0.94, False), (0.60, 0.94, True), (0.99, 0.10, True)],
)
def test_the_gate_refuses_below_the_bar_and_diverged(
    tmp_path, asr, clean_accuracy, refused
):
    checkpoints, results = tmp_path / "checkpoints", tmp_path / "results"
    checkpoints.mkdir()
    results.mkdir()
    write_sidecar(checkpoints, "vit_cifar10_benign", {"clean_accuracy": 0.95})
    write_sidecar(
        checkpoints,
        "vit_cifar10_tact_0_1_src5",
        {
            "dataset": "cifar10",
            "attack": "tact",
            "asr": asr,
            "clean_accuracy": clean_accuracy,
        },
    )
    declaration = {
        "asr_bar": 0.85,
        "benign_reference": {"cifar10": "vit_cifar10_benign"},
    }

    verdict = generator.clearing_verdict(
        "vit_cifar10_tact_0_1_src5", str(checkpoints), str(results), declaration
    )

    assert (verdict is not None) == refused
