"""The generators of the evidence-surplus, final-method attacker and SIG k=20 jobs.

The expensive failures this prevents: a job whose command a parser rejects after
hours in the queue, a walltime below its own estimate, a SIG rerun that repeats a
placement the login runner already redid, and a k=20 sweep rerun over a complete
cache.
"""

import json

import pytest

import pbs.generate_adaptive_final_jobs as adaptive
import pbs.generate_evidence_surplus_jobs as surplus
import pbs.generate_sig_k20_jobs as sig_k20

pytestmark = pytest.mark.fast


def placements() -> list[dict]:
    loaded = surplus.load_basis_placements(
        surplus.DECLARATION, surplus.HEADLINE_PLACEMENT_IDS
    )
    return loaded


def test_every_run_sits_in_1_job_and_every_command_parses():
    run_ids = sorted(run_id for _, ids in surplus.JOBS for run_id in ids)
    assert run_ids == sorted(run["id"] for run in surplus.RUNS)

    for run in surplus.RUNS:
        surplus.validate_calls(surplus.run_calls(run, placements()))


def test_backdoor_runs_record_telemetry_and_snapshots_and_benign_runs_probe_badnets():
    for run in surplus.RUNS:
        module, argv = surplus.train_argv(run)
        if run["attack"] == "benign":
            assert module == "cli.train_benign"
            sweep_calls = surplus.run_calls(run, placements())[1:-1]
            assert all("--probe-attack" in argv for _, argv in sweep_calls)
            continue
        assert "--telemetry" in argv and "--save-every-epoch" in argv
        assert ("--record-sample-loss" in argv) == bool(run.get("dynamics"))


def test_every_job_asks_for_at_most_12_hours_and_at_least_its_estimate():
    runs_by_id = {run["id"]: run for run in surplus.RUNS}
    for name, ids in surplus.JOBS:
        _, estimate = surplus.render_job(
            name, [runs_by_id[i] for i in ids], placements(), "/base", "batch"
        )
        hours, minutes, _ = surplus.requested_walltime(estimate).split(":")
        requested = int(hours) * 60 + int(minutes)
        assert estimate * 1.25 <= requested <= 12 * 60


def test_requested_walltime_adds_the_margin_and_rounds_up_to_a_quarter_hour():
    assert surplus.requested_walltime(60) == "01:15:00"
    assert surplus.requested_walltime(61) == "01:30:00"


def test_the_adaptive_attacker_trains_against_both_probes_of_the_final_method():
    metadata = {
        "folder": "vit_gtsrb_badnet_a2o_0_1",
        "dataset": "gtsrb",
        "attack": "badnet_a2o",
        "poison_rate": 0.1,
        "target_label": 0,
        "architecture": "vit",
        "epochs": 15,
        "seed": 0,
    }
    sweep_placements = surplus.load_basis_placements(
        surplus.DECLARATION, adaptive.SWEEP_PLACEMENT_IDS
    )

    calls = adaptive.run_calls(metadata, sweep_placements)
    surplus.validate_calls(calls)

    train_arguments = calls[0][1]
    assert "pre_residual:dropout:5-8" in train_arguments
    assert calls[-2][1][calls[-2][1].index("--block-range") + 1 :][:2] == ["5", "8"]


def test_sig_work_skips_what_the_login_runner_redid(tmp_path):
    commands = tmp_path / "commands.txt"
    commands.write_text(
        "post_residual  python -m cli.sweep --checkpoint-folder swin_cifar10_sig_0_1 "
        "--position post_residual --operator dropout --rates 0.1 0.2\n"
        "before_attention_norm_token_mask  python -m cli.sweep --checkpoint-folder "
        "swin_cifar10_sig_0_1 --position before_attention_norm --operator token_mask "
        "--rates 0.5\n"
        "before_attention_norm_token_mask  python -m cli.sweep --checkpoint-folder "
        "swin_cifar10_sig_0_05 --position before_attention_norm --operator token_mask "
        "--rates 0.5\n"
        "before_mlp_norm  python -m cli.sweep --checkpoint-folder vit_cifar10_sig_0_1 "
        "--position before_mlp_norm --operator dropout --rates 0.1\n"
    )
    summary = tmp_path / "summary.txt"
    summary.write_text(
        "Tue Sep 29 19:18:54 CEST 2026 swin_cifar10_sig_0_1 "
        "before_attention_norm_token_mask exit 0 673s\n"
        "Tue Sep 29 19:20:00 CEST 2026 swin_cifar10_sig_0_1 post_residual exit 1 5s\n"
    )

    work = sig_k20.ordered_sig_work(
        sig_k20.read_sig_commands(str(commands)), [0.05, 0.1]
    )
    done = sig_k20.redone_on_record(str(summary))

    assert [item[:2] for item in work] == [
        ("swin_cifar10_sig_0_1", "before_attention_norm_token_mask"),
        ("swin_cifar10_sig_0_1", "post_residual"),
        ("swin_cifar10_sig_0_05", "before_attention_norm_token_mask"),
        ("vit_cifar10_sig_0_1", "before_mlp_norm"),
    ]
    assert work[2][2][-2:] == ["0.05", "0.1"]
    assert done == {("swin_cifar10_sig_0_1", "before_attention_norm_token_mask")}
    assert sig_k20.rate_count(work[1][2]) == 2


def test_pending_k20_leaves_out_complete_caches(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            [
                {"folder": "vit_a", "tm": 0.5, "rd": 0.07},
                {"folder": "vit_b", "tm": None, "rd": 0.09},
            ]
        )
    )
    cache = tmp_path / "results" / "vit_a" / "psbd" / "post_residual_k20"
    cache.mkdir(parents=True)
    for split in ("validation", "clean", "backdoor"):
        (cache / f"rate_0_07_{split}.pt").write_text("")

    pending = sig_k20.pending_k20(str(plan), str(tmp_path / "results"))

    assert pending == [("vit_a", "tm", 0.5), ("vit_b", "rd", 0.09)]
