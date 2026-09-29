# Recounts the integrity rows Q9, Q10, Q11 and Q14 of docs/open-questions.md on the tree
# as it stands, so the document states the current count rather than the 2026-09-23 one.
import glob
import json
import re
import subprocess

args_paths = glob.glob("checkpoints/*/args.json")
commits = [json.load(open(path)).get("git_commit") for path in args_paths]
print("Q9 args.json", len(args_paths), "null", sum(c is None for c in commits),
      "dirty", sum(bool(c) and c.endswith("-dirty") for c in commits))

run_paths = glob.glob("results/*/psbd/**/run_*.json", recursive=True)
run_commits = []
for path in run_paths:
    try:
        run_commits.append(json.load(open(path)).get("git_commit"))
    except (json.JSONDecodeError, OSError):
        run_commits.append("unreadable")
print("Q10 run records", len(run_paths), "dirty",
      sum(bool(c) and str(c).endswith("-dirty") for c in run_commits))

reachable = subprocess.run(["git", "branch", "-a", "--contains", "f402dba2"],
                           capture_output=True, text=True)
print("Q11 f402dba2 branches:", repr(reachable.stdout.strip()), reachable.returncode,
      reachable.stderr.strip()[:80])
referencing = subprocess.run(["grep", "-rl", "f402dba2", "results/swin_cifar100_badnet_a2o_0_1/"],
                             capture_output=True, text=True).stdout.split()
print("Q11 records referencing it", len(referencing))

tables = glob.glob("paper/tables/*.tex") + glob.glob("paper/figures/*.json")
stamps = []
for path in tables:
    head = open(path, errors="ignore").read(3000)
    match = re.search(r"commit ([0-9a-f]{7,40}(?:-dirty)?)", head) or re.search(
        r'"git_commit": "([^"]+)"', head)
    stamps.append(match.group(1) if match else None)
stamped = [s for s in stamps if s]
print("Q14 artifacts", len(tables), "stamped", len(stamped), "dirty",
      sum(s.endswith("-dirty") for s in stamped), "distinct commits",
      len({s.replace("-dirty", "") for s in stamped}))
