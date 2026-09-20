"""Compare vLLM main, PR #52412, and a scoped local candidate fix."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCHEDULER_PATH = Path("vllm/v1/core/sched/scheduler.py")
GUARD = (
    b"                self.scheduler_config.disable_chunked_mm_input\n"
    b"                # `<=`, not `<`:"
)
SCOPED_GUARD = (
    b"                self.scheduler_config.disable_chunked_mm_input\n"
    b"                and not self.is_encoder_decoder\n"
    b"                # `<=`, not `<`:"
)
OFFICIAL_TESTS = [
    "tests/v1/core/test_scheduler.py::test_cross_attn_blocks_not_over_allocated",
    "tests/v1/core/test_scheduler.py::test_cross_attn_blocks_not_under_allocated",
]


def run(
    name: str,
    command: list[str],
    checkout: Path,
    project: Path,
    output: Path,
) -> tuple[int, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (str(checkout), str(project), environment.get("PYTHONPATH", ""))
        if path
    )
    process = subprocess.run(
        command,
        cwd=checkout,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
    )
    output_text = process.stdout + process.stderr
    home = str(Path.home())
    recorded_command = " ".join(command).replace(home, "~")
    recorded_output = output_text.replace(home, "~")
    (output / f"{name}.log").write_text(
        f"exit_code={process.returncode}\ncommand={recorded_command}\n\n"
        f"{recorded_output}"
    )
    print(f"{name}: exit={process.returncode}", flush=True)
    return process.returncode, output_text


def git_head(checkout: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def check_expected(results: dict[str, tuple[int, str]], official: bool) -> None:
    expected = {
        "main_contract": (1, "test_multimodal_item_is_not_split[400-500-0]"),
        "pr_contract": (1, "test_encoder_decoder_input_does_not_block_decoder_prefill"),
        "candidate_contract": (0, "4 passed"),
    }
    if official:
        expected.update(
            {
                "pr_official_cross_attn": (1, "2 failed"),
                "candidate_official_cross_attn": (0, "2 passed"),
            }
        )
    mismatches = [
        name
        for name, (code, marker) in expected.items()
        if results[name][0] != code or marker not in results[name][1]
    ]
    if mismatches:
        raise RuntimeError(
            f"Unexpected result in {', '.join(mismatches)}; inspect its log. "
            "Do not interpret a dependency or network error as a scheduler result."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True, help="vLLM main checkout")
    parser.add_argument("--pr", type=Path, required=True, help="PR #52412 worktree")
    parser.add_argument("--output", type=Path, help="result directory")
    parser.add_argument(
        "--official-tests",
        action="store_true",
        help="also run two upstream cross-attention tests (may need network)",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    main_checkout = args.main.resolve()
    pr_checkout = args.pr.resolve()
    scheduler_file = pr_checkout / SCHEDULER_PATH
    tests = project / "tests/test_scheduling_contract.py"
    if not scheduler_file.is_file() or not (main_checkout / SCHEDULER_PATH).is_file():
        parser.error("both --main and --pr must point to vLLM source checkouts")
    output = (args.output or project / "results/latest").resolve()
    output.mkdir(parents=True, exist_ok=True)

    pr_head = subprocess.check_output(
        ["git", "-C", str(pr_checkout), "show", f"HEAD:{SCHEDULER_PATH}"],
    )
    if pr_head.count(GUARD) != 1:
        parser.error("the PR HEAD no longer contains the expected guard")
    candidate = pr_head.replace(GUARD, SCOPED_GUARD)
    original_worktree = scheduler_file.read_bytes()
    if original_worktree not in (pr_head, candidate):
        parser.error("PR scheduler has unrelated edits; refusing to overwrite it")

    python = sys.executable
    contract = [python, "-m", "pytest", str(tests), "-q", "--tb=short"]
    official = [python, "-m", "pytest", *OFFICIAL_TESTS, "-q", "--tb=short"]
    results: dict[str, tuple[int, str]] = {}

    try:
        results["main_contract"] = run(
            "main_contract", contract, main_checkout, project, output
        )
        scheduler_file.write_bytes(pr_head)
        results["pr_contract"] = run(
            "pr_contract", contract, pr_checkout, project, output
        )
        if args.official_tests:
            results["pr_official_cross_attn"] = run(
                "pr_official_cross_attn", official, pr_checkout, project, output
            )
        scheduler_file.write_bytes(candidate)
        results["candidate_contract"] = run(
            "candidate_contract", contract, pr_checkout, project, output
        )
        if args.official_tests:
            results["candidate_official_cross_attn"] = run(
                "candidate_official_cross_attn",
                official,
                pr_checkout,
                project,
                output,
            )
    finally:
        scheduler_file.write_bytes(original_worktree)

    rows = [
        "# Scheduler boundary comparison",
        "",
        f"Main commit: `{git_head(main_checkout)}`. "
        f"PR commit: `{git_head(pr_checkout)}`. Python: `{sys.version.split()[0]}`.",
        "",
        "| Variant | Contract tests | Upstream cross-attention tests |",
        "| --- | --- | --- |",
        "| Main (`<`) | 1 failed, 3 passed: MM item split | not run here |",
        "| PR #52412 (`<=`) | 1 failed, 3 passed: encoder-decoder blocked | "
        + ("2 failed" if args.official_tests else "not run")
        + " |",
        "| Scoped candidate (`<=` only outside encoder-decoder) | 4 passed | "
        + ("2 passed" if args.official_tests else "not run")
        + " |",
        "",
        "Each cell is backed by the corresponding `.log` in this directory. "
        "These are scheduler correctness checks, not speed or model-quality metrics.",
        "The script restored the PR worktree's original file bytes after testing.",
        "",
    ]
    check_expected(results, args.official_tests)
    (output / "summary.md").write_text("\n".join(rows))
    print(f"Verified comparison saved to {output / 'summary.md'}")


if __name__ == "__main__":
    main()
