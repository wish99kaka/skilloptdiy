"""Command line interface for TextSkill Optimizer."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
from pathlib import Path
from types import ModuleType

from .command_editor import CommandEditorConfig, CommandSkillEditor
from .executive_optimizer import ExecutiveOptimizerConfig, ExecutiveSkillOptimizer
from .io import load_tasks_jsonl, load_text, write_json, write_text
from .lr_profiles import get_lr_profile, profile_names
from .optimizer import OptimizerConfig, SkillOptimizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="textskill",
        description="Optimize reusable natural-language skill documents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Score a skill on a task set")
    add_common_runtime_args(eval_parser)
    eval_parser.add_argument("--tasks", required=True, help="JSONL task file")

    opt_parser = subparsers.add_parser("optimize", help="Run validation-gated optimization")
    add_common_runtime_args(opt_parser)
    opt_parser.add_argument("--train", required=True, help="Training JSONL task file")
    opt_parser.add_argument("--valid", required=True, help="Validation JSONL task file")
    opt_parser.add_argument(
        "--holdout",
        default=None,
        help="Optional holdout JSONL task file. Evaluated after optimization only.",
    )
    opt_parser.add_argument("--epochs", type=int, default=3)
    opt_parser.add_argument(
        "--protocol",
        choices=("legacy", "executive"),
        default=os.environ.get("TEXTSKILL_PROTOCOL", "legacy"),
        help="Use executive for batched atomic-edit optimization with slow/meta updates.",
    )
    opt_parser.add_argument("--out", required=True, help="Directory for run artifacts")
    opt_parser.add_argument(
        "--meta-skill",
        default=os.environ.get("TEXTSKILL_META_SKILL"),
        help="Optional optimizer-side meta skill markdown file.",
    )
    opt_parser.add_argument(
        "--lr-profile",
        choices=profile_names(),
        default=os.environ.get("TEXTSKILL_LR_PROFILE"),
        help=(
            "Named textual learning-rate budget profile. "
            "Explicit --max-* values override the profile."
        ),
    )
    opt_parser.add_argument(
        "--max-skill-chars",
        type=int,
        default=None,
        help="Learning-rate budget: maximum candidate skill length.",
    )
    opt_parser.add_argument(
        "--max-skill-delta-chars",
        type=int,
        default=None,
        help="Learning-rate budget: maximum absolute character delta per candidate.",
    )
    opt_parser.add_argument(
        "--max-added-bullet-lines",
        type=int,
        default=None,
        help="Learning-rate budget: maximum new bullet lines per candidate.",
    )
    opt_parser.add_argument(
        "--rejected-buffer-limit",
        type=int,
        default=int(os.environ.get("TEXTSKILL_REJECTED_BUFFER_LIMIT", "20")),
        help="Maximum rejected proposals passed to the editor.",
    )
    opt_parser.add_argument("--rollout-batch-size", type=int, default=40)
    opt_parser.add_argument("--reflection-minibatch-size", type=int, default=8)
    opt_parser.add_argument("--text-learning-rate", type=int, default=4)
    opt_parser.add_argument("--text-learning-rate-floor", type=int, default=2)
    opt_parser.add_argument(
        "--text-learning-rate-schedule",
        choices=("constant", "linear", "cosine"),
        default="cosine",
    )
    opt_parser.add_argument("--slow-update-sample-size", type=int, default=20)
    opt_parser.add_argument("--disable-slow-update", action="store_true")
    opt_parser.add_argument("--optimizer-seed", type=int, default=42)
    opt_parser.add_argument("--validation-confirmation-rounds", type=int, default=0)
    opt_parser.add_argument("--validation-required-wins", type=int, default=1)
    opt_parser.add_argument("--validation-mean-delta", type=float, default=0.0)
    opt_parser.add_argument(
        "--editor-command",
        default=None,
        help=(
            "External skill editor command. Also supported via "
            "TEXTSKILL_EDITOR_CMD. Receives JSON on stdin and prints JSON proposals."
        ),
    )
    opt_parser.add_argument("--editor-timeout", type=int, default=120)
    opt_parser.add_argument(
        "--proposal-log-out",
        default=os.environ.get("TEXTSKILL_PROPOSAL_LOG_OUT"),
        help=(
            "Optional JSONL path that records command-editor proposals for "
            "deterministic replay experiments."
        ),
    )
    opt_parser.add_argument(
        "--proposal-log-seed",
        default=os.environ.get("TEXTSKILL_PROPOSAL_LOG_SEED", "default"),
        help="Seed/run label stored with --proposal-log-out records.",
    )
    opt_parser.add_argument(
        "--proposal-log-case",
        default=os.environ.get("TEXTSKILL_PROPOSAL_LOG_CASE", "default"),
        help="Ablation case label stored with --proposal-log-out records.",
    )

    init_parser = subparsers.add_parser("init-example", help="Copy a runnable example")
    init_parser.add_argument(
        "--kind",
        choices=["extraction", "coding"],
        default="extraction",
        help="Example kind to copy. Default: extraction",
    )
    init_parser.add_argument("--out", required=True, help="Destination directory")

    args = parser.parse_args(argv)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    if args.command == "optimize":
        return cmd_optimize(args)
    if args.command == "init-example":
        return cmd_init_example(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skill", required=True, help="Skill markdown file")
    parser.add_argument(
        "--plugin",
        default="extraction",
        help="Plugin module or built-in plugin name. Default: extraction",
    )


def cmd_evaluate(args: argparse.Namespace) -> int:
    plugin = load_plugin(args.plugin)
    optimizer = build_optimizer(plugin)
    report = optimizer.evaluate(
        load_text(args.skill),
        load_tasks_jsonl(args.tasks),
        name="evaluation",
    )
    print_score(report.average_score, report.pass_rate)
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    plugin = load_plugin(args.plugin)
    editor_command = args.editor_command or os.environ.get("TEXTSKILL_EDITOR_CMD")
    max_skill_chars = resolve_lr_value(
        args.max_skill_chars,
        "TEXTSKILL_MAX_SKILL_CHARS",
        lr_defaults(args.lr_profile)["max_skill_chars"],
    )
    editor = build_editor(
        plugin,
        editor_command=editor_command,
        editor_timeout=args.editor_timeout,
        proposal_log_out=args.proposal_log_out,
        proposal_log_seed=args.proposal_log_seed,
        proposal_log_case=args.proposal_log_case,
    )
    if args.protocol == "executive":
        optimizer = ExecutiveSkillOptimizer(
            runner=plugin.build_runner(),
            scorer=plugin.build_scorer(),
            editor=editor,
            config=ExecutiveOptimizerConfig(
                epochs=args.epochs,
                rollout_batch_size=args.rollout_batch_size,
                reflection_minibatch_size=args.reflection_minibatch_size,
                learning_rate=args.text_learning_rate,
                learning_rate_floor=args.text_learning_rate_floor,
                learning_rate_schedule=args.text_learning_rate_schedule,
                max_skill_chars=max_skill_chars,
                rejected_buffer_limit=args.rejected_buffer_limit,
                slow_update_sample_size=args.slow_update_sample_size,
                enable_slow_update=not args.disable_slow_update,
                seed=args.optimizer_seed,
                meta_skill_path=args.meta_skill,
                validation_confirmation_rounds=args.validation_confirmation_rounds,
                validation_required_wins=args.validation_required_wins,
                validation_mean_delta=args.validation_mean_delta,
            ),
        )
    else:
        optimizer = SkillOptimizer(
            runner=plugin.build_runner(),
            scorer=plugin.build_scorer(),
            editor=editor,
            config=OptimizerConfig(
                epochs=args.epochs,
                max_skill_chars=max_skill_chars,
                max_skill_delta_chars=resolve_lr_value(
                    args.max_skill_delta_chars,
                    "TEXTSKILL_MAX_SKILL_DELTA_CHARS",
                    lr_defaults(args.lr_profile)["max_skill_delta_chars"],
                ),
                max_added_bullet_lines=resolve_lr_value(
                    args.max_added_bullet_lines,
                    "TEXTSKILL_MAX_ADDED_BULLET_LINES",
                    lr_defaults(args.lr_profile)["max_added_bullet_lines"],
                ),
                rejected_buffer_limit=args.rejected_buffer_limit,
                meta_skill_path=args.meta_skill,
            ),
        )
    result = optimizer.optimize(
        load_text(args.skill),
        load_tasks_jsonl(args.train),
        load_tasks_jsonl(args.valid),
        run_dir=args.out,
    )
    out_dir = Path(args.out)
    write_text(out_dir / "best_skill.md", result.best_skill_text)
    write_json(out_dir / "result.json", result.to_dict())
    holdout_report = None
    if args.holdout:
        holdout_report = optimizer.evaluate(
            result.best_skill_text,
            load_tasks_jsonl(args.holdout),
            name="holdout:final",
        )
        write_json(out_dir / "holdout_final.json", holdout_report.to_dict())
    print_score(
        result.final_validation_report.average_score,
        result.final_validation_report.pass_rate,
    )
    if holdout_report is not None:
        print(
            "holdout_"
            f"average_score={holdout_report.average_score:.4f} "
            f"holdout_pass_rate={holdout_report.pass_rate:.4f}"
        )
    print(f"best_skill={out_dir / 'best_skill.md'}")
    return 0


def lr_defaults(profile_name: str | None) -> dict[str, int]:
    if profile_name:
        profile = get_lr_profile(profile_name)
        return {
            "max_skill_chars": profile.max_skill_chars,
            "max_skill_delta_chars": profile.max_skill_delta_chars,
            "max_added_bullet_lines": profile.max_added_bullet_lines,
        }
    return {
        "max_skill_chars": 6000,
        "max_skill_delta_chars": 1800,
        "max_added_bullet_lines": 8,
    }


def resolve_lr_value(arg_value: int | None, env_name: str, default: int) -> int:
    if arg_value is not None:
        return arg_value
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return int(env_value)
    return default


def cmd_init_example(args: argparse.Namespace) -> int:
    destination = Path(args.out)
    source = Path(__file__).resolve().parent.parent / "examples" / args.kind
    if source.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        write_text(destination / "skill.md", EXAMPLE_SKILL)
        write_text(destination / "train.jsonl", EXAMPLE_TRAIN)
        write_text(destination / "valid.jsonl", EXAMPLE_VALID)
    print(f"example={destination}")
    return 0


def build_optimizer(
    plugin: ModuleType,
    config: OptimizerConfig | None = None,
    *,
    editor_command: str | None = None,
    editor_timeout: int = 120,
    proposal_log_out: str | None = None,
    proposal_log_seed: str = "default",
    proposal_log_case: str = "default",
) -> SkillOptimizer:
    editor = build_editor(
        plugin,
        editor_command=editor_command,
        editor_timeout=editor_timeout,
        proposal_log_out=proposal_log_out,
        proposal_log_seed=proposal_log_seed,
        proposal_log_case=proposal_log_case,
    )
    return SkillOptimizer(
        runner=plugin.build_runner(),
        scorer=plugin.build_scorer(),
        editor=editor,
        config=config,
    )


def build_editor(
    plugin: ModuleType,
    *,
    editor_command: str | None,
    editor_timeout: int,
    proposal_log_out: str | None,
    proposal_log_seed: str,
    proposal_log_case: str,
):
    return (
        CommandSkillEditor(
            CommandEditorConfig(
                command=editor_command,
                timeout_seconds=editor_timeout,
                proposal_log_path=proposal_log_out,
                proposal_log_seed=proposal_log_seed,
                proposal_log_case=proposal_log_case,
            )
        )
        if editor_command
        else plugin.build_editor()
    )


def load_plugin(name: str) -> ModuleType:
    if name in {"extraction", "builtin:extraction"}:
        name = "textskill_optimizer.plugins.extraction"
    elif name in {"coding", "builtin:coding"}:
        name = "textskill_optimizer.plugins.coding"
    return importlib.import_module(name)


def print_score(average_score: float, pass_rate: float) -> None:
    print(f"average_score={average_score:.4f} pass_rate={pass_rate:.4f}")


EXAMPLE_SKILL = """# Contact Extraction Skill

Extract labeled contact fields from short text.

## Field Aliases
- name: aliases=name, full name
- email: aliases=email
- company: aliases=company
"""

EXAMPLE_TRAIN = """{"id":"train-1","input":"Name: Ada Lovelace; E-mail: ada@example.com; Company: Analytical Engines","expected":{"name":"Ada Lovelace","email":"ada@example.com","company":"Analytical Engines"}}
{"id":"train-2","input":"Full name: Grace Hopper; Email: grace@example.com; Org: Navy","expected":{"name":"Grace Hopper","email":"grace@example.com","company":"Navy"}}
"""

EXAMPLE_VALID = """{"id":"valid-1","input":"Name: Alan Turing; E-mail: alan@example.com; Company: Bletchley Park","expected":{"name":"Alan Turing","email":"alan@example.com","company":"Bletchley Park"}}
{"id":"valid-2","input":"Full name: Katherine Johnson; Email: kj@example.com; Org: NASA","expected":{"name":"Katherine Johnson","email":"kj@example.com","company":"NASA"}}
"""


if __name__ == "__main__":
    raise SystemExit(main())
