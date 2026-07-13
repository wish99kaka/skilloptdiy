# SkillOpt Manifest Check

packet_id:
stage:
manifest:
run_dir:
created_at:
approved_to_run:

## Accounting Scope

cost_scope:
target_agent_usage_scope:
cost_claim_limit:

## Manifest Checks

- stage_full_selection_development:
- stage_same_run_baseline_matrix:
- mechanical_runner_role:
- timeout_seconds_at_least_43200:
- timeout_seconds_at_least_86400:
- early_stop_validation_score_1:
- coco_target_model_not_overridden:
- conditions_executive_only:
- conditions_include_no_skill_human_skill_one_shot_executive:
- confirmation_rounds_at_least_1:
- baseline_summary_present:
- baseline_summary_absent:
- baseline_summary_full_development_not_targeted:
- no_train_task_ids:
- no_selection_task_ids:
- no_task_contracts:
- no_task_limit:
- fresh_run_dir_or_report_regen_plan:
- rows_same_seed_task_model_scorer_confirmation_retry_environment:

## Preflight

command:
status:
reason:

## Proposed Execution Command

```bash

```

## Operator Notes
