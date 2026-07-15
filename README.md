# TextSkill Optimizer

TextSkill Optimizer is a small, validation-gated framework for improving
reusable natural-language skill documents.

It does not fine-tune model weights. It treats the skill text as the trainable
state, runs tasks with a frozen runner, proposes bounded edits from failed
trajectories, and accepts a candidate only when validation score improves.

## Why this exists

Agent behavior often depends on external text: prompts, playbooks, tool-use
rules, review checklists, or `skill.md` files. Hand-editing those documents does
not scale. Unchecked self-revision is worse because it can make the document
look smarter while lowering real task performance.

The core rule here is simple:

```text
No validation improvement, no skill update.
```

## Protocols

- The default `legacy` protocol remains available for minimal examples and
  historical replay.
- `--protocol executive` selects the existing contract-aware extension with
  external-editor proposals, contract feedback, paired confirmation, and local
  guards. It requires an editor with the `atomic_edits` capability; the built-in
  full-replacement editors fail before evaluation instead of producing a no-op
  executive run.
- The isolated `paper-faithful-v1` contract is implemented under
  `textskill_optimizer.paper`; its optimization engine is not yet exposed by
  the CLI.

See `docs/skillopt-executive-protocol.md` only for the historical executive
JSON contract and `coding-hidden-v2` workflow. Its locked test has already been
consumed and must not be run again.

## Paper-Faithful Mainline

The active goal is to align with the SkillOpt paper. The paper is normative;
the Microsoft implementation is used selectively as a code reference and to
resolve ambiguities. The current `executive` protocol remains a supported
contract-aware extension, but its contract feedback, paired confirmation, and
benchmark-specific guards are not part of the paper-faithful default.

The one-attempt `coding-hidden-v2` locked test was consumed on 2026-07-13. Its
20/20 result is frozen as historical `contract-aware-extension-v1` evidence and
must not be rerun or relabeled as paper-faithful. A future paper-faithful
held-out claim requires a new untouched split or an official paper benchmark.

M1 freezes the paper contract before the engine exists. The bundled JSON
profile rejects both extension controls and unregistered deviations, claim
lineage is hash-bound and validated against a versioned schema, and the
consumed-split registry blocks relabeling old evidence. The public
`assess_paper_profile(...)` and
`assess_paper_run(...)` functions perform these checks without constructing a
model backend. M2 adds the runtime firewall: a content-addressed registry binds
each dataset-owning executable/runner launch chain and response key to exactly
one train, selection, or final-test role and one hash-verified split manifest;
signed train responses are reverified at the optimizer
seam, selection exposes one normalized scalar, and final test requires a
revalidated frozen plan bound to the real runner/scorer/harness bytes. The
paper package uses lazy exports so a cold final-only import does not execute
optimization modules. The paper fast loop is the next milestone.

Read `docs/specs/skillopt-paper-faithful-roadmap.md` first. It defines source
precedence, protocol isolation, the official-code reuse boundary, conformance
invariants, implementation phases, evidence gates, and claim provenance. Read
`docs/specs/skillopt-current-state.md` for the latest repository status and
`docs/specs/skillopt-experiment-runbook.md` before operating existing historical
experiment tooling.

For low-Codex-token execution handoffs, use
`docs/specs/skillopt-operator-handoff.md`: the operator runs the manifest and
returns a compact result packet instead of expanding raw run artifacts.
Stage-specific operator READMEs and result templates live under `docs/ops/`.

## Quick start

Install the project and its reproducible test dependency, then run the default
gate:

```bash
python3 -m pip install --editable ".[dev]"
python3 -m pytest -q
```

Run the built-in extraction example without network access:

```bash
python3 -m textskill_optimizer.cli evaluate \
  --skill examples/extraction/skill.md \
  --tasks examples/extraction/valid.jsonl

python3 -m textskill_optimizer.cli optimize \
  --skill examples/extraction/skill.md \
  --train examples/extraction/train.jsonl \
  --valid examples/extraction/valid.jsonl \
  --epochs 2 \
  --out runs/extraction-demo
```

The optimized skill is written to:

```text
runs/extraction-demo/best_skill.md
```

## Data format

Tasks are JSONL:

```json
{"id":"case-1","input":"Name: Ada; E-mail: ada@example.com","expected":{"name":"Ada","email":"ada@example.com"}}
```

`expected` can be any JSON value. The scorer decides how to compare outputs to
expected values.

## Architecture

The framework has three plugin interfaces:

```python
class SkillRunner:
    def run(self, skill_text: str, task: Task) -> TaskOutput: ...

class SkillScorer:
    def score(self, task: Task, output: TaskOutput) -> Score: ...

class SkillEditor:
    def propose(self, skill_text: str, train_results: list[TaskResult], *, epoch: int) -> list[EditProposal]: ...
```

The optimizer owns the invariant:

1. Evaluate current skill on validation tasks.
2. Run training tasks with the frozen runner.
3. Ask the editor for candidate skill edits.
4. Evaluate each candidate on validation tasks.
5. Accept only candidates with a strictly better validation score.
6. Write `best_skill.md` and audit artifacts.

## Built-in plugin

`extraction` is a deterministic plugin for labeled information extraction. It
uses aliases from a skill document:

```markdown
## Field Aliases
- name: aliases=name, full name
- email: aliases=email
- company: aliases=company
```

If training failures reveal labels such as `E-mail` or `Org`, the editor adds
them as aliases. Validation decides whether the edit survives.

`coding` is a command-driven plugin for coding-agent tasks. It copies a fixture
repo to a temporary directory, writes the current skill into `.textskill/`,
invokes an agent command, then runs the task's test command. The score is `1.0`
when post-agent tests pass and `0.0` when they fail.

Run the offline coding example:

```bash
python3 -m textskill_optimizer.cli evaluate \
  --plugin coding \
  --skill examples/coding/skill.md \
  --tasks examples/coding/valid.jsonl

python3 -m textskill_optimizer.cli optimize \
  --plugin coding \
  --skill examples/coding/skill.md \
  --train examples/coding/train.jsonl \
  --valid examples/coding/valid.jsonl \
  --holdout examples/coding/holdout.jsonl \
  --epochs 1 \
  --out runs/coding-demo
```

The example uses `examples/coding/demo_agent.py` so it can run without network
access. For a real agent, set `metadata.agent_command` per task or export:

```bash
export TEXTSKILL_CODING_AGENT_CMD='your-agent-command --repo {repo} --skill {skill} --task {task}'
```

The runner also passes these environment variables to the agent:

```text
TEXTSKILL_REPO_DIR
TEXTSKILL_SKILL_PATH
TEXTSKILL_TASK_PATH
TEXTSKILL_INSTRUCTION
```

To replace the built-in heuristic skill editor with an external LLM-backed
editor, pass `--editor-command` or set `TEXTSKILL_EDITOR_CMD`:

```bash
python3 -m textskill_optimizer.cli optimize \
  --plugin coding \
  --skill examples/coding/skill.md \
  --train examples/coding/train.jsonl \
  --valid examples/coding/valid.jsonl \
  --holdout examples/coding/holdout.jsonl \
  --epochs 1 \
  --editor-command "python3 examples/coding/demo_skill_editor.py" \
  --out runs/coding-command-editor-demo
```

For `--protocol executive`, the editor command first receives
`{"operation":"capabilities"}` and must return an `atomic_edits` capability
without calling a model. Normal reflection calls then receive JSON on stdin:

```json
{
  "epoch": 1,
  "skill_text": "# Coding Agent Skill\n...",
  "train_results": [
    {
      "task": {"id": "coding-train-add-one", "input": "...", "metadata": {}},
      "output": {"value": {"tests_passed": false}, "metadata": {"diff": "..."}},
      "score": {"value": 0.0, "success": false, "message": "..."}
    }
  ]
}
```

It must print JSON proposals to stdout:

```json
{
  "proposals": [
    {
      "name": "root-cause-loop",
      "skill_text": "# full replacement skill text",
      "rationale": "Why this edit should improve future tasks"
    }
  ]
}
```

The optimizer still decides whether to accept the edit by running validation.
The editor cannot mark its own proposal as successful.

To capture external-editor proposals for deterministic replay experiments, add
`--proposal-log-out` plus labels for the run and ablation case:

```bash
python3 -m textskill_optimizer.cli optimize \
  --plugin coding \
  --skill examples/coding/skill.md \
  --train examples/coding/train.jsonl \
  --valid examples/coding/valid.jsonl \
  --epochs 2 \
  --editor-command "python3 examples/coding/demo_skill_editor.py" \
  --proposal-log-out runs/proposal-logs/external.jsonl \
  --proposal-log-seed seed-a \
  --proposal-log-case gate_lr_rejected \
  --out runs/coding-command-editor-demo
```

For the coding-hidden mechanism ablation, the helper below captures all four
standard cases and then replays the fixed proposal log:

```bash
python3 work/run_real_external_editor_proposal_log_ablation.py \
  --out runs/coding-hidden-real-external-editor-capture-v1 \
  --replay-out runs/coding-hidden-real-proposal-log-ablation-v1 \
  --seeds seed-a \
  --cases gate_only,gate_lr,gate_lr_rejected,gate_lr_rejected_meta \
  --timeout-seconds 30
```

Use `--lr-profile` to choose a named textual learning-rate budget:

```bash
python3 -m textskill_optimizer.cli optimize \
  --plugin coding \
  --skill examples/coding/skill.md \
  --train examples/coding/train.jsonl \
  --valid examples/coding/valid.jsonl \
  --epochs 1 \
  --lr-profile real-editor \
  --editor-command "python3 examples/coding/openai_compatible_skill_editor.py" \
  --out runs/coding-real-editor-profile-demo
```

Available profiles:

- `strict`: 600 chars / 260 delta / 1 added bullet
- `real-editor`: 600 chars / 520 delta / 1 added bullet
- `loose-diagnostic`: 750 chars / 700 delta / 3 added bullets

An OpenAI-backed editor is included as a real integration point:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5.2

python3 -m textskill_optimizer.cli optimize \
  --plugin coding \
  --skill examples/coding/skill.md \
  --train examples/coding/train.jsonl \
  --valid examples/coding/valid.jsonl \
  --holdout examples/coding/holdout.jsonl \
  --epochs 1 \
  --editor-command "python3 examples/coding/openai_skill_editor.py" \
  --out runs/coding-openai-editor-demo
```

The script uses OpenAI's Responses API with structured JSON output. It does not
decide success; it only proposes replacement skill text. The optimizer still
runs validation and rejects non-improving edits.

For external models, use the OpenAI-compatible Chat Completions editor. This is
the best default for private model gateways, LiteLLM, vLLM, Ollama-compatible
routes, or hosted models that expose `/v1/chat/completions`:

```bash
export EXTERNAL_LLM_BASE_URL=http://localhost:4000/v1
export EXTERNAL_LLM_MODEL=qwen2.5-coder-32b-instruct
export EXTERNAL_LLM_API_KEY=not-needed

python3 -m textskill_optimizer.cli optimize \
  --plugin coding \
  --skill examples/coding/skill.md \
  --train examples/coding/train.jsonl \
  --valid examples/coding/valid.jsonl \
  --holdout examples/coding/holdout.jsonl \
  --epochs 1 \
  --editor-command "python3 examples/coding/openai_compatible_skill_editor.py" \
  --out runs/coding-external-editor-demo
```

If your endpoint rejects `response_format`, disable JSON mode and let the prompt
enforce JSON:

```bash
export EXTERNAL_LLM_JSON_MODE=0
```

Validate external-model configuration without sending a model request:

```bash
export EXTERNAL_LLM_DRY_RUN=1
printf '{"epoch":1,"skill_text":"# Skill","train_results":[]}' \
  | python3 examples/coding/openai_compatible_skill_editor.py
unset EXTERNAL_LLM_DRY_RUN
```

`EXTERNAL_LLM_BASE_URL` may be either a base URL or the full chat completions
endpoint. Both forms are valid:

```bash
export EXTERNAL_LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
export EXTERNAL_LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
```

This external editor only changes skill text. To use an external model as the
coding agent itself, point `TEXTSKILL_CODING_AGENT_CMD` or task
`metadata.agent_command` at a wrapper script that edits the temporary repo using
these environment variables:

```text
TEXTSKILL_REPO_DIR
TEXTSKILL_SKILL_PATH
TEXTSKILL_TASK_PATH
TEXTSKILL_INSTRUCTION
```

To use Codex as the real coding agent, use the included wrapper:

```bash
export TEXTSKILL_CODING_AGENT_CMD="python3 examples/coding/codex_agent_wrapper.py"
export CODEX_AGENT_MODEL=gpt-5.2

python3 -m textskill_optimizer.cli optimize \
  --plugin coding \
  --skill examples/coding/skill.md \
  --train examples/coding/train.jsonl \
  --valid examples/coding/valid.jsonl \
  --holdout examples/coding/holdout.jsonl \
  --epochs 1 \
  --editor-command "python3 examples/coding/openai_compatible_skill_editor.py" \
  --out runs/coding-codex-agent-demo
```

The wrapper calls:

```bash
codex --ask-for-approval never exec --cd "$TEXTSKILL_REPO_DIR" \
  --skip-git-repo-check --ephemeral --sandbox workspace-write -
```

It sends the skill, task instruction, and test command through stdin as the
Codex prompt. By default, `CODEX_AGENT_PROMPT_MODE=minimal`, so the wrapper
does not teach the agent a repair strategy; the skill document is the process
variable being tested. Set `CODEX_AGENT_PROMPT_MODE=guided` only for debugging.
Set `CODEX_AGENT_DRY_RUN=1` to inspect the generated command and prompt without
starting Codex.

Coding tasks use JSONL metadata:

```json
{
  "id": "bugfix-001",
  "input": "Fix the failing tests without editing tests.",
  "expected": {"tests_passed": true},
    "metadata": {
      "repo": "fixtures/bugfix-001",
      "test_command": "python3 -m unittest discover -s tests",
      "agent_command": "python3 {task_dir}/demo_agent.py"
    }
}
```

For hidden-test experiments, keep `metadata.test_command` as the scorer command
and add `metadata.agent_test_command` for the public command shown to the coding
agent:

```json
{
  "metadata": {
    "repo": "fixtures/bugfix-001",
    "agent_test_command": "python3 -m unittest discover -s tests",
    "test_command": "python3 {task_dir}/run_hidden_tests.py bugfix-001 {repo}",
    "agent_command": "python3 {task_dir}/../coding/openai_compatible_agent_wrapper.py"
  }
}
```

The runner writes only the public command into `.textskill/task.json`, then uses
the scorer command after the agent edits the temporary repo. Both test commands
support `{repo}`, `{task_dir}`, `{skill}`, `{task}`, and `{instruction}`
placeholders.

The coding example now contains 16 small fixtures:

```text
train:   6 tasks
valid:   5 tasks
holdout: 5 tasks
```

`holdout.jsonl` is evaluated only after optimization. It does not decide whether
a candidate skill is accepted.

## Real Coding-Agent Fixtures

`examples/coding-real/` contains fixtures without answer markers. These are for
testing a real coding agent wrapper instead of the offline `demo_agent.py`:

```text
train:   3 tasks
valid:   2 tasks
holdout: 2 tasks
```

First verify the wrapper prompt without starting Codex:

```bash
CODEX_AGENT_DRY_RUN=1 \
python3 -m textskill_optimizer.cli evaluate \
  --plugin coding \
  --skill examples/coding-real/skill.md \
  --tasks examples/coding-real/train.jsonl
```

Then run the real Codex-agent path:

```bash
python3 -m textskill_optimizer.cli optimize \
  --plugin coding \
  --skill examples/coding-real/skill.md \
  --train examples/coding-real/train.jsonl \
  --valid examples/coding-real/valid.jsonl \
  --holdout examples/coding-real/holdout.jsonl \
  --epochs 1 \
  --editor-command "python3 examples/coding/openai_compatible_skill_editor.py" \
  --out runs/coding-real-codex-agent-demo
```

To use an OpenAI-compatible external model as the coding agent, use
`openai_compatible_agent_wrapper.py`. It reads repo files and the initial test
failure, asks the model for JSON file edits, applies those edits locally, and
then the runner executes tests:

```bash
export EXTERNAL_AGENT_BASE_URL=https://ark-cn-beijing.bytedance.net/api/v3
export EXTERNAL_AGENT_MODEL=ep-20260507113406-9h6cz
export EXTERNAL_AGENT_API_KEY=...
export EXTERNAL_AGENT_JSON_MODE=0

python3 -m textskill_optimizer.cli evaluate \
  --plugin coding \
  --skill examples/coding-external-agent/skill.md \
  --tasks examples/coding-external-agent/valid.jsonl
```

Set `EXTERNAL_AGENT_DRY_RUN=1` to inspect the request without calling the model.

## Hidden-Test Coding Fixtures

`examples/coding-hidden/` separates public tests from hidden scorer tests:

```text
train:   8 tasks
valid:   4 tasks
holdout: 4 tasks
```

The hidden tests live outside each copied fixture repo under
`examples/coding-hidden/hidden/`, so the external coding agent can inspect only
the public tests while the runner scores with `run_hidden_tests.py`.

```bash
EXTERNAL_AGENT_DRY_RUN=1 \
EXTERNAL_AGENT_BASE_URL=https://example.invalid/api/v3 \
EXTERNAL_AGENT_MODEL=dry-run-model \
python3 -m textskill_optimizer.cli evaluate \
  --plugin coding \
  --skill examples/coding-hidden/skill.md \
  --tasks examples/coding-hidden/valid.jsonl
```

For a real external-agent run without storing the API key:

```bash
EXTERNAL_AGENT_SKILL=examples/coding-hidden/skill.md \
EXTERNAL_AGENT_TASKS=examples/coding-hidden/valid.jsonl \
python3 work/run_bytedance_external_agent_eval.py
```

To use Coco as the coding agent instead of the OpenAI-compatible wrapper:

```bash
COCO_AGENT_DRY_RUN=1 python3 work/run_coco_hidden_eval.py

python3 work/run_coco_hidden_eval.py
```

The Coco wrapper defaults to `/Users/bytedance/.local/bin/coco` when present and
calls Coco in non-interactive `--print --yolo` mode inside the temporary repo.
Useful controls:

```bash
export COCO_SKILL=examples/coding-hidden/skill.md
export COCO_TASKS=examples/coding-hidden/valid.jsonl
export COCO_AGENT_EXTRA_ARGS="-c model=GPT-5.2"
export COCO_TASK_TIMEOUT=900
```

To run SkillOpt with Coco as the coding agent and an external model as the skill
editor:

```bash
python3 work/run_coco_hidden_skillopt_experiment.py
```

Experiment report:

```text
docs/skillopt-coco-hidden-experiment-report.md
```

## Extending it

Create a Python module with:

```python
def build_runner(): ...
def build_scorer(): ...
def build_editor(): ...
```

Then run:

```bash
python3 -m textskill_optimizer.cli optimize \
  --plugin your_package.your_plugin \
  --skill skill.md \
  --train train.jsonl \
  --valid valid.jsonl \
  --out runs/your-run
```

For a coding-agent version, the runner would call the agent on a repo task, the
scorer would run tests or static checks, and the editor would propose edits to
the agent's skill document from failed trajectories.

## Tests

```bash
python3 -m unittest discover -s tests
```
