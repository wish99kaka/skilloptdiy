# Coco Manifest Runner Smoke Task

You are acting only as a mechanical experiment runner.

## Workspace

`/Users/bytedance/Documents/Codex/2026-06-08/skillopt`

## Command

Run exactly this command from the workspace root:

```bash
python3 work/experiment_runner.py run --manifest work/experiment_runner_manifest.smoke.json
```

## Required Preconditions

- `EXTERNAL_LLM_API_KEY` must already exist in the environment.
- Do not print, request, save, or transform the API key.
- Do not edit `work/experiment_runner_manifest.smoke.json`.
- Do not edit any source code, skill file, benchmark fixture, test, or model config.
- Do not change Coco's configured model. The manifest intentionally uses `target_model_policy: read-local-default-without-override`.

## Allowed Actions

- Run the exact command above.
- If the command exits nonzero, report the exit code and the path to `runner_execution.json` if it exists.
- Return only the final artifact paths and a short status.

## Forbidden Actions

- Do not decide whether to run locked test.
- Do not interpret the benchmark result.
- Do not modify the manifest to make the run faster.
- Do not add or remove `--task-limit`.
- Do not override target model, Coco model, or benchmark paths.
- Do not open raw result JSON files unless `work/experiment_runner.py` fails to produce `runner_report.json`.

## Completion Output

Return this exact shape:

```text
status=<complete|failed|blocked>
runner_report=<path or missing>
runner_execution=<path or missing>
stdout=<path or missing>
stderr=<path or missing>
note=<one short sentence>
```
