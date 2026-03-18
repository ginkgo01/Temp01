# experiment01

This directory contains the first runnable implementation of the local chain-of-thought repair experiment described in `实验设计.md`.

## Input sources

The experiment reads solved and failed cases directly from:

- `/mnt/common/lx/Temp01/report_success.json`
- `/mnt/common/lx/Temp01/report_failed.json`

Those files already contain the fields needed by this experiment:

- `problem`
- `solution`
- `gold_answer`
- `model_output`
- `dataset_idx`
- `unique_id`

## Main files

- `config.py`: experiment configuration and model endpoints
- `data_loader.py`: load and normalize success/failed report cases
- `cot_utils.py`: thought extraction, splitting, slicing, and length counting
- `repair_generator.py`: generate repaired local continuations
- `local_quality_judge.py`: judge local repair quality on 5 T/F dimensions
- `continuation_runner.py`: continue solving from original or repaired prefixes
- `report_builder.py`: aggregate detailed records into summary statistics
- `run_experiment.py`: experiment entry point
- `prompts/`: prompt templates

## Suggested usage

Run from the workspace root:

```bash
python /mnt/common/lx/Temp01/experiment01/run_experiment.py
```

Optional example:

```bash
python /mnt/common/lx/Temp01/experiment01/run_experiment.py \
  --sources failed \
  --max-cases-per-source 5 \
  --max-cutpoints-per-case 3
```

## Model configuration

By default, the stages are split as follows:

- `repair_generator.py`: local model `http://localhost:6002/v1`, `model/Qwen3-8B`
- `continuation_runner.py`: local model `http://localhost:6002/v1`, `model/Qwen3-8B`
- `local_quality_judge.py`: GPT judge configured directly in `config.py`

The local-quality judge defaults to:

- `https://www.right.codes/codex/v1`
- model `gpt-5.4-high`

You can override endpoints per stage with environment variables:

- `EXPERIMENT01_REPAIR_BASE_URL`
- `EXPERIMENT01_REPAIR_API_KEY`
- `EXPERIMENT01_REPAIR_MODEL`
- `EXPERIMENT01_LOCAL_JUDGE_BASE_URL`
- `EXPERIMENT01_LOCAL_JUDGE_API_KEY`
- `EXPERIMENT01_LOCAL_JUDGE_MODEL`
- `EXPERIMENT01_CONTINUATION_BASE_URL`
- `EXPERIMENT01_CONTINUATION_API_KEY`
- `EXPERIMENT01_CONTINUATION_MODEL`
- `EXPERIMENT01_EQUIV_JUDGE_BASE_URL`
- `EXPERIMENT01_EQUIV_JUDGE_API_KEY`
- `EXPERIMENT01_EQUIV_JUDGE_MODEL`

This makes it easy to switch the repair/judge stages to another API such as GPT-style endpoints without touching the code.

Important:

- The local quality comparison stage is intentionally configured to use GPT-5.4 by default.
- The continuation stage intentionally keeps using the original local model, so the final continuation comparison stays aligned with the original solver family.

## Outputs

Each run writes:

- a detailed JSON report to `data/runs/`
- a markdown summary to `outputs/summaries/`

The detailed report stores per-slice intermediate data, including:

- cut position
- original suffix
- repaired suffix
- local T/F judgments with reasons
- continuation outputs
- correctness results
- shortening ratio when both continuations are correct
