# Experiment 01: 局部思维链 (CoT) 修复实验

本目录包含思维链局部修复实验的第一版完整实现。实验基于 `实验设计.md` 中的设计方案。

## 输入数据

实验直接从根目录读取成功和失败的模型运行报告：

- `/mnt/common/lx/Temp01/report_success.json`
- `/mnt/common/lx/Temp01/report_failed.json`

每个案例应包含以下关键字段：
- `problem`: 题目描述
- `solution`: 标准题解
- `gold_answer`: 标准答案
- `model_output`: 原始模型生成的 CoT
- `dataset_idx`: 数据集索引
- `unique_id`: 唯一标识符

## 核心文件说明

- **`run_experiment.py`**: **实验主入口**。控制整个实验流：加载案例 -> 对 CoT 进行切片 -> 调用修复 -> 调用局部判定 -> 运行后续补全 -> 生成汇总报告。
- **`config.py`**: 实验配置和模型端点定义。
- **`data_loader.py`**: 负责加载并预处理成功/失败案例数据。
- **`cot_utils.py`**: 思维链处理工具。执行思维链提取、分割（按换行符）、切片和单元计数。
- **`repair_generator.py`**: 负责调用 LLM 生成修复后的局部思维链片段。
- **`local_quality_judge.py`**: 局部质量评审模块。基于真实性、一致性、正确性、推进性、非泄露性五个 T/F 维度对修复结果进行判定。
- **`continuation_runner.py`**: 运行补全任务。在给定的思维链前缀（原始或修复后）基础上，让模型继续解题。
- **`report_builder.py`**: 将零散的实验记录聚合成详细的 JSON 报告和 Markdown 摘要。
- **`prompts/`**: 存放修复、判定和补全任务所用的 Prompt 模板。

## 使用方法

请从项目根目录运行：

```bash
python experiment01/run_experiment.py
```

可选参数示例（指定来源、限制数量、限制切点）：

```bash
python experiment01/run_experiment.py \
  --sources failed \
  --max-cases-per-source 5 \
  --max-cutpoints-per-case 3
```

## 模型配置

实验默认配置如下：

- **修复生成 (`repair_generator.py`)**: 本地模型 `Qwen3-8B` (`http://localhost:6002/v1`)
- **后续补全 (`continuation_runner.py`)**: 本地模型 `Qwen3-8B` (`http://localhost:6002/v1`)
- **局部判定 (`local_quality_judge.py`)**: `gpt-5.4-high` (`https://www.right.codes/codex/v1`)

您可以通过环境变量覆盖各阶段的配置，例如：`EXPERIMENT01_REPAIR_BASE_URL`、`EXPERIMENT01_LOCAL_JUDGE_MODEL` 等。

## 输出结果

每次运行会生成：

- **详细运行报告**: `data/runs/experiment_run_{run_id}.json`
- **Markdown 统计摘要**: `outputs/summaries/experiment_run_{run_id}.md`

报告包含每个切点的中间数据，包括切点位置、原始后续 vs 修复后续、局部 T/F 判定及原因、补全输出、正确性对比、以及当两边都正确时的思维链缩短比例。
