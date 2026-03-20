# CoT 局部修复实验项目

本项目旨在探索通过向模型提供标准答案（gold answer）和题解（solution）来修复数学题解过程中的局部思维链（Chain-of-Thought, CoT），并验证这种修复是否能提升最终的解题正确率或优化思维链长度。

## 目录结构

### 根目录文件

*   **`实验计划.md`**: 项目的总体规划文档，阐述了实验的核心思路、切片逻辑以及初步的评价维度。
*   **`answer_judge.py`**: 数学答案判定核心工具。包含 LaTeX 规范化处理、数学等效性检查（SymPy 符号计算），以及在复杂情况下使用 LLM 进行兜底判定的逻辑。
*   **`llm_settings.py`**: 全局 LLM 服务配置。定义了不同阶段使用的模型端点（如 Qwen3-32B, Qwen3-8B, DeepSeek 等）及其 API 密钥和基础 URL。
*   **`report_success.json` / `report_failed.json`**: 实验的基础输入数据。包含原始模型在 MATH500 等数据集上运行的成功和失败案例，提供题目、标准答案、题解及模型生成的原始 CoT。
*   **`math500_samples.txt`**: MATH500 数据集的样本参考。

### `experiment01/` 目录

此目录包含局部思维链修复实验的第一版完整实现。

*   **`run_experiment.py`**: **实验主入口**。负责加载数据、对思维链进行切片、调用修复生成、执行质量评测以及运行后续补全对比。
*   **`config.py`**: 实验专用配置。定义了实验特有的路径、L 值（切片长度）、模型阶段配置（修复阶段、本地判定阶段、补全阶段）等。
*   **`cot_utils.py`**: 思维链处理工具。负责将 CoT 按 `\n\n` 分割为思考单元、计算合法切点、以及构建前缀/后续切片。
*   **`data_loader.py`**: 数据加载模块。负责从根目录的 JSON 报告中读取并规范化实验案例。
*   **`repair_generator.py`**: 局部修复生成器。根据题目、前缀 CoT 和标准题解，引导模型生成“修复后”的局部思考片段。
*   **`local_quality_judge.py`**: 局部修复质量判定。调用 GPT-5.4 等高强度模型，从真实性、一致性、正确性、推进性、非泄露性五个维度对修复片段进行二值判定。
*   **`continuation_runner.py`**: 补全运行器。在给定修复或原始片段后，让模型继续完成题目解答，以验证局部修复对全局结果的影响。
*   **`report_builder.py`**: 实验报告构建。汇总实验过程中的各项指标，生成详细的 JSON 运行记录和 Markdown 格式的统计摘要。
*   **`io_utils.py`**: 基础 I/O 辅助工具，处理目录创建和 JSON 保存等。
*   **`实验设计.md`**: `experiment01` 的详细设计方案，包括具体的问题定义、切分方式、评测维度和成功信号定义。

## 快速开始

在项目根目录下运行以下命令启动 `experiment01`:

```bash
python experiment01/run_experiment.py --sources failed --max-cases-per-source 5
```

实验结果将保存在 `experiment01/data/runs/` (详细 JSON) 和 `experiment01/outputs/summaries/` (Markdown 摘要) 中。
