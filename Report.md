# Mini AutoResearch Agent：受控自主强化学习实验系统技术报告

> **报告范围与证据基线**  
> 本报告分析仓库中 Mini AutoResearch Agent 的设计、实现和一次完整实验运行。代码事实以当前工作树中的源代码和 `configs/autoresearch.yaml` 为依据；实验事实只采用状态为 `completed` 且通过 Harness 完成校验的 `runs/research_agent_003`，该历史运行对应的仓库 `HEAD` 为 `94c5f8af464f557c05ffa1f03cc503de99dff5dd`。验证日期为 2026-08-13。除明确标记为“解释”“假设”或“局限”的内容外，本文不把 LLM 的主观判断当作系统事实。

**Contents:** [Introduction](#introduction) · [System Architecture](#system-architecture) · [Autonomous Experimentation Workflow](#autonomous-experimentation-workflow) · [Key Design Decisions](#key-design-decisions-and-safety-mechanisms) · [Experiments and Results](#experiments-and-results) · [Limitations and Future Improvements](#failure-cases-limitations-and-future-improvements) · [Conclusion](#conclusion) · [Evidence Index](#evidence-index)

## Introduction

Mini AutoResearch Agent 是一个面向小规模机器学习实验的自主研究 Harness。本工程将大语言模型用于规划、假设生成、工具选择和结果解释，同时将文件访问、子进程执行、指标校验、回滚、预算与终止条件交给确定性的 Python 控制层。当前示例任务是在固定 Q-table 状态空间的前提下，对 Flappy Bird Q-learning 超参数进行有界搜索，以提高 100 回合评估的 `mean_score`，并使用 `mean_reward` 作为诊断及同分判定指标。

工程要解决的核心问题并非单纯“让模型给出一组参数”，而是建立一个可以真实执行、失败可恢复、结果可验证、过程可追踪的实验闭环。一次完整运行包含目标读取、显式计划、baseline、候选实验、独立比较、快照回滚、报告生成和完成条件验证。所有候选结果均来自实际训练/评估子进程，而不是由 LLM 预测或预先写入。

本系统的主要特点如下：

- **真实执行：** LLM 通过受控工具写入运行级配置并启动真实训练/评估进程；
- **职责分离：** LLM 提出动作，Harness 决定动作是否合法并独立核验结果；
- **有界自治：** 搜索空间、路径、命令、实验次数、工具次数和运行时间都有显式上限；
- **失败恢复：** 每个候选运行前保存已接受配置，回滚后进行读取一致性验证；
- **完整审计：** 对话、请求、工具参数、指标、日志、状态迁移及最终报告均落盘；
- **可测试性：** 单元与端到端测试覆盖 SDK 消息处理、路径隔离、指标验证、回滚和完成门槛。

在已完成运行 `research_agent_003` 中，系统执行了 1 次 baseline 和 5 次候选实验，39 次 LLM 调用对应 39 次工具调用，最终以 `agent_finish` 正常结束。最佳候选的 `mean_score` 从 baseline 的 5.20 提高到 14.31，绝对增加 9.11，相对增加约 175.2%。该结果证明了本次运行在固定种子和既定评估协议下取得了明显改进；由于仅使用单一随机种子且评估方差较高，它不构成跨种子泛化或统计显著性的证明。
 
## System Architecture
 
### Architecture Overview

系统采用“LLM 决策层—受控执行层—任务运行层—证据层”的分层架构。`main.py` 只负责解析命令行并调用 `run_agent()`；核心控制循环位于 `autoresearch/agent.py`，模型传输位于 `autoresearch/planner.py`，所有可执行动作集中在 `autoresearch/tooling.py`。具体任务由 `tasks/flappy_qlearning/run.py` 适配，确定性指标比较和报告验证分别位于 `autoresearch/execution.py` 与 `autoresearch/report.py`。

```text
自然语言目标 + 运行配置
          │
          ▼
┌─────────────────────────────┐
│ LLM Agent / Planner         │  生成计划、实验假设和下一工具调用
└──────────────┬──────────────┘
               │ 单个结构化 tool call
               ▼
┌─────────────────────────────┐
│ Tool Harness                │  校验路径、命令、配置、状态前置条件
└───────┬─────────────┬───────┘
        │             │
        ▼             ▼
 文件/快照操作   Experiment Runner
                      │ train → eval
                      ▼
               指标、日志、返回码
        │             │
        └──────┬──────┘
               ▼
┌─────────────────────────────┐
│ Validator + State Machine   │  接受/回滚、预算检查、完成门槛
└──────────────┬──────────────┘
               ▼
 state / messages / trajectory / report
```

该架构中存在两条互补的数据链：

1. **交互链：** system prompt、原始目标、近期 assistant/tool 回合与权威状态摘要构成 LLM 的决策上下文；
2. **事实链：** 配置、子进程状态、指标文件、快照和 `state` 构成 Harness 的权威证据。

两条链不会相互替代。LLM 可以解释工具结果，却不能直接把某个结果写成“已验证”；只有 Harness 从文件和进程状态中重新计算并接受的事实，才会更新当前最佳配置或允许运行结束。
 
### Main Components
 
#### LLM Agent / Planner

LLM Agent 是系统中的决策者，负责把自然语言研究目标转化为可执行的实验过程，但不直接读写文件或启动进程。运行开始时，`agent.py` 构造 system prompt，向模型明确研究目标、运行目录、目标指标、实验次数、回滚要求、文件权限和完整实验协议。模型首先必须提交至少五步的研究计划，随后根据机器生成的权威状态摘要、近期完整工具回合和当前观察，逐轮选择下一项操作。

Planner 通过 OpenAI-compatible Chat Completions 接口调用配置的模型。每次请求均携带精简后的模型上下文和原生 function-calling schemas，并强制设置 `tool_choice="required"`、`parallel_tool_calls=False`。因此，模型每轮必须且只能选择一个工具，形成清晰的“决策—执行—观察”边界。模型输出由 `planner.py` 规范化并解析为工具名称、调用 ID 和 JSON 参数；近期窗口内 Kimi/DashScope 的 `reasoning_content` 会随对应 assistant 回合一起保留和回放。

LLM 主要承担以下认知任务：

- 阅读目标、运行配置、任务代码、历史指标和日志，建立实验背景；
- 提出带有假设和预期效果的候选超参数配置；
- 根据指标证据判断候选结果应接受还是回滚；
- 在工具报错或实验失败后反思原因并选择恢复动作；
- 更新研究计划，最终撰写实验报告并请求结束运行。

这种设计将开放式推理限制在“提出动作和解释证据”的范围内。路径安全、命令合法性、指标比较和完成条件均由确定性的 Python 代码控制，模型的陈述不会被直接视为事实。
 
#### Tool

工具层由 `tooling.py` 中的 `ToolHarness` 实现，是 LLM 与真实环境之间唯一的执行接口。系统当前提供九个原生工具：`submit_plan`、`update_plan`、`list_files`、`read_file`、`write_file`、`run_command`、`evaluate_result`、`restore_snapshot` 和 `finish`。其中，前两项维护计划，文件工具收集或写入信息，实验工具完成运行与决策，最后两项负责恢复和受控终止。

Tool Harness 不提供任意 Python/Shell 权限，而是在执行每个动作前实施约束：

- 首次工具调用必须是 `submit_plan`，并校验计划的步骤数和唯一 ID；
- 读取路径只能位于配置的 `read_roots` 中，写入只能发生在当前 `runs/<run_id>/`；
- 文件读取限制行数和返回字符数，文件写入限制大小并记录修改前后的 SHA-256；
- `run_command` 使用参数数组和 `shell=False`，只允许配置中指定的 Python 实验脚本；
- 命令中的配置路径、run ID、实验类型和超时时间必须符合当前状态；
- 候选配置只能修改 `search_space` 允许的字段和值，且不能重复已测试配置。

每次工具调用都会返回结构化的 `ok`、结果证据和 `next_action`。可预期的路径错误、参数错误、非法状态或子进程错误会被转换为工具观察并反馈给 LLM，而不是直接破坏控制循环。这使 Agent 能够针对真实错误进行纠正，同时保证环境边界不被绕过。
 
#### Experiment Runner

实验执行采用两层进程结构。外层由 `ToolHarness._tool_run_command()` 调用配置好的 `tasks/flappy_qlearning/run.py`，负责校验命令、创建实验目录、施加超时并捕获 stdout、stderr 和返回码。baseline 必须与配置文件中的基准配置完全一致；每次 candidate 运行前，系统会保存上一份已接受配置的快照，并记录候选配置和变更内容，为后续确定性回滚提供依据。

内层任务适配器 `tasks/flappy_qlearning/run.py` 将一次实验进一步拆分为训练和评估两个阶段：

1. 读取并规范化 YAML 配置，生成独立的 train/eval 配置；
2. 启动训练子进程，生成新的 Q-table 和训练结果；
3. 仅在训练成功后启动评估子进程，并关闭探索行为；
4. 在共享 deadline 内汇总两个阶段的返回码、耗时和指标；
5. 输出 `summary.json`，其中包含 `mean_score`、`mean_reward`、训练/评估状态及运行时间。

Runner 为 pygame 设置 dummy video/audio driver，从而支持无界面的自动实验。完整输出保存在实验目录的日志文件中，而返回给 LLM 的仅是有界的日志尾部和结构化指标，既保留诊断证据，也避免单次观察过度占用上下文。
 
#### State / Trace Recorder

状态与轨迹记录由 `agent.py` 和 `ToolHarness` 共同维护，分别解决“当前事实是什么”和“这些事实如何产生”两个问题。

`state.json` 是当前运行状态的可序列化快照，记录运行阶段、计划、baseline、当前配置、最佳配置与指标、已测试配置指纹、待决实验、历史实验、失败和恢复事件，以及已使用和剩余预算。候选实验首先进入 `pending_run`；只有经过 `evaluate_result`，并在需要时完成 `restore_snapshot`，才会被转移到不可变的 `history`。因此，未决实验不会被误认为已接受结果。

轨迹记录则按用途拆分：

- `messages.jsonl` 保存完整的 system/user/assistant/tool 对话；
- `trajectory.jsonl` 保存工具决策、参数、结果、错误、恢复动作和停止事件；
- `llm_calls/call_XXX/` 保存每次实际请求与原始响应，请求在网络调用前写入，因而模型请求卡死时仍可诊断；
- `llm_calls/call_XXX/context.json` 保存该轮权威状态摘要、完整/请求消息数量和窗口裁剪统计；
- `tool_calls/call_XXX_<tool>/` 保存每次工具的参数和结构化结果；
- baseline 和各 iteration 目录保存配置、执行信息、指标、stdout/stderr 及原始训练/评估产物。

终端进度由线程安全的 `ProgressReporter` 输出，并在长时间 LLM 请求或实验期间定期发送 heartbeat。终端只显示有界摘要，不打印完整 prompt、文件内容或 API key；API key 仅从环境变量读取，不写入运行产物。
 
#### Validator

Validator 是独立于 LLM 判断的确定性证据检查层，贯穿初始化、实验执行、结果决策、回滚和最终报告五个阶段：

- **运行配置验证：** 检查必需配置段、正数预算、至少三次候选实验、模型配置、目标方向、指标名称、实验命令和工具 allowlist；
- **动作与配置验证：** 检查路径包含关系、命令参数、baseline 一致性、候选变更是否属于搜索空间，以及配置指纹是否重复；
- **执行验证：** `verify_execution()` 同时检查超时、进程返回码、指标文件是否存在且为合法 JSON、`status` 是否成功，以及目标指标是否存在、可转为数值且为有限值；
- **结果验证：** `compare_metrics()` 根据 maximize/minimize 方向、最小改进阈值和可选 tie-breaker 比较 candidate 与当前最佳结果。LLM 可以保守地回滚一个改善结果，但不能接受被验证器判定为下降或无效的结果；
- **恢复验证：** 回滚时读取运行前快照，与权威的 `current_config` 比较，复制后再次读取并进行内容一致性检查，同时记录恢复文件的 SHA-256，确保恢复确实生效且可审计；
- **报告验证：** 检查必需章节、所有实验轮次、精确最佳指标和恢复证据，防止最终报告与持久化事实不一致。

因此，工具返回和 LLM 解释都只是待核实的证据。只有验证器认可的配置、指标、回滚和报告才能推动状态机进入下一阶段或完成状态。
 
#### Budget / Termination Controller

预算与终止控制器位于 Agent 主循环和 Tool Harness 两层。预算从 `configs/autoresearch.yaml` 载入并写入运行状态；当前默认值为至少 3、最多 5 次候选实验，最多 40 次工具调用，最多 3 次连续失败，总墙钟时间 30 分钟，单次实验最长 300 秒。`remaining_budget` 会在每轮后重新计算，使预算消耗成为显式状态。

主循环在每次 LLM 调用前检查工具调用数、连续失败数和墙钟时间。工具层还会在实际执行前再次检查 candidate 数量和工具预算，并将请求的实验超时压缩到“单次实验上限”和“剩余墙钟时间”中的较小值，从而避免某次子进程越过全局 deadline。成功的工具调用会清零连续失败计数；LLM 请求失败、非法工具调用和工具执行错误则会累加失败记录。

系统区分正常完成和安全退出：

- `agent_finish`：计划全部完成、达到最少实验次数、存在符合要求的真实回滚证据，且最终报告验证通过；
- `tool_call_budget_exhausted`：工具调用预算耗尽，状态为 `incomplete_requirements`；
- `wall_time_budget_exhausted`：墙钟预算耗尽，状态为 `incomplete_requirements`；
- `too_many_failures`：连续失败达到阈值，状态为 `failed`；
- `unrecoverable_error`：未预期异常终止，状态为 `failed`。

这种双层检查保证模型不能通过提前调用 `finish`、反复试验或请求超长进程绕过完成要求；即使运行被安全限制终止，已有状态、轨迹和实验产物仍会在 `finally` 阶段落盘，便于后续分析。
 
### Configuration and Reproducibility

系统使用单一 YAML 文件统一描述模型、预算、实验协议、优化目标、baseline、搜索空间和工具权限。默认配置位于 `configs/autoresearch.yaml`，运行开始后会复制为 `runs/<run_id>/run_config.yaml`，使后续审计不依赖可能已经变化的全局配置。

关键配置如下：

| 类别 | 当前设置 | 作用 |
|---|---|---|
| Planner | `kimi/kimi-k3`，OpenAI-compatible endpoint，`temperature=0.2` | 提供原生工具调用决策 |
| 上下文 | 最近 4 个完整回合、最多 24,000 字符、每轮注入权威 JSON 摘要 | 限制历史增长并直接暴露当前事实 |
| API 认证 | 环境变量 `DASHSCOPE_API_KEY` | 密钥不进入配置和运行产物 |
| 主目标 | 最大化 `mean_score`，目标值 5.0 | 决定候选是否优于当前最佳 |
| 同分指标 | `mean_reward` | 仅当主指标未超过最小改进阈值时参与比较 |
| 实验预算 | 3–5 次候选、40 次工具调用、30 分钟墙钟时间 | 约束自主搜索规模 |
| 失败预算 | 最多 3 次连续失败 | 防止无效动作无限重试 |
| 子进程预算 | 每次实验最多 300 秒 | 限制训练/评估运行时间 |
| 评估协议 | 训练后执行 100 回合无探索评估 | 产生可比较的候选指标 |
| 文件权限 | 读 `tasks/flappy_qlearning`、`configs` 和当前 run；只写当前 run | 隔离源代码和其他运行 |

baseline 使用 `seed=0`、1000 个训练回合、100 个评估回合、启发式 Q-table 初始化、`learning_rate=0.6`、`discount_factor=0.95`、固定探索率 0.001、`sample_t=3`。目标要求状态空间保持不变，因此 `dx_dy_bin_size=25` 和 `velocity_bin_size=2` 不属于搜索变量。允许搜索的字段只有：

- `training_episodes ∈ {1000, 1500, 2000}`；
- `learning_rate ∈ {0.3, 0.4, 0.6, 0.7}`；
- `epsilon_start ∈ {0, 0.001, 0.01, 0.05}`；
- `epsilon_end ∈ {0, 0.001}`；
- `epsilon_decay ∈ {0.99, 0.995, 1.0}`；
- `discount_factor ∈ {0.9, 0.95, 0.99}`；
- `sample_t ∈ {2, 3, 4}`；
- `rewards.alive ∈ {0.02, 0.05, 0.1}`。

每个 run ID 必须是新目录名，程序使用 `exist_ok=False` 拒绝覆盖旧运行。每轮的候选配置、原始指标、训练得到的 Q-table、日志和执行元数据都会保存在独立目录中；`fingerprint()` 以稳定序列化后的完整配置检测重复候选。复现实验时，应使用相同源码版本、`run_config.yaml`、输入 Q-table、Python 依赖和随机种子，并指定新的 run ID：

```powershell
python -m pip install -r requirements.txt
$env:DASHSCOPE_API_KEY="<your-key>"
python main.py --config configs/autoresearch.yaml --run-id <new-run-id>
python -m unittest discover -s tests -v
```

`requirements.txt` 对 NumPy、pygame、PyYAML、OpenAI SDK 和 setuptools 规定了主版本范围，但没有提供完整 lockfile；操作系统、Python 小版本和硬件也未写入实验摘要。因此当前设计能够复现协议与配置，尚不能保证跨环境的逐位相同结果。公开复现时应额外记录 `python --version`、平台信息、完整依赖冻结结果和源码 commit。
 
## Autonomous Experimentation Workflow
 
### Initialization and Context Collection

`run_agent()` 在任何模型请求和实验执行之前完成确定性初始化，主要步骤为：

1. 解析并规范化仓库根目录、运行目录和配置路径；
2. 加载 YAML，并通过 `_validate_runtime_config()` 检查必需字段、预算关系、目标方向、模型 provider、实验命令和 allowlist；
3. 校验 run ID 只能是单层目录名，在 `runs/` 下以 `exist_ok=False` 创建唯一目录；
4. 从 `goal_file` 读取自然语言目标，或采用命令行 `--goal` 覆盖值；
5. 创建初始状态，其中包含目标、阶段、预算、计数器、完成条件和空的实验历史；
6. 将运行配置和初始 `state.json` 写入磁盘；
7. 构建 system prompt、用户目标和九个工具 schema，初始化 `messages`；
8. 创建绑定到当前 run、只读根目录和唯一写目录的 `ToolHarness`。

system prompt 不只描述目标，还给出强制实验协议：先提交计划；检查目标、配置及源文件；写入与 baseline 完全一致的 `current_config.yaml`；运行 baseline；随后只能在声明的搜索空间内产生候选；对每个候选检查指标并执行接受或回滚；最后生成包含规定章节和所有轮次证据的报告。baseline 不是 Python 中隐藏的自动步骤，LLM 必须显式调用 `write_file` 和 `run_command`，因此其动作会和候选实验一样进入审计轨迹。

上下文收集采用按需读取。Agent 可以列出允许目录并分段读取目标、配置、代码、指标和日志，但 `read_file` 默认最多返回 500 行和 12,000 字符。这既使模型能基于真实工程资料规划，又防止一次文件读取吞噬过多上下文。
 
### Context Management

#### Overall Strategy

当前工程采用“完整审计历史 + 有界请求窗口 + 权威状态摘要”的上下文管理方式。模型对话与实验状态仍然分离：Python Harness 维护权威结构化状态并验证全部状态迁移；每次请求前，再从该状态生成小型 JSON 摘要，直接向 LLM 提供当前事实。

```text
完整 messages ───────────────→ messages.jsonl（全量审计）
      │
      ├─ system prompt + 原始目标
      ├─ 最近 4 个完整 assistant/tool 回合（≤ 24,000 字符）
state ──→ AUTHORITATIVE_STATE_SUMMARY（JSON）
      │
      └──────────────────────→ 本轮模型请求
                                  ↓
                         request.json + context.json
```

#### Conversation and State

每次运行仍以 system prompt 和自然语言目标作为固定锚点。模型回复与工具结果继续只追加到内存 `messages` 和 `messages.jsonl`，但网络请求不再回放全部历史。Harness 将 assistant 消息及其后续 tool result 或 correction user 消息视为一个不可拆分的回合组，从最新位置向前最多保留 4 组；若序列化后的近期消息超过 24,000 字符，则从最旧组开始整体移除。这样不会产生缺少 `tool_calls` 来源的孤立 tool result。Kimi/DashScope 的 `reasoning_content` 会随被保留的 assistant 回合一起回放。

与此同时，Harness 在每轮调用前从 `state` 生成 `schema_version=1` 的权威摘要，覆盖运行阶段、迭代进度、计划状态、完整当前/最佳配置、最佳指标、待决实验、最近实验结果、精简后的最近工具观察、完成条件、失败和实时剩余预算。最近工具观察会剔除文件正文、日志尾部和大型文件列表，只保留结果、路径、哈希、指标、下一动作及省略体积；即使一个超大回合被窗口整体移除，模型仍能获知该工具是否成功。摘要作为请求末尾的临时 user JSON 消息注入，并明确声明旧消息与摘要冲突时以摘要为准。该派生消息不进入完整 `messages.jsonl`，LLM 也不能直接修改权威状态，只能通过受控工具触发状态迁移。

#### Persistence and Traceability

每个运行目录都会保存以下上下文资料：

- `messages.jsonl`：完整的 LLM、用户和工具消息；
- `state.json`：最新的结构化运行状态；
- `trajectory.jsonl`：工具调用、决策、错误、恢复和后续动作；
- `llm_calls/call_XXX/`：每轮发送的完整请求和模型原始响应；
- `llm_calls/call_XXX/context.json`：该轮摘要、保留/省略回合数、字符用量和消息数量；
- `tool_calls/call_XXX_<tool>/`：工具参数及执行结果。

`request.json` 表示模型实际看到的精简上下文，`messages.jsonl` 表示未裁剪的完整交互，两者由 `context.json` 中的构造元数据连接。这些记录支持复盘、审计和故障定位，但目前尚未提供从 `state.json` 和 `messages.jsonl` 自动恢复中断任务的入口；重新调用 `run_agent()` 仍会创建新的运行目录和上下文。历史运行 `research_agent_003` 生成于该机制引入之前，因此其旧产物没有 `context.json`，实验指标不受本次上下文改造影响。

#### Context Growth and Limitations

工程同时限制单次观察与累计近期历史：文件读取默认最多 500 行、返回最多 12,000 字符，文件列表最多 200 项，实验命令只返回 stdout/stderr 尾部；模型请求最多保留近期 4 个完整回合且这些回合合计不超过 24,000 字符。`context.enabled=false` 可恢复原有全量请求行为，用于兼容或诊断。

当前窗口按 JSON 序列化字符数计量，并不等价于模型 tokenizer 的精确 token 预算；固定 system prompt、原始目标、工具 schemas 和状态摘要也不计入 `max_recent_chars`。系统仍没有请求前 token 估算、上下文超限专项恢复或跨进程断点续跑。对于更长任务，下一步应增加模型感知的 token 预算、摘要体积上限和持久化恢复机制。

#### Post-Implementation Comparison

为评估改造效果，将启用完整历史的 `research_agent_003` 与启用权威摘要和滑动窗口的 `context_control_001` 对齐到“Iteration 1 已接受”阶段。两次运行的 baseline `mean_score` 均为 5.20；token 统计来自各轮 `response.json` 的 API usage，仅累计成功响应。

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| 到达该阶段的成功 LLM 调用 | 17 | 19 | +2 |
| 累计输入 token | 248,855 | 121,842 | **-51.0%** |
| 累计输出 token | 5,918 | 9,515 | +60.8% |
| 累计 total token | 254,773 | 131,357 | **-48.4%** |
| Iteration 1 `mean_score` | 10.47 | **18.96** | **+81.1%** |
| Iteration 1 `mean_reward` | 218.29 | **459.66** | **+110.6%** |

在产生 Iteration 1 评估决定的单轮请求中，输入量从 26,057 token 降至 7,116 token，降幅为 **72.7%**；优化后的请求只包含 11 条消息、最近 4 个完整回合，并省略 14 个较早回合。旧运行完成 5 个候选后共使用 1,012,192 total token，最佳 `mean_score` 为 14.31；新运行在第 1 个候选、累计 131,357 token 时已观察到 18.96。因此，本次运行同时表现出更低的上下文成本和更高的早期搜索效率。

该比较不是严格的因果 A/B 实验。两次 Iteration 1 使用了不同候选配置，且都只使用 `seed=0`；新运行还出现了重复计划、非法参数和命令超时等工具纠错，最终因 DashScope 返回余额不足的 429 错误而在 Iteration 1 后中止，未完成全部实验、回滚证据和最终报告。由此可以确认上下文压缩显著降低了 token 增长，但实验分数提升仍需通过相同候选、多随机种子和多次完整运行进一步验证。原始证据见 [`research_agent_003/state.json`](runs/research_agent_003/state.json)、[`context_control_001/state.json`](runs/context_control_001/state.json) 及两次运行的 `llm_calls/` 目录。
 
### Plan–Act–Observe–Validate Loop

主循环由五个连续阶段组成。虽然 LLM 每轮只发起一个工具调用，但多轮组合后可完成完整实验。状态、计划与工具前置条件确保动作顺序不是纯粹依赖 prompt 服从。
 
#### Planning

第一次模型响应必须调用 `submit_plan`。计划至少包含五个具有唯一 ID 的步骤，每步描述预期使用的工具和成功信号，并记录风险。后续使用 `update_plan` 将步骤标记为 `pending`、`in_progress`、`completed` 或 `failed`，同时写入证据和下一动作。`finish` 会拒绝任何仍未完成的计划步骤，因此计划既是 LLM 的工作记忆，也是最终完成门槛的一部分。

实验层面的规划采用“基于当前已接受配置提出增量候选”的方式。`run_command` 要求候选工具参数中包含非空的 `hypothesis` 和 `expected_effect`；这些字段随候选配置和结果进入 `pending_run`，最终写入实验历史。系统验证参数是否合法，但不自动验证假设是否科学合理，因此报告必须把假设与观测结论区分开。
 
#### Tool Execution

模型响应首先由 `parse_tool_call()` 解析。若模型没有调用工具、一次调用多个工具或产生无效 JSON 参数，Agent 会构造错误观察并要求下一轮纠正。合法调用的参数会在执行前写入 `tool_calls/call_XXX_<tool>/arguments.json`，随后由 `ToolHarness.execute()` 分派。

文件操作通过解析后的绝对路径进行 containment 检查。命令操作则验证 Python 可执行文件、脚本路径、`--config`、`--run-id` 和 `--timeout-seconds`；包含管道、重定向或命令连接符的参数会被拒绝。实际实验使用 `subprocess.run(..., shell=False, capture_output=True)`，因此模型不能借助 shell 语法组合额外命令。

候选运行前，Harness 比较磁盘上的 `current_config.yaml` 与当前已接受配置：没有变化、搜索空间外变化、非法候选值或重复指纹都会被拒绝。通过校验后才增加 iteration 计数、创建实验目录并保存运行前快照。
 
#### Observation

工具结果会被序列化为 `role="tool"` 消息追加到对话，其中包含成功标志、结构化结果、错误信息和建议的 `next_action`。对于实验运行，观察包括：

- 外层进程返回码和是否超时；
- 任务生成的 metrics 对象；
- 独立校验发现的问题；
- stdout/stderr 各最多最后 2,000 个字符；
- candidate 的 snapshot ID；
- 建议执行评估或回滚的下一动作。

完整 stdout/stderr 不进入终端和 LLM 上下文，而是保存在相应实验目录。这样，模型获得足够的诊断信息，审计者仍可读取完整证据，同时减少日志噪声和潜在敏感内容扩散。
 
#### Validation

每次实验完成后，Harness 不依赖进程输出文本中的自报结果，而是按配置的 `metrics_path` 重新读取 `summary.json`。验证要求进程未超时、返回码为 0、metrics 是合法 JSON 对象、`status="success"`，且主指标存在、可转换为数值并为有限值。任一条件不满足都会在 `verification_errors` 中留下证据，candidate 随后只能进入回滚路径。

对于有效指标，`compare_metrics()` 使用当前 `best_metrics` 而非上一个任意实验作为比较基准。当前配置设定 `direction=maximize`、`min_improvement=0.0`：candidate 的 `mean_score` 严格提高才直接接受；当差异不超过阈值时，才比较 `mean_reward`。这意味着 `mean_reward` 是同分判定与诊断指标，而不是可覆盖主指标下降的第二目标。
 
#### Decision and State Update

LLM 使用 `evaluate_result` 提交 `accept` 或 `rollback` 以及理由和证据。Harness 同时计算自己的决定：

- LLM 请求接受且独立验证也接受时，candidate 成为新的 `current_config` 和 `best_config`；
- LLM 保守地回滚一个实际改善候选是允许的；
- LLM 请求接受下降或无效结果时，工具调用被拒绝并要求纠正；
- 回滚决定产生后，在恢复完成前不允许启动新 candidate 或结束运行。

`restore_snapshot` 会校验 snapshot ID、将快照与权威已接受配置比较、复制到 `current_config.yaml` 并重新读取确认内容一致。候选随后从 `pending_run` 移至 `history`，恢复事件记录迭代号、原因、快照和 SHA-256。接受候选同样会进入 `history`，但不会产生恢复记录。
 
### State Tracking and Traceability

系统以 `state` 作为单次运行的权威状态机，并在每轮工具结果处理后覆盖写入 `state.json`。核心字段及含义如下：

| 状态域 | 代表字段 | 作用 |
|---|---|---|
| 生命周期 | `status`、`phase`、`started_at`、`finished_at`、`stop_reason` | 描述运行所处阶段和终止原因 |
| 计数与预算 | `iteration`、`llm_call_count`、`tool_call_count`、`limits`、`remaining_budget` | 防止无限循环并显示剩余额度 |
| 计划 | `plan.steps[].status/evidence/next_action` | 连接任务分解、证据和完成检查 |
| 基准与最优 | `baseline`、`current_config`、`best_config`、`best_metrics`、`best_source` | 保存当前可接受的实验事实 |
| 候选事务 | `pending_run`、`tested_fingerprints`、`history` | 防止重复实验并保证候选先评估后提交 |
| 故障恢复 | `failures`、`last_tool_error`、`recovery_events` | 记录工具错误、执行失败和已验证回滚 |
| 完成条件 | `requirements`、`agent_summary` | 记录最少迭代、恢复证据和报告校验结果 |

持久化记录支持三个层次的追踪：

1. **重建模型视角：** `request.json` 可回答模型实际看到了什么，`context.json` 解释其摘要和裁剪过程，`messages.jsonl` 则保留完整未裁剪交互；
2. **重建系统动作：** `trajectory.jsonl` 与工具目录可回答“调用了什么、返回了什么”；
3. **重建实验事实：** baseline/iteration 下的配置、metrics、execution、日志和 Q-table 可回答“实际运行产生了什么”。

`research_agent_003` 的轨迹中共有 39 个 `tool_call` 事件和 1 个 `stop` 事件；工具分布为：6 次 `run_command`（1 baseline + 5 candidates）、5 次 `evaluate_result`、2 次 `restore_snapshot`、8 次 `read_file`、1 次 `list_files`、7 次 `write_file`、1 次 `submit_plan`、8 次 `update_plan` 和 1 次 `finish`。状态最终显示 `requirements` 三项全部为 `true`，与 `completed / agent_finish` 相互一致。
 
## Key Design Decisions and Safety Mechanisms
 
### Separation of Reasoning and Execution

系统有意将非确定性的研究推理与确定性的环境执行分开。LLM 可以决定“下一步值得尝试什么”和“如何解释观察”，但不能直接调用文件系统或 `subprocess`；所有副作用必须经过 Tool Harness。这样做的直接收益是：模型输出即使出现幻觉、遗漏或格式错误，也只能形成一个待校验请求，不会自动成为系统事实。

这种分离还使测试能够绕过真实网络模型，用脚本化 LLM 固定动作序列，同时保留真实工具、状态迁移和子进程。生产逻辑与测试逻辑共享同一个 `run_agent()` 和 `ToolHarness`，测试替换的只是 `chat_fn`，因此可以稳定验证协议，而不把模型随机性混入控制层测试。
 
### Restricted Tool Execution

工具系统采用 allowlist 而非 denylist。只读范围、唯一写目录和唯一允许的实验脚本都来自运行配置，路径在比较前会解析为规范绝对路径。`run_command` 不接受一段 shell 字符串，而是接收参数数组，并进一步禁止 `|`、`;`、`&&`、`||`、`>` 和 `<` 等控制符。这些机制共同降低路径穿越、命令拼接、覆盖源文件和跨运行污染的风险。

限制也作用于实验语义：baseline 必须逐项等于声明配置；candidate 必须与当前已接受配置不同，且每项变化必须出现在搜索空间中；配置指纹不得重复；candidate run ID 必须与下一个迭代号一致。在权限约束之外，这些规则保证“第 N 次实验”在审计意义上确实是一个合法、唯一的候选。
 
### Independent Result Validation

系统坚持“LLM 不是验证器”。候选进程可能打印成功文本，LLM 也可能宣称指标改善，但 Harness 仍会单独检查返回码、超时状态和磁盘 metrics。指标比较使用结构化数据和固定规则，不解析自然语言结论；最终报告也必须包含规定章节、每个 `Iteration N`、精确最佳指标以及存在时的 recovery 讨论。

当前测试套件提供了对应的回归保障。2026-08-13 在 Python 3.11.14、NumPy 2.4.1、pygame 2.6.1、PyYAML 6.0.3、OpenAI SDK 2.54.0 环境下执行：

```powershell
python -m unittest discover -s tests -v
```

共 12 项测试全部通过。覆盖内容包括：原生工具调用解析与 Kimi 推理字段回放、上下文配置校验、权威状态摘要、assistant/tool 原子窗口、最近工具大载荷压缩与禁用兼容模式、OpenAI SDK 参数、进度 heartbeat 与内容防泄漏、配置最低要求、独立执行/指标比较、报告证据校验、越界写入和提前结束拒绝，以及包含 baseline、3 次 candidate、真实回滚和裁剪审计的端到端状态机。该测试结果支持控制逻辑按预期工作，但不等价于证明所有操作系统、模型 provider 或长时间故障模式均已覆盖。
 
### Rollback and Recovery Mechanism

candidate 执行被建模为一个事务。运行前，Harness 把当前已接受配置保存为 `snapshots/iteration_NN_before.yaml`，candidate 运行后进入 `pending_run`。如果结果下降、执行无效或 LLM 主动选择回滚，系统不会立即继续实验，而是要求显式调用 `restore_snapshot`。

恢复操作包含四项检查：待决 candidate 已完成评估、评估决定确实为 rollback、请求的 snapshot ID 与待决状态一致、快照内容与内存中的权威当前配置一致。复制后再次读取 YAML 比较内容，并记录 SHA-256。只有完成这些步骤，candidate 才进入历史，`pending_run` 才清空。

正常完成还可要求真实恢复证据。默认 `evidence.require_recovery_event=true`，且只有独立验证器同样认为 candidate 应回滚时，恢复事件的 `qualifies` 才为真。普通工具参数纠错虽然也记作 recovery event，但不能满足这一研究要求。这避免了通过故意制造非法命令来伪造“失败恢复能力”。
 
### Termination and Budget Control

运行结束不是自由文本响应，而是一次受控的 `finish` 工具调用。正常完成需要同时满足：不存在 `pending_run`、全部计划步骤已完成、candidate 数达到最低要求、存在合格的实验回滚证据、最终报告通过校验。任一条件不满足都会作为工具错误返回给 LLM，允许其在剩余预算内补救。

安全终止与正常完成严格区分。工具调用或墙钟预算耗尽产生 `incomplete_requirements`；连续失败达到阈值或出现未捕获异常产生 `failed`。不论何种退出路径，`finally` 都会更新时间、持续时长、剩余预算和 stop 事件并保存 `state.json`。因此，“进程停止”不会被错误报告为“研究成功”。

需要指出，当前预算中的 `agent_steps` 实际与 `llm_call_count` 同步，并没有独立的 `max_agent_steps` 配置；真正触发终止的是工具次数、candidate 次数、连续失败、墙钟时间和单次实验时间。公开描述时不应把 `agent_steps` 误写成一个独立受控预算。
 
## Experiments and Results

### Experimental Setup

正式结果来自 `runs/research_agent_003`。该运行开始于 2026-08-12 21:40:38（UTC+8），结束于 22:02:58，总持续时间 1,339.79 秒（约 22 分 20 秒），最终状态为 `completed`，终止原因为 `agent_finish`。运行共完成 39 次 LLM 调用、39 次工具调用、1 次 baseline 和 5 次 candidate；`state.json` 中 `total_failures=0`。这里的 failure 计数表示 LLM/工具/执行协议错误，不包括指标下降但执行本身成功的候选。

任务采用离散 Q-learning。输入 Q-table 形状为 `(30, 100, 100, 2)`、数据类型为 `float64`；状态由下一管道的水平距离 `dx`、相对管道间隙中心的垂直距离 `dy`、垂直速度 `vy` 离散化得到，最后一维对应“不拍翼/拍翼”两个动作。训练使用：

```text
Q(s_t, a_t) ← (1 - α)Q(s_t, a_t)
              + α[r_t + γ max_a Q(s_{t+1}, a)]
```

学习率按 `max(learning_rate_min, learning_rate × learning_rate_decay^counter)` 衰减；训练阶段使用 epsilon-greedy，评估阶段将 epsilon 固定为 0。训练启动时采用启发式 Q 值初始化，评估阶段加载本轮训练产生的 `q_table.npy`。pygame 的视频和音频驱动设置为 `dummy`，渲染关闭，训练与评估帧率均为 1500。

所有实验使用 `seed=0` 和 100 个评估回合。baseline 训练 1000 回合；candidate 可以在搜索空间内选择 1000、1500 或 2000 回合。状态离散化参数固定为 `dx_dy_bin_size=25`、`velocity_bin_size=2`，满足“不改变 Q-table 状态空间”的目标约束。

本节报告的指标直接来自各运行的 `summary.json`，并与最终 `state.json` 的 baseline/history 交叉核对。由于任务适配器当前把 `min_score` 错误赋值为 `mean_score`，本文不把 `min_score` 列为有效实验指标。
 
### Baseline and Iterative Experiments

baseline 配置为：1000 个训练回合、`learning_rate=0.6`、`discount_factor=0.95`、固定 `epsilon=0.001`、`sample_t=3`、`rewards.alive=0.05`。它取得 `mean_score=5.20`、`std_score=4.67`、`max_score=21`、`mean_reward=65.7870`，已略高于目标值 5.0，因此后续目标是继续提高而非仅达到阈值。

完整候选结果如下。表中的“相对基准”指每轮运行前的当前最佳配置，而不是固定 baseline。

| 阶段 | 相对当前最佳的配置变化 | `mean_score` | `std_score` | `max_score` | `mean_reward` | 训练时间 (s) | 决策 |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline | 声明的基准配置 | 5.20 | 4.67 | 21 | 65.7870 | 17.67 | 建立基准 |
| Iteration 1 | `training_episodes: 1000→2000`; `discount_factor: 0.95→0.99` | 10.47 | 8.24 | 44 | 218.2885 | 43.11 | Accept |
| Iteration 2 | `sample_t: 3→2` | 10.96 | 10.79 | 54 | 196.8880 | 31.71 | Accept |
| Iteration 3 | `rewards.alive: 0.05→0.1` | 3.92 | 3.45 | 13 | 43.2380 | 31.89 | Rollback |
| Iteration 4 | `epsilon_start: 0.001→0.01`; `epsilon_end: 0.001→0`; `epsilon_decay: 1→0.995` | 6.71 | 5.25 | 24 | 85.8205 | 30.06 | Rollback |
| Iteration 5 | `learning_rate: 0.6→0.4` | **14.31** | 16.39 | **92** | **287.3115** | 44.31 | Accept |

逐轮分析如下：

- **Iteration 1：** 同时增加训练回合并提高折扣因子，`mean_score` 从 5.20 增至 10.47，Harness 接受该候选。由于两个因素同时改变，不能从本次结果单独判断增益来自训练时长、折扣因子或二者交互。
- **Iteration 2：** 在已接受配置上把 `sample_t` 从 3 降至 2，`mean_score` 小幅增至 10.96，因此按确定性规则接受；但 `mean_reward` 从 218.2885 降至 196.8880，且分数标准差为 10.79。该 0.49 增益小于观测波动尺度，不应被解释为已证明的稳健提升。
- **Iteration 3：** 将存活奖励提高到 0.1 后，`mean_score` 降至 3.92，低于当前最佳 10.96。系统回滚至 Iteration 2 配置。数据证明该设置在本次协议下表现较差，但不足以证明某个具体策略行为是原因。
- **Iteration 4：** 从恢复后的 Iteration 2 配置出发，引入从 0.01 退火至 0 的探索率，`mean_score=6.71`，仍低于 10.96，因此再次回滚。该结果表明这组联合探索参数在本次有限训练预算下未改善指标，无法分别归因于起始值、终止值或衰减率。
- **Iteration 5：** 从再次恢复的 Iteration 2 配置出发，只把初始学习率从 0.6 降至 0.4，`mean_score` 提高到 14.31，Harness 接受为最终最佳。虽然平均分和平均奖励均提高，但 `std_score` 也从 10.79 增至 16.39，因此不能声称方差降低或策略更加稳定。

5 次 candidate 中有 3 次接受、2 次回滚。所有 6 次实验进程（含 baseline）的 `status` 均为 `success`、train/eval return code 均为 0，且没有 verification error。两次“失败”是合法候选的性能回退，不是命令错误或进程故障。
 
### Best Result

最终最佳结果来自 Iteration 5，完整关键配置为：

| 参数 | 最佳值 |
|---|---:|
| `training_episodes` | 2000 |
| `evaluation_episodes` | 100 |
| `learning_rate` | 0.4 |
| `learning_rate_min` | 0.1 |
| `learning_rate_decay` | 0.9995 |
| `discount_factor` | 0.99 |
| `epsilon_start / end / decay` | 0.001 / 0.001 / 1.0 |
| `sample_t` | 2 |
| `rewards.alive` | 0.05 |
| `q_init` | `heuristic` |
| `seed` | 0 |

与 baseline 相比：

- `mean_score`：5.20 → **14.31**，绝对增加 **9.11**，相对增加 **175.2%**，为 baseline 的 **2.752 倍**；
- 相对配置目标 5.0：高出 **9.31**，为目标的 **2.862 倍**；
- `mean_reward`：65.7870 → **287.3115**，绝对增加 **221.5245**；
- `max_score`：21 → **92**；
- 最终 `std_score=16.39`，说明评估回合之间仍存在较大离散程度。

因此，在本次固定 seed、固定状态空间和 100 回合评估协议下，可以可靠地陈述“最佳已观测候选明显优于 baseline”。不能据此陈述该配置在多随机种子、不同机器或所有初始化条件下都具有同等增益。
 
### Execution Trace Example

`research_agent_003` 的关键轨迹展示了系统如何把模型决策转换为受控状态迁移：

```text
call 01  submit_plan                         → 计划建立
call 02–10  read/list files                 → 收集目标、配置和任务代码
call 12  write current_config.yaml          → 写入 baseline
call 13  run baseline                       → mean_score 5.20
call 15–17  write/run/evaluate Iteration 1  → 10.47, accept
call 19–21  write/run/evaluate Iteration 2  → 10.96, accept
call 23–25  write/run/evaluate Iteration 3  → 3.92, rollback
call 26  restore iteration_03_before.yaml   → verified=true
call 28–30  write/run/evaluate Iteration 4  → 6.71, rollback
call 31  restore iteration_04_before.yaml   → verified=true
call 33–35  write/run/evaluate Iteration 5  → 14.31, accept
call 37  write final report                 → 报告落盘
call 39  finish                             → completed / agent_finish
```

两次恢复使用的快照内容哈希均记录为 `a8ce474372b7b6bbc7b3eab8838f625ecfd27f1a14dad16d573ac5cf0e673f4c`，对应已接受的 Iteration 2 配置。最终状态显示 5 个候选全部进入历史、`pending_run=null`，最优来源为 `iteration_05`，三项完成要求 `min_iterations_met`、`recovery_demonstrated` 和 `report_verified` 均为 `true`。
 
## Failure Cases, Limitations, and Future Improvements
 
### Failure Cases and Recovery

#### Observed Performance Regressions

已完成运行没有发生 LLM 请求失败、工具协议错误、子进程异常或指标校验错误；`total_failures=0`。实际观察到的失败是两个执行成功但性能下降的 candidate：

- **Iteration 3：** `rewards.alive=0.1`，`mean_score` 从当前最佳 10.96 降至 3.92。`evaluate_result` 返回 rollback，随后恢复 `iteration_03_before.yaml`；
- **Iteration 4：** 联合改变 epsilon 起始值、终止值和衰减率，`mean_score` 为 6.71，仍低于 10.96。系统恢复 `iteration_04_before.yaml`。

两次 `restore_snapshot` 均返回 `restored=true, verified=true`，并恢复到相同的已接受 Iteration 2 配置。之后 Iteration 5 从干净的恢复状态出发，而不是建立在回退候选上。这证明快照机制在本次真实性能回退场景中发挥了作用。

#### Handled Protocol and Execution Failures

以下故障路径由代码处理，并有部分自动测试覆盖，但没有在 `research_agent_003` 中实际触发：

| 故障类型 | 系统响应 | 是否在当前测试覆盖 |
|---|---|---|
| 模型未调用工具或一次调用多个工具 | 写入失败记录，返回纠正消息 | 原生调用协议有覆盖 |
| 工具 JSON、路径或参数非法 | 返回 `ok=false`，累计连续失败 | 越界写入有覆盖 |
| baseline 不一致或 candidate 越界/重复 | 拒绝实验，不启动子进程 | 由 Harness 校验逻辑覆盖，组合场景仍可扩充 |
| 子进程超时、非零返回码或 metrics 缺失 | 写入 execution failure，要求评估并回滚 | 非零返回码验证有覆盖 |
| LLM 接受下降结果 | `evaluate_result` 拒绝接受 | 独立比较逻辑有覆盖 |
| 过早调用 `finish` | 返回缺失条件，不结束运行 | 有覆盖 |
| 网络/API 错误 | 记录 LLM failure，在连续失败预算内重试 | SDK 错误转换存在，真实网络故障未做集成测试 |
| 未捕获异常 | 标记 `failed / unrecoverable_error` 并在 `finally` 落盘 | 缺少故障注入测试 |

工具成功后连续失败计数会清零，并将此前工具纠错记录为不满足研究恢复要求的 `tool_correction`。只有真实无效/下降 candidate 的已验证回滚才可满足 `recovery_demonstrated`。
 
### Current Limitations

#### Experimental Validity

- **单一随机种子：** 所有实验只使用 `seed=0`，不能估计跨种子均值、置信区间或失败概率；
- **评估方差较高：** 最佳结果 `std_score=16.39`，高于其 `mean_score=14.31`。当前 100 回合样本虽能描述本次评估，却不足以支持细小差异的稳健排序；
- **自适应且非完全消融：** Iteration 1 同时改变训练回合与折扣因子，Iteration 4 同时改变三个 epsilon 参数，无法识别单因素因果效应；
- **接受阈值过于宽松：** `min_improvement=0.0` 会接受任意严格正增益，即使增益远小于评估噪声，例如 Iteration 2 的 +0.49；
- **搜索预算有限：** 只允许 5 个候选和离散搜索值，没有重复评估同一配置，也没有系统探索参数交互；
- **baseline 已超过目标：** 目标值 5.0 对当前 baseline 不具挑战性。本次贡献是进一步优化，而不是从未达标状态实现达标。

#### Metrics and Reporting

- `tasks/flappy_qlearning/run.py` 的 `summarize()` 当前把 `min_score` 写成 `mean_score`，因此产物中的 `min_score` 不可信；
- `max_score` 对极端回合敏感，不能代替均值和分布统计；当前没有 median、分位数或置信区间；
- `validate_report()` 只检查必需章节名称、Iteration 文本、精确最佳指标和恢复关键词，不能验证报告内每个参数、百分比或因果解释；
- LLM 生成的实验解释可能超出证据。例如从一次超参数变化推断具体策略行为，需要轨迹分析或受控重复实验，现有 Harness 不会自动识别这类过度归因。

#### Context and Durability

- 已实现权威状态摘要和字符预算滑动窗口，但 24,000 字符只是 token 用量的近似代理，且状态摘要本身尚无独立体积上限；
- 窗口最多保留 4 个近期回合，较早的源代码观察和研究理由可能被移出请求；当前摘要保存事实状态，但不替代长期语义记忆；
- `messages.jsonl`、`state.json` 和每轮请求虽可审计，但 `run_agent()` 没有从已有记录恢复中断运行的入口；
- `state.json` 使用普通覆盖写而非临时文件加原子替换，极端掉电或进程中止可能留下部分写入；
- 运行记录会保存完整 prompt、模型回复和工具返回。虽然 API key 不落盘，公开分享 run 目录前仍应进行内容与隐私审查。

#### Reproducibility and Scope

- 依赖文件只给出版本范围，没有 lockfile 或容器镜像；原始完成运行也没有把 Python 版本、依赖精确版本、OS、CPU/GPU 信息写入 manifest；
- 绝对训练/评估时间受硬件和系统负载影响，只能用于描述本次运行；
- 当前只支持 OpenAI-compatible Chat Completions provider，并对 Kimi 的 `reasoning_content` 有特定兼容逻辑；
- 工具层是应用级 allowlist，不是操作系统级容器沙箱。被允许的实验脚本仍在宿主 Python 进程权限范围内运行；
- 当前结论只适用于 Flappy Bird 离散 Q-learning 示例，不能直接推广到深度学习、大规模并行实验或高风险生产系统。
 
### Future Improvements

建议按以下优先级推进：

1. **修复指标正确性。** 修正 `min_score` 聚合；增加 median、P25/P75、完整 episode scores 或可验证摘要，并为所有数值字段增加一致性测试。
2. **升级实验判定。** 对每个候选运行多个随机种子，按跨种子均值与置信区间比较；设置高于噪声水平的 `min_improvement`，避免接受统计上不可区分的小增益。
3. **增强实验设计。** 将多参数候选拆成消融实验，或使用受预算约束的贝叶斯优化/逐次淘汰，同时保持 Harness 对候选合法性和最终指标的独立验证。
4. **记录完整环境。** 每次 run 自动写入源码 commit、dirty-worktree 标记、Python/依赖版本、平台和硬件摘要；提供 lockfile 或容器配置，使协议复现升级为环境复现。
5. **实现可恢复状态机。** 使用原子文件替换保存状态，为每轮建立一致性 checkpoint，并支持从 `state.json + messages.jsonl` 恢复；恢复时验证最后一条 tool call、`pending_run` 和磁盘产物是否一致。
6. **深化长上下文管理。** 在现有权威摘要与字符窗口之上增加模型 tokenizer 估算、摘要体积上限和 context-length error 恢复；将被淘汰的研究理由压缩为带文件引用的长期语义记忆。
7. **强化报告校验。** 从 `state.json` 自动生成结果表和关键数字，或让报告中的机器可读 front matter 与状态逐字段比对；自然语言只负责解释，不手工复制核心指标。
8. **加强隔离与资源控制。** 将实验放入容器或低权限进程，增加 CPU、内存、磁盘和子进程树限制；对公开产物执行自动密钥与隐私扫描。
9. **扩展故障测试。** 增加超时、损坏 metrics、丢失快照、状态写入中断、API 连续失败、上下文超限和跨平台路径等故障注入测试。
10. **改善运行治理。** 为未完成 run 增加明确的 `aborted`/`stale` 状态和清理策略，提供汇总索引区分 completed、incomplete 和 failed 运行，防止分析时误用半成品结果。

## Conclusion

Mini AutoResearch Agent 展示了一个边界清晰的自主实验实现：LLM 负责计划与研究判断，Python Harness 负责真实执行、约束、验证和证据持久化。其价值不在于让模型拥有无限执行权限，而在于把模型的开放式推理嵌入一个可拒绝、可恢复、可审计的确定性状态机。

唯一完成且通过全部完成条件的 `research_agent_003` 在固定 seed 和评估协议下，将 Flappy Bird Q-learning 的 `mean_score` 从 5.20 提高到 14.31，并对两个合法但退化的候选执行了经过验证的快照回滚。现有 12 项自动测试全部通过，支持核心控制逻辑以及新增权威摘要/滑动窗口机制的正确性。与此同时，单 seed、高方差、指标聚合缺陷、缺少精确 token 预算与断点恢复、环境锁定不足等问题仍限制结论强度。

因此，当前系统已经达到“小规模、受控、可审计的自主实验原型”水平，但若用于更长时间、更高成本或公开基准比较，应先完成指标修复、多种子统计验证、环境固化、原子恢复和更强隔离。本文的最佳结果应被理解为一次完整运行中的最佳已观测结果，而不是未经限定的算法性能声明。

## Evidence Index

为便于复核，核心证据位置如下：

- 系统入口与控制循环：[main.py](main.py)、[autoresearch/agent.py](autoresearch/agent.py)；
- 模型传输与工具解析：[autoresearch/planner.py](autoresearch/planner.py)；
- 工具、状态迁移、实验校验与回滚：[autoresearch/tooling.py](autoresearch/tooling.py)；
- 指标验证和比较：[autoresearch/execution.py](autoresearch/execution.py)；
- 最终报告检查：[autoresearch/report.py](autoresearch/report.py)；
- Flappy Bird train/eval 适配器：[tasks/flappy_qlearning/run.py](tasks/flappy_qlearning/run.py)；
- Q-learning 实现：[tasks/flappy_qlearning/src/flappyq.py](tasks/flappy_qlearning/src/flappyq.py)；
- 默认实验协议：[configs/autoresearch.yaml](configs/autoresearch.yaml)；
- 完成运行状态：[runs/research_agent_003/state.json](runs/research_agent_003/state.json)；
- 完整工具轨迹：[runs/research_agent_003/trajectory.jsonl](runs/research_agent_003/trajectory.jsonl)；
- baseline 和候选指标：[baseline](runs/research_agent_003/baseline/metrics.json)、[Iteration 1](runs/research_agent_003/iteration_01/metrics.json)、[Iteration 2](runs/research_agent_003/iteration_02/metrics.json)、[Iteration 3](runs/research_agent_003/iteration_03/metrics.json)、[Iteration 4](runs/research_agent_003/iteration_04/metrics.json)、[Iteration 5](runs/research_agent_003/iteration_05/metrics.json)；
- 自动测试：[tests/test_core.py](tests/test_core.py)。
