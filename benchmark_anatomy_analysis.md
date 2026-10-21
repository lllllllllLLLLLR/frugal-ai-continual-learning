# Continual Learning Agent Benchmarks: Anatomy, Evaluation, and Comparability

> 分析对象：TRACE 加上 10 个面向 language agent、memory、skill 或 harness 的 continual-learning benchmark，共 11 个。论文版本核对日期：2026-09-15。本文中的数值均为论文报告值，不代表在统一实验条件下复现得到的排名。

## 0. 先回答最终问题

**这 11 个 benchmark 的论文 headline score 不能被放在同一张排行榜中。** 它们改变的时间对象、允许更新的系统部件、反馈强度、任务依赖、重测协议和指标量纲都不同。跨论文直接比较 74.5% 与 25.4% 之类的数值没有统计或语义意义。

目前能够成立的比较分为三层：

1. **可直接比较：** 只限同一个 benchmark、同一版本 harness、同一任务序列、同一模型与预算下的不同方法。
2. **重跑后可比较：** 有共享任务或共享方法的 benchmark 子集可以在统一协议下比较。例如 AgentCL 与 Evo-Memory 共享 MMLU-Pro、BabyAI、ScienceWorld 及多种 memory baseline，但论文原始结果使用的模型、任务顺序和评价协议不同，必须重跑。
3. **只能比较设计，不能比较成绩：** 11 个 benchmark 都可以用统一的 benchmark anatomy 比较覆盖面和证据强度，但不能用原始分数比较“谁更难”或“谁的方法更好”。

在这组工作中：

- **TRACE** 最适合回答参数级顺序训练是否造成遗忘，但不是 agent benchmark。
- **AgentCL** 最完整地分离了 plasticity、stability 和 held-out generalization。
- **CL-Bench** 最清楚地用同一系统的 stateful-versus-stateless 配对差识别在线学习贡献。
- **EvoHarnessBench** 唯一把非平稳性明确放在外部 harness，而不只放在任务流或 memory 中。
- **SWE-Bench-CL** 提出了合理的 coding-CL 数据结构和指标，但当前论文的标准 harness 实验不足以形成可靠 baseline 结论。

## 1. Benchmark anatomy：应该拆成什么

一个 continual-learning benchmark 至少应写成以下六元组：

`B = <evaluation unit, stream, feedback, persistent state, re-evaluation, verifier/metric>`

| 部件 | 必须回答的问题 | 常见混淆 |
|---|---|---|
| Evaluation unit | 一个样本、一个交互 episode、一个 task，还是一个 task cohort？ | 不同论文对 task 和 instance 的定义相反 |
| Stream | 顺序来自随机排列、真实时间、难度课程、显式依赖、潜在结构还是 harness release？ | “有顺序”不等于“有可学习关系” |
| Feedback | 标签、binary correctness、自然语言 critique、执行结果、reward，还是无反馈？ | oracle feedback 与自生成 feedback 不能等价 |
| Persistent state | 参数、完整上下文、检索 memory、skill 文档、代码或 routing policy 中什么可跨任务保留？ | 把 context gain 错算成 skill learning |
| Re-evaluation | 是否重测旧任务、是否第二遍只读、是否有 held-out task、是否与 stateless twin 配对？ | 只测后期任务提升不能证明没有遗忘 |
| Verifier/metric | exact match、执行成功、partial reward、BWT/FWT、成本、安全如何计算？ | 指标同名但参照点不同 |

## 2. 总体 anatomy matrix

| Benchmark | 构成与 evaluation unit | 学习场景与持久状态 | 任务与测试类型 | 主要测量 |
|---|---|---|---|---|
| [TRACE](https://arxiv.org/abs/2310.06762) | 8 个顺序数据集；每个 5,000 train、2,000 test，总计 40k/16k；另有通用、指令跟随和安全测试 | 已知任务边界的 supervised sequential fine-tuning；主要更新参数，另含 6-shot ICL 对照 | 科学/金融/会议、中文/德文、代码补全、数学；静态 held-out 与旧任务重测 | OP、BWT、General/Instruction/Safety Delta |
| [StreamBench](https://arxiv.org/abs/2406.08747) | 7 个静态数据集被随机序列化，共 9,702 个 test-time items | online、逐 item binary feedback；prompt、memory、retriever或参数均可更新，论文 baseline 更新 prompt/memory | Text-to-SQL、Python、tool use、诊断、QA；一次连续预测流 | 全流 task-native 平均分与 streaming-over-non-streaming gain |
| [LifelongAgentBench](https://arxiv.org/abs/2505.11942) | DB 500、OS 500、KG 396，共 1,396 个 skill-grounded tasks | 固定顺序的交互式 POMDP；近期成功 trajectory 以 replay 形式跨任务保留 | SQL、Bash、SPARQL；容器执行与最终状态 verifier | success rate、replay size、输入 token/OOM 和 failure mode |
| [SWE-Bench-CL](https://arxiv.org/abs/2507.00014) | 8 个 Python repo 的 273 个 SWE-Bench Verified issue 序列 | repository chronology 加 difficulty curriculum；提议 parameter training 或 FAISS semantic memory | 真实 issue 修复、测试通过、工具效率；提议每步重测旧 issue | SR、ACC、forgetting、FT、BWT、AULC、TUE、CL-F-beta |
| [MemoryBench](https://arxiv.org/abs/2510.17281) | 11 个来源数据集拆成约 20k multilingual cases；Task Provider、User Simulator、Performance Monitor | service-time on-policy/off-policy learning；parametric 或 non-parametric memory；训练反馈与测试集分开 | open/legal/academic，四种长短输入输出组合；原生指标和 LLM judge | feedback utility、held-out task score、归一化总分、memory/inference time |
| [Evo-Memory](https://arxiv.org/abs/2511.20857) | 5 个 single-turn benchmark family（MMLU-Pro 分 3 个领域）加 4 个 interactive environment | test-time search-synthesis-evolve；正确性反馈；evolving external memory | 知识、数学、API、家居、导航、科学实验、规划 | accuracy、success/progress、steps、sequence robustness |
| [SkillLearnBench](https://arxiv.org/abs/2604.20087) | 20 个 task、100 个 instance、human skill、oracle trajectory、deterministic verifier | 每个 task 用 1 个 seed 生成 skill，再在同任务多实例测试；skills 不跨 20 个 task 连续累积 | 6 类 15 子领域的真实工具/文件/artifact task | skill quality、trajectory alignment/usage、task pass、tokens |
| [AgentCL](https://arxiv.org/abs/2606.02461) | naive 与显式 compositional streams；同一流运行两遍；另有 held-out tasks | non-parametric memory；第一遍读写、第二遍冻结只读 | coding、deep research、BabyAI、ScienceWorld、MMLU-Pro | Plasticity Gain、Stability Gain、Generalization Gain，加 native outcome |
| [CL-Bench](https://arxiv.org/abs/2606.05661) | 6 个从头设计、专家验证、含 shared latent structure 的 stateful task schedule | 同一系统逐 instance 累积任意形式状态，并与每个 instance 的 stateless twin 配对 | 频谱、代码库、流行病学、数据库、扑克、销量预测 | reward、paired gain、normalized reward/gain、stability/plasticity、美元成本 |
| [ContinualSkillBench](https://arxiv.org/abs/2608.03874) | 从约 30k 候选中构造 5 个领域流，每流 100 task；dependency graph 和 difficulty order | 三阶段 instruction-execution-reflection；context 与 skill repository 跨任务保留 | 医疗、法律、数学、金融、办公；异构自动/LLM verifier | raw/normalized reward、Seq-Ind delta、skill size/calls/reuse/quality |
| [EvoHarnessBench](https://arxiv.org/abs/2609.04280) | EOG/ALE 派生的 17 个 3-6 stage stream；802 unique tasks；520 tools、42 skills、62 agents | 外部 harness 单调扩展；deployment 无持久状态，self-evolving 模式保留 memory/prompt/code state | 企业 workflow 与 agentic software task；adapt/eval split 和 verifier | Pass、Score、FWT、BWT、final ACC、tokens/tool calls/latency |

这 11 个 benchmark 实际采用六类测试协议：

| 测试协议 | Benchmark | 能支持的结论 |
|---|---|---|
| 顺序训练后重测旧任务/基础能力 | TRACE；SWE-Bench-CL（提议） | 参数或 coding competence 是否遗忘 |
| 单遍 online cumulative evaluation | StreamBench；LifelongAgentBench；Evo-Memory | 过去反馈或经验是否改善流内后续表现 |
| train-feedback / held-out-test | MemoryBench；SkillLearnBench | memory/skill 是否超出产生它的样本而复用 |
| 同一 stream 的 two-pass frozen-memory retest | AgentCL | 写入的经验经后续更新后是否仍能检索和使用 |
| 同一系统的 stateful/stateless paired test | CL-Bench；ContinualSkillBench 的部分对照 | 增益是否来自持久状态；若加 pure ICL 对照可进一步隔离 skills |
| 随 harness stage 构造 cohort performance matrix | EvoHarnessBench | 外部 capability 扩展造成的 adaptation 与 retention 变化 |

## 3. 逐 benchmark 回答七个问题

### 3.1 TRACE

**Construction.** 8 个异构顺序任务是 ScienceQA、FOMC、MeetingBank、C-STANCE、20Minuten、Py150、NumGLUE-cm、NumGLUE-ds。每个任务平衡为 5,000 train 与 2,000 test。外部能力测试包括 MMLU、GSM8K、BBH、TyDiQA、BoolQ、PIQA，指令跟随测试使用 Self-Instruct 与 LIMA，安全测试使用 CoNa（论文 §4.1-4.3、Table 4）。

**What it measures.** 顺序目标任务的 Overall Performance 与 Backward Transfer，以及训练前后通用能力、instruction following 和 safety 的 delta。

**Learning scenario.** 有明确 task boundary 的 offline supervised task stream；SeqFT、LoRA-SeqFT、10% replay 与无参数更新的 6-shot ICL。由于输出形式跨分类、生成、代码和数学变化，它不是严格的经典 class-incremental setting，更准确是 heterogeneous task-incremental/sequential instruction tuning。

**Tasks/tests.** 静态 held-out task tests、每轮旧任务重测、通用能力测试、GPT-4 指令跟随/安全评判。

**Successful evaluation.** 目标任务 OP 高、BWT 不为负，同时 General/Instruction/Safety Delta 接近 0 或为正。只提高 OP 而大幅损害 GSM8K 或 safety 不算成功。

**Baselines.** Replay 在五个模型上取得最高 OP（51.7-56.9）且 BWT 为 +0.2% 至 +2.6%；SeqFT 的 BWT 为 -5.9% 至 -15.4%，LoRA-SeqFT 为 -9.0% 至 -45.7%。LLaMA-2-13B-Chat 经 SeqFT 后 GSM8K 从 43.14 降至 2.12，综合 General Ability Delta 为 -13.32（Table 1-2）。

**Interpretation limit.** 它能证明参数训练遗忘，但没有工具行动、environment feedback 或跨 episode agent state，不能作为 agent continual learning 的完整替代。

### 3.2 StreamBench

**Construction.** Spider 2,147、CoSQL 1,007、BIRD 1,534、DS-1000 1,000、ToolBench 750、DDXPlus 1,764、HotpotQA 1,500。静态 test set 按固定 seed 随机排列成流，并用五个 shuffle seed 做稳健性分析（§3、Table 1）。

**What it measures.** agent 能否利用前面 item 的 binary correctness feedback，提高整个输入-反馈序列的累计满足率。

**Learning scenario.** online instance-incremental learning；任务边界是 dataset 级已知，任务内 item 无显式依赖。允许更新 prompt、retriever、memory 或参数，论文实验主要测试 prompt/memory 与 multi-agent shared memory。

**Tasks/tests.** SQL execution accuracy、code pass@1、API/tool accuracy、diagnosis accuracy、QA exact match。没有第二遍旧任务重测或独立 held-out generalization test。

**Successful evaluation.** 在相同模型、顺序和调用成本下，streaming 方法应显著优于 zero/few-shot 等 non-streaming twin，并在多个 shuffle seed 下保持增益。

**Baselines.** 三个模型平均结果中，MAM-StreamICL 相对 zero-shot 在七个数据集全部提高：Spider 67.89→75.69、CoSQL 50.55→55.17、BIRD 29.60→36.38、DS-1000 37.70→43.10、ToolBench 61.38→75.87、DDXPlus 52.85→83.50、HotpotQA 48.49→55.20（Table 2）。

**Interpretation limit.** 随机序列没有保证 earlier-to-later transfer；全流平均分可以证明反馈有用，却不能单独识别 retention、forgetting 或抽象能力获得。

### 3.3 LifelongAgentBench

**Construction.** 500 DB tasks 覆盖 22 SQL skills，500 OS tasks 覆盖 29 Bash skills，396 KG tasks 覆盖 7 类 SPARQL/action skills；任务生成后用执行结果、hash/checksum、输出比对和 10% 人工抽查验证（§4、Table 7）。

**What it measures.** 在长序列交互任务中，过去成功轨迹是否能提高当前任务 success rate，以及 replay 数量、任务难度、轨迹长度和 context budget 如何影响收益。

**Learning scenario.** 固定顺序、明确环境、稀疏终局 reward 的 episodic experience replay。任务容器/表状态按协议 reset，跨任务主要保留 recent successful trajectories，而非模型参数。

**Tasks/tests.** DB query/state mutation、OS command sequence、KG query/action sequence；自动 verifier 返回成功/失败，并记录未提交、格式错误、context overflow 等 failure modes。

**Successful evaluation.** 在同一 backbone 上，replay 应高于 Exp=0，增益应在不同难度和环境保持，同时不以 OOM 或不可接受 token cost 换取成功率。

**Baselines.** Llama-3.1-8B 的 DB success 从 Exp=0 的 0.19 增至 Exp=64 的 0.78；OS 从 0.43 在 Exp=4/16 达到 0.50 后下降；KG 从 0.28 在 Exp=1 达到 0.35，但 Exp≥16 OOM（Table 2）。Qwen2.5-32B 在 DB 无 replay 已为 0.82，加入 replay 多数下降，说明 replay gain 强烈依赖 backbone（Table 3）。

**Interpretation limit.** 主实验没有构造 performance matrix 或重测旧任务，因此“积累/迁移”证据强于普通静态测试，但“避免 catastrophic forgetting”的直接证据较弱。

### 3.4 SWE-Bench-CL

**Construction.** 从 SWE-Bench Verified 选择至少有 15 个 issue 的 8 个 Python repo，共 273 tasks。issue 主要按创建时间，再按人工 fix-time 难度形成 curriculum；用 gold patch 的重叠文件标注潜在 dependency（§3.2、Table 1）。

**What it measures.** coding agent 在持续接触同一 repo 后的 issue success、迁移、遗忘、学习曲线和工具效率。

**Learning scenario.** chronological repository stream；论文同时设想顺序 fine-tuning 与 test-time semantic memory。其 dependency proxy 是修改文件重叠，不保证语义复用。

**Tasks/tests.** 真实 GitHub issue 修复，FAIL_TO_PASS/PASS_TO_PASS tests，agent shell/tool interaction。论文提出在每个学习步后对所有已见任务形成 `a[i,j]` performance matrix。

**Successful evaluation.** 新 issue 测试通过；ACC/AULC 高；forgetting 低；FT/BWT 为正；CL-F-beta 在 plasticity 与 stability 之间平衡；TUE 显示工具使用没有恶化。

**Baselines.** 作者使用原 SWE-Bench harness 比较 independent 与 semantic-memory 条件时，pass rate 通常低于 8.5%，memory 大多持平或略差；由于 container/base version/test script 不匹配和大量执行失败，论文明确称结果 inconclusive（§5、Figure 3-4）。完整 agentic framework 与 CL metrics 在该版本仍是 proposed experiments，而非已验证 leaderboard。

**Interpretation limit.** 不应把 <8.5% 当成可靠方法排名，也不应声称 benchmark 已经实证验证了所提出的 CL-Score。

### 3.5 MemoryBench

**Construction.** 11 个 public dataset 被整理为约 20k cases，并按 open-domain、legal、academic，以及 LiSo/SiLo/LiLo/SiSo 四种输入输出长度组合组织；框架包含 Task Provider、LLM-as-user simulator 和 Performance Monitor。各 partition 按 4:1 分 train/test，feedback 只在 train 产生（§2.2-3.2、Table 3）。

**What it measures.** LLM system 能否把 corpus 的 declarative memory 与服务期用户反馈形成的 procedural memory 用于 unseen test cases；同时比较 memory architecture 的效率和泛化。

**Learning scenario.** service-time continual learning，包含 off-policy 预生成 feedback log 和 on-policy 逐 batch 交互两种设置；可更新 parameter 或 external memory。反馈包括 verbal critique、like/dislike、copy 等显式/隐式 action。

**Tasks/tests.** 长对话记忆、中文法律摘要/判决/QA、科研 limitation/idea、长文写作、新闻等；采用原生 F1、accuracy、ROUGE-L、METEOR、多指标 LLM judge，再做 dataset 内 min-max 或 z-score 聚合。

**Successful evaluation.** 相同 backbone 和初始 corpus 下，memory+feedback 应在 held-out test 上优于 Vanilla 和 no-feedback control，并报告 memory construction 与 inference time；应跨 domain/format 稳定，而非只在 Locomo 有效。

**Baselines.** Qwen3-8B off-policy verbal-feedback 设置中，没有统一赢家。Embed-S 在 Open/Academic/Legal 为 0.6582/0.7185/0.4806，Vanilla 为 0.6523/0.7154/0.4637；BM25-M、A-Mem、Mem0、MemoryOS 在若干 partition 反而低于 Vanilla，Mem0 对长输入还缺失结果（Table 10-11）。

**Interpretation limit.** min-max 和 z-score 依赖参评系统集合，不适合跨论文复制 headline；simulated-user validity、截断策略和 off-policy log 也必须一起控制。

### 3.6 Evo-Memory

**Construction.** single-turn 包含 MMLU-Pro（Economics/Engineering/Philosophy）、GPQA-Diamond、AIME-24、AIME-25、ToolBench；interactive 包含 ALFWorld 134、BabyAI 112、ScienceWorld 90、PDDL 60。静态数据被转成固定顺序 memory evolution streams（§4.1、Appendix F）。

**What it measures.** memory system 在 search-predict/evolve 循环中检索、综合、压缩/修订经验的能力，以及这些操作对正确率、目标完成和步骤效率的影响。

**Learning scenario.** test-time continual non-parametric memory；每步使用 correctness feedback，默认 top-k=4、同一 encoder 和 prompt budget。既有 single-turn QA，也有长 horizon environment interaction。

**Tasks/tests.** single-turn exact match/API accuracy；multi-turn success、partial progress；另测 steps、easy→hard/hard→easy 顺序与失败经验写入的鲁棒性。

**Successful evaluation.** 相同 backbone 与 retrieval budget 下应优于 no-memory/history baseline，减少完成步骤，并对顺序变化和错误经验保持稳健。

**Baselines.** Claude 3.7 Sonnet 在四个 interactive environment 的无 memory baseline 平均 S/P 为 0.24/0.52，ReMem 为 0.78/0.91；例如 ALFWorld 0.18/0.49→0.92/0.96，ScienceWorld 0.10/0.53→0.62/0.89（Table 1b）。

**Interpretation limit.** 论文没有像 AgentCL 那样重测旧任务或设置真正 held-out transfer；部分流只是固定顺序的静态 benchmark，因而更强地证明 memory-assisted test-time adaptation，而不是完整 retention/generalization。

### 3.7 SkillLearnBench

**Construction.** 20 个 multi-step task、100 instances，覆盖 6 类 15 子领域。每个 instance 在固定 agent 的 10 次 no-skill 运行中 pass rate 必须 ≤0.5，同时 human-authored skill 至少成功一次。每个 task 有 deterministic verifier、human skill 与成功 oracle trajectory（§3.1、Table 1）。

**What it measures.** 从一个 seed instance 生成的 procedural skill 是否内容正确、可执行、安全、会被 agent 实际调用，并能复用于同一 task 的其他实例。

**Learning scenario.** non-parametric skill generation：One-Shot、Self Feedback、Teacher Feedback、Skill Creator。固定 Claude Sonnet 4.6 solver，变化的是 skill-generation LLM/algorithm。

**Tasks/tests.** 软件工程、搜索、生产力、数据分析、创作和本地工具任务；Level 1 测 coverage/executability/safety，Level 2 测 trajectory alignment/skill usage，Level 3 用 verifier pass 和 tokens。

**Successful evaluation.** generated skill 应在 held-out instances 上稳定超过 no-skill，接近 human-authored ceiling；不仅文本分高，而且被调用、轨迹正确、最终 verifier 通过且成本合理。

**Baselines.** 六个 skill-generation LLM 平均：No Skill accuracy 10.17%，One-Shot 30.44%，Self Feedback 31.08%，Teacher Feedback 27.47%，Skill Creator 27.33%，Human-authored 74.50%；Self Feedback 同时使用最少 solving tokens（390k，总计口径；Table 2）。

**Interpretation limit.** skills 在每个 task 内生成并测试，不在 20 个 task 之间形成一个持续增长的共享 skill repository；因此它主要是 skill acquisition/reusability benchmark，而不是完整 cross-task lifelong sequence。

### 3.8 AgentCL

**Construction.** 以同一任务来源构造 naive 与 compositional stream：CodeEval-Pro 为 48 subtask→48 complex task，BrowseComp+ 为 308 synthesized subtask→100 complex task，BabyAI 为 28→28；ScienceWorld 90 与 MMLU-Pro 300 作为 naive streams；HumanEval-Pro 120 用于 coding held-out（§3.2-4.2、Table 1）。

**What it measures.** 是否从 earlier task 获得对 later task 有用的经验（PG），经验写入并经后续更新后是否仍可用（SG），最终 memory 是否泛化到 unseen task（GG）。

**Learning scenario.** 无 ground-truth guidance 的 non-parametric memory。第一遍允许 retrieve/update，第二遍同序列只读冻结 memory；另在 memory 构建后评估 held-out。

**Tasks/tests.** 代码执行、离线 deep research evidence QA、BabyAI/ScienceWorld interaction、MMLU-Pro；native accuracy/success/progress 加 PG/SG/GG。

**Successful evaluation.** 不能只看 PG；强 CL 方法应同时有正 PG、非负 SG、正 GG，并在 naive stream 不被 irrelevant memory 干扰。

**Baselines.** CodeEval-Pro compositional stream 上，MemProbe 第一遍 66.7%，PG +21.9 pp，但第二遍 SG -2.1 pp、held-out GG -0.8 pp；ExpRAG PG +17.7、SG 0、GG -4.2；ReMem PG +13.5、SG -2.0、GG -5.0（Table 10）。BrowseComp+ 上 MemProbe PG +40、SG -1；ReMem PG +26、SG -10（Table 11）。结果直接说明 acquisition gain 不等于稳定或泛化。

**Interpretation limit.** compositional stream 有意放大可复用关系，适合诊断机制，但不代表自然部署中这种关系的频率；naive 与 compositional 结果必须并列报告。

### 3.9 Continual Learning Bench / CL-Bench

**Construction.** 六个 expert-validated task schedule：Blind Spectrum Monitoring N=90、Codebase Adaptation N=19、Cohort Studies N=20、Database Exploration N=40、Exploitable Poker N=120、Sales Prediction N=12。每个 task 必须满足 headroom、shared latent structure、learning mechanism，并由 2 位作者和 2-3 位领域专家审查（§3、Figure 3）。

**What it measures.** end-to-end AI system 是否因在线经验而提升，而不只是基础模型本来更强。核心是同一系统、同一 instance 的 stateful reward 减 stateless reward。

**Learning scenario.** task 内 online stateful learning；状态形式不受限，可为完整 context、notepad、semantic memory、playbook 或 agent workspace。variant boundary 引入 concept drift，并用于 stability/plasticity decomposition。

**Tasks/tests.** 频谱 IoU、修 issue 的 step efficiency、cohort survival 的 KL gain、SQL exploration efficiency、poker profit、forecast WAPE；全部自动结构化评分，通常 5 rollouts。

**Successful evaluation.** 需要同时报告高 absolute reward、正 paired gain、跨 variant 的 stability、variant 内 plasticity 和 cost。单纯 stateful 分高但不高于自己的 stateless twin，不算学到了。

**Baselines.** Full-context ICL + Claude Sonnet 4.6 的 aggregate normalized reward 22.3±4.1%、normalized gain 25.4±3.6%、平均 cost $30.4，为论文总榜第一。Claude Code + Sonnet 4.6 为 19.0/23.9/$38.6；ACE + GPT-5.4 为 4.6/8.6/$62.8。简单 ICL 整体优于专门 memory systems（Table 1）。

**Interpretation limit.** 六个 task 的 reward 经过 headroom normalization 才能汇总；其总分不能与 accuracy benchmark 直接比较，且专家设计的 latent structure 提高因果识别但限制任务覆盖。

### 3.10 ContinualSkillBench

**Construction.** 从约 30k tasks 的多源 pool 中筛选 Finance、Law、Healthcare、Office、Math 各 100 个。每领域抽 200 对，由 LLM 双向判断 transfer，建立 dependency graph，再在 difficulty constraint 下排序并人工复核。独立 post-hoc skill annotation 显示 69.5% eligible tasks 复用至少一个历史 core skill，平均 35.5% target-skill requirements 在历史中有 counterpart（§3.1-3.3）。

**What it measures.** 长异构领域流中，context/feedback 与显式 skill repository 的持续积累是否改善 later-task reward，以及 agent 是否真正生成、维护和复用 skills。

**Learning scenario.** 每个 task 三阶段：instruction、execution、reflection。Sequential 保留 context 与 skills；Independent 每 task reset；pure ICL 保留同样 context/feedback 但禁止 create/modify skills。

**Tasks/tests.** Exact Match、F1、numeric tolerance、rubric LLM judge、programmatic artifact verifier；另测 repository size、skill invocation 和 quality。

**Successful evaluation.** Seq 应超过同模型 Ind；若声称“skill learning”，还必须超过 matched pure ICL，而非只从上下文受益，并显示 skill 被后续任务复用。

**Baselines.** 15 个 model-domain pair 中 raw reward 13 个提高、normalized reward 14 个提高，平均 Δraw +0.071、Δnorm +0.078。GPT-5.3-Codex 在 Law/Finance/Healthcare 的平均 normalized reward：Ind 0.466、pure ICL 0.605、skill-maintaining Seq 0.602；显式 skills 没有一致的 aggregate advantage（Table 1-2）。

**Interpretation limit.** 主 Seq-Ind 差同时包含历史 context、反馈和 skills；只有 pure-ICL ablation 能部分隔离 skill contribution，而且论文只在三个领域做了该对照。normalized reward 还只在 Seq 与 Ind 都生成有效输出的交集上计算，必须与 raw reward 和 invalid-output rate 一起解释。

### 3.11 EvoHarnessBench

**Construction.** 从 EnterpriseOps-Gym 与 Agentic Last Exam 确定性派生 17 个 3-6 stage harness streams：802 unique tasks、1,510 axis-specific examples、520 tools、42 hidden reference skills、62 specialist agents。capability 按使用频率分 release bucket；task 放在其所需 capability 首次全部可用的 stage；每 stage 分 adapt/eval（§3.1-3.3）。

**What it measures.** harness 扩展本身会不会造成旧能力下降，以及 persistent inner adaptation 能否使用新 capability 并保留旧 competence。

**Learning scenario.** 两层变化：outer harness 在 tools/skills/agents 三轴单调扩展；deployment evaluation 每 stage fresh state，self-evolving evaluation 则跨 stage 保留 memory、prompt 或 executable code state。

**Tasks/tests.** 每个 held-out cohort 用 source verifier 测 Pass/partial Score；lower-triangular performance matrix 分别读取 diagonal adaptation 和 off-diagonal retention；另测 FWT、BWT、final ACC 与运行成本。

**Successful evaluation.** 新 cohort 的 FWT 为正、旧 cohort 的 BWT 非负、final Pass/Score 高且 cost 可接受；必须同时报告 deployment BWT，以区分 harness-induced forgetting 与 adaptation-induced interference。

**Baselines.** evolving tools 的 cumulative deployment overall pass 28.1%，MemToolAgent 35.1%；evolving skills 为 15.6%，GEPA 18.8%；evolving agents 为 7.4%，Meta-Harness 13.9%。观测到的最大 harness-induced forgetting 分别为 tools -5.3%、skills -4.0%、agents -34.7%（Table 2/4/6/8）。

**Interpretation limit.** skill annotation 有 heuristic matching，三种 axis 的任务子集和 agent architecture 不完全相同；三轴之间宜比较 failure pattern，不宜比较原始 pass rate 难度。

## 4. What counts as a successful evaluation

建议不用单一总分，而报告以下七维 evaluation vector：

`E = <A, L, P, S, G, C, R>`

| 维度 | 定义 | 最低成功证据 |
|---|---|---|
| A: Absolute outcome | 原生 accuracy/pass/reward | 明显高于随机、无技能或不可用 floor |
| L: Learning contribution | 同系统 stateful - stateless/no-memory | 配对增益 > 0，最好有置信区间 |
| P: Plasticity / FWT | 旧经验对新任务或新 capability 的作用 | 在预先声明的可迁移 task 上为正 |
| S: Stability / BWT | 后续更新后旧 competence 是否保留 | BWT/SG 非负，或 forgetting 在容忍阈值内 |
| G: Generalization | 未用于 memory/skill 构建的任务 | held-out gain > 0，排除同题记忆 |
| C: Cost | tokens、tool calls、latency、美元、memory size | 在相同预算下仍有增益 |
| R: Robustness | seeds、顺序、model/harness version、failure rate | 多次运行与顺序扰动下结论一致 |

因此，一个 benchmark 如果只报告 `A` 或 `L`，最多支持“性能提高”或“历史有用”；只有同时覆盖 `P/S/G`，才足以声称兼顾 acquisition、retention 和 generalization 的 continual learning。

## 5. 哪些 benchmark 可以 genuinely compare

### 5.1 现有论文结果可比较到什么程度

| 比较组 | 可以比较 | 不能直接比较 |
|---|---|---|
| TRACE | SeqFT、LoRA、Replay、ICL 在同一 TRACE 协议内 | 与 agent memory/skill pass rate 排名 |
| StreamBench / LifelongAgentBench / MemoryBench | 是否允许 test-time feedback、state 如何保留、是否有 held-out split | 原始 accuracy；feedback 类型、环境交互和序列依赖不同 |
| MemoryBench / Evo-Memory / AgentCL | memory architecture、检索/更新设计的覆盖范围 | 论文数值；backbone、stream、feedback、retest 均不匹配 |
| SkillLearnBench / ContinualSkillBench / EvoHarnessBench-skill | skill 是生成、累积还是外部 release；是否有 usage/quality/outcome | accuracy/pass；skill 供给机制与时间单位不同 |
| SWE-Bench-CL / AgentCL-coding / CL-Bench-Codebase | coding CL 的任务单位、时间结构、verifier 和效率定义 | pass rate；使用的 repo/task/harness 不同，SWE-Bench-CL baseline 还不可靠 |
| AgentCL / CL-Bench | 是否隔离 learning gain，如何诊断 plasticity/stability | PG/SG 与 normalized gain 的数值；参照点与任务结构不同 |
| EvoHarnessBench | tools/skills/agents 三轴的 retention/adaptation failure pattern | 三轴原始 pass 难度；task subset 和执行架构不同 |

### 5.2 最有价值的“重跑后直接比较”子集

1. **Evo-Memory ↔ AgentCL：** 统一到 MMLU-Pro、BabyAI、ScienceWorld；固定同一个 LLM、embedding model、top-k、memory budget、task order；同时运行 Evo-Memory 的 step/success 指标和 AgentCL 的 two-pass PG/SG/GG。这样能检验 ReMem/ExpRAG 的 headline gain 是否同时具有 retention 与 held-out transfer。
2. **SkillLearnBench ↔ ContinualSkillBench：** 选取可映射的 software/productivity/data tasks；统一 fixed solver；设置 `no skill / pure ICL / generated skill / human skill` 四臂；skill 只能从 train/seed instances 产生，再测 held-out task variants 与 later related tasks。
3. **SWE-Bench-CL ↔ CL-Bench Codebase Adaptation：** 统一 SWE 容器和 base commit；同一 issue 同时运行 stateless twin、chronological memory 和旧 issue retest；共同报告 tests passed、paired gain、BWT、tool calls 和 cost。

### 5.3 真正的统一比较协议

要跨 benchmark 比较方法，至少锁定以下变量：

- 相同 base model、system prompt、temperature 和模型版本；
- 相同 agent loop、tool schema、sandbox、context compaction 和 retry policy；
- 相同 task subset、顺序、dependency relation 和 environment snapshot；
- 相同 feedback access，明确 oracle、binary、natural-language 或 verifier feedback；
- 相同 persistent-state 容量：context tokens、memory entries/bytes、skill count；
- 相同 retrieval top-k、embedding model 与 adaptation call budget；
- 预注册 stateless/no-memory、oracle/human upper bound 和 random/order controls；
- 至少报告 `A/L/P/S/G/C/R`，而不是把异构 native scores 平均成无解释的单值；
- 多 seed/rollout、置信区间、失败与 OOM 都计入结果，不把无效输出从主分数悄悄删除。

## 6. 结论

这组 benchmark 的演化路径不是简单的“后来的更全面”：

`parameter retention (TRACE) → online feedback gain (StreamBench) → interactive replay/memory → explicit transfer diagnostics (AgentCL/CL-Bench) → skill consolidation → evolving interface/harness (EvoHarnessBench)`

它们回答的是互补问题。最合理的研究结论不是选一个 benchmark 取代其余全部，而是采用一个最小组合：

- 用 **AgentCL** 测 acquisition、retention、held-out transfer；
- 用 **CL-Bench** 测 real-world latent-structure learning 与 paired stateful gain；
- 用 **EvoHarnessBench** 测 deployment 中外部 capability drift；
- 根据研究载体再加 **TRACE**（参数）、**MemoryBench/Evo-Memory**（memory）或 **SkillLearnBench/ContinualSkillBench**（skills）。

因此，对最终大问题的严格回答是：**所有 benchmark 都能在 anatomy 和证据维度上比较；没有任何 published headline score 能让全部 benchmark 被公平地统一排名。**
