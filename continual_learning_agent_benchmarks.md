# Compare Continual Learning Benchmarks for AI Agents

> 课程作业草稿。文献覆盖截至 2026 年 9 月 13 日。Memory、Stability、Lifelong learning 与 Acquisition 仅作为被重新审视的初始设想；Part 2 不直接沿用该划分，而依据 CL 的任务流结构、更新载体与评估目标重新推导分类体系。

# Part 1: Definition

## 1. Continual learning 的定义与研究维度

Continual learning（CL，持续学习）研究的是：模型或系统面对按时间到达、分布可能变化的数据或任务流时，能否从新经验中学习，同时保留并复用过去获得的能力。它并不等同于把若干静态数据集依次训练一遍。一个严格的 CL 问题至少要交代经验以什么顺序到达、系统何时可以更新、更新后是否重测旧任务，以及系统是否知道任务边界。其核心矛盾通常被概括为 stability-plasticity dilemma：系统既要有足够的可塑性（plasticity）吸收新知识，又要有足够的稳定性（stability）避免灾难性遗忘或负迁移 \citep{parisi2019continual,delange2022survey}。

经典 CL 文献通常从以下几组彼此正交的维度界定研究问题，而不是只给出一种分类 \citep{parisi2019continual,vandeven2019three,delange2022survey,lopezpaz2017gem}。

1. **任务流与分布变化。** Task-incremental learning 假定测试时可知任务身份；domain-incremental learning 中输入分布变化，但输出空间通常不变；class-incremental learning 则要求在不断扩展的输出空间中统一预测 \citep{vandeven2019three}。对 Agent 还要补充环境状态、工具集合、用户偏好和反馈规则是否随时间变化 \citep{ke2026evoharnessbench}。
2. **时间与数据协议。** 包括 online 或 offline、单遍或多遍、任务边界已知或未知、任务顺序随机或按真实时间排列，以及能否回放历史样本 \citep{delange2022survey,wu2024streambench,joshi2025swebenchcl}。它决定实验究竟模拟一次性适配，还是长期部署中的持续更新。
3. **更新载体。** 传统 CL 主要更新模型参数；LLM Agent 还可以更新上下文、外部记忆、检索索引、提示词、技能库、代码、工具选择策略或多 Agent 的委派策略 \citep{ai2025memorybench,wei2025evomemory,guan2026continualskillbench,ke2026evoharnessbench}。不同载体的容量、持久性和删除机制并不相同，不能只以最终准确率将其视为同一种学习。
4. **学习目标。** 除当前任务表现外，还包括平均性能（average accuracy）、遗忘或 backward transfer、对未来任务的 forward transfer、样本效率、泛化和资源开销 \citep{lopezpaz2017gem,shu2026agentcl,asawa2026clbench}。仅证明后续任务得分提高，不能同时证明旧能力得到保持 \citep{shu2026agentcl}。
5. **非平稳性的来源。** 变化既可能来自输入和任务，也可能来自外部世界、反馈分布、模型版本或 Agent 的 harness；不同来源对应不同的因果解释 \citep{ke2026evoharnessbench}。

上述经典维度表明，Memory 更接近信息保留与调用的能力及实现载体，Stability 与 Acquisition 分别对应 stability-plasticity 的两端，而 Lifelong learning 更接近时间跨度、任务依赖和部署协议 \citep{parisi2019continual,delange2022survey,ai2025memorybench}。因此，本文不直接把四者作为互斥类别，而在 Part 2 中重新推导分类体系。

## 2. 当前 AI Agent 的定义、系统边界与前沿应用

在传统人工智能定义中，Agent 通过观察环境并采取行动实现目标。面向当前 LLM 系统，更有区分力的定义是：**AI Agent 是以模型为决策核心、能够根据环境反馈动态决定后续步骤和工具调用，并在多轮行动中推进目标的系统。** \citet{anthropic2024agents} 将预定义代码路径编排 LLM 与工具的系统称为 workflow，而把由 LLM 动态控制过程和工具使用的系统称为 agent。因此，单次生成、固定 prompt chain 或普通 RAG 本身不必然构成 Agent；关键在于是否存在“观察—决策—行动—新观察”的闭环，以及模型是否对路径具有实际控制权 \citep{anthropic2024agents}。

Agent 不应被缩减为一个基础模型；现有 Agent 与 continual-learning benchmark 都把工具、记忆、技能或 harness 纳入系统实现与评测边界 \citep{anthropic2024agents,ai2025memorybench,guan2026continualskillbench,ke2026evoharnessbench}。本文据此采用如下系统边界：

| 系统层 | 主要内容 | 对评测的影响 |
|---|---|---|
| 模型与 Agent policy | 基础模型、system prompt、规划与工具选择策略 | 决定推理、生成和行动选择的基础能力 \citep{anthropic2024agents} |
| Harness / runtime | 工具定义与解析、执行循环、上下文组装与压缩、记忆/技能加载、子 Agent 编排、沙箱、权限、重试和停止条件 | 决定 Agent 能观察什么、能采取什么行动，以及失败能否恢复；EvoHarnessBench 将其概括为由 tools、skills 和 specialist agents 构成并持续演化的外部能力层 \citep{ke2026evoharnessbench} |
| 持久状态 | 对话历史、经验库、检索索引、技能库、参数或可执行代码 | 是 continual learning 实际发生的位置，必须说明任务之间哪些状态被保留、更新或清除 \citep{ai2025memorybench,wei2025evomemory,guan2026continualskillbench} |
| Gateway / control plane | 位于应用与模型提供方之间的认证、模型路由、限流、缓存、重试、fallback、日志和成本控制 | 它通常不是学习算法本身，却可能改变实际调用的模型、响应、延迟和失败分布；Cloudflare AI Gateway 的公开定义明确包括 analytics/logging、caching、rate limiting、retry 与 model fallback \citep{cloudflare2026gateway} |
| 环境与 evaluator | 浏览器、代码库、操作系统、数据库、API、模拟世界、用户反馈及 verifier | 给出可观察状态、可行动作、反馈和最终成功标准 \citep{zheng2025lifelongagentbench,shu2026agentcl} |

据此，一个可复现的 Agent benchmark 实际评测的是一个版本化系统配置，而不只是“某个 LLM” \citep{joshi2025swebenchcl,ke2026evoharnessbench}。可将评测对象简写为：

`Agent system = model + policy/prompt + harness + persistent state + gateway + environment/evaluator` \citep{anthropic2024agents,ai2025memorybench,ke2026evoharnessbench}。

当前前沿应用主要集中在软件工程、deep research 与 Web 搜索、computer-use 与企业工作流、专业领域决策以及多 Agent 协作；相关 benchmark 已覆盖 coding、deep research、OS/DB、企业工具、医疗与法律、具身环境和多 Agent harness 等场景 \citep{wu2024streambench,zheng2025lifelongagentbench,zhong2026skilllearnbench,shu2026agentcl,guan2026continualskillbench,ke2026evoharnessbench}。

## 3. 为什么需要面向 AI Agent 的 continual learning benchmark

普通 LLM 的 CL benchmark 通常把模型作为主要学习对象，在若干静态数据集上顺序微调，然后重测旧任务与通用能力。\citet{wang2023trace} 的 TRACE 由八个领域、语言、代码和数学数据集组成；Llama-2-Chat-13B 在顺序训练后，GSM8K 准确率由 43.14% 降至 2.12%，展示了参数更新导致的灾难性遗忘。这种设置能够回答“模型参数在顺序训练中是否遗忘”，却不足以回答“一个部署中的 Agent 系统能否从交互经验中持续改善” \citep{wu2024streambench,zheng2025lifelongagentbench}。

| 比较方面 | 普通 LLM CL benchmark | AI Agent CL benchmark 的额外要求 |
|---|---|---|
| 基本样本 | 静态 input-output 数据 \citep{wang2023trace} | 多步 trajectory：观察、工具调用、环境反馈与恢复过程 \citep{zheng2025lifelongagentbench,shu2026agentcl} |
| 学习载体 | 主要是模型参数 \citep{wang2023trace} | 参数、上下文、记忆、技能、代码、检索器和委派策略均可能更新 \citep{ai2025memorybench,guan2026continualskillbench,ke2026evoharnessbench} |
| 反馈 | 标签或训练损失 \citep{wang2023trace} | 延迟、稀疏或部分可观察的环境、用户及 verifier 反馈 \citep{wu2024streambench,ai2025memorybench,zheng2025lifelongagentbench} |
| 状态 | 数据集之间按协议切换 \citep{wang2023trace} | 环境状态与 Agent 状态同时演化，必须规定 reset 和 persistence 边界 \citep{zheng2025lifelongagentbench,wei2025evomemory} |
| 非平稳性 | 多为任务或数据分布变化 \citep{vandeven2019three,delange2022survey} | 任务、工具/API、技能、harness、模型路由和环境都可能变化 \citep{ke2026evoharnessbench} |
| 核心指标 | accuracy、forgetting、BWT/FWT \citep{lopezpaz2017gem,wang2023trace} | 还需任务成功率、步骤/工具效率、恢复率、记忆成本、延迟、token 与货币成本、安全违规等 \citep{zhong2026skilllearnbench,asawa2026clbench,ke2026evoharnessbench} |
| 可复现性 | 模型、数据、优化器与随机种子 \citep{delange2022survey} | 还需锁定 harness、工具版本、容器、权限、gateway 路由/缓存/fallback 和外部服务状态 \citep{joshi2025swebenchcl,cloudflare2026gateway,ke2026evoharnessbench} |
| 风险 | 主要是遗忘与分布偏移 \citep{parisi2019continual,delange2022survey} | 错误经验写入、记忆投毒、隐私泄露、权限滥用和多步错误累积 \citep{ai2025memorybench,zhong2026skilllearnbench} |

专门 benchmark 的必要性由此体现在三点。第一，Agent 的学习结果存在于模型、上下文、记忆和技能等整个系统中；只测模型权重会遗漏非参数适应 \citep{ai2025memorybench,wei2025evomemory,guan2026continualskillbench}。第二，Agent 的行动会改变后续观察，静态问答准确率无法刻画探索、工具使用和错误恢复 \citep{zheng2025lifelongagentbench,shu2026agentcl}。第三，harness 与 gateway 会造成系统层混杂：同一模型可能因工具描述、上下文裁剪、重试或模型 fallback 而得到不同结果 \citep{cloudflare2026gateway,ke2026evoharnessbench}。\citet{joshi2025swebenchcl} 使用原 SWE-Bench harness 时发现总体通过率通常低于 8.5%，且频繁出现执行不兼容，因而明确认为结果不足以支持可靠结论；\citet{ke2026evoharnessbench} 进一步显示，仅扩展 harness 就会使已会任务发生遗忘。这说明 Agent CL benchmark 必须同时控制学习算法和系统基础设施。

# Part 2: Related Work

## 1. Benchmark Category

| Category | Primary evaluation question | Primary temporal object | Required evidence of learning | Typical update substrate | Benchmarks |
|---|---|---|---|---|---|
| **Online Interaction-Stream Adaptation** | Does feedback from earlier interactions improve later decisions in the same stream? | Sequential task instances, outcomes, and environment feedback | Online or cumulative gain over a stateless/no-history baseline | Prompt, demonstrations, replay buffer, or short-term feedback history | StreamBench \citep{wu2024streambench}; LifelongAgentBench \citep{zheng2025lifelongagentbench} |
| **Transfer-Retention Diagnostics** | Does acquired experience transfer to related tasks without degrading prior or unseen capabilities? | Chronological, latent-structure, or explicitly compositional task sequences | Separate measurements of plasticity, retention/forgetting, and forward or held-out transfer | Substrate-agnostic: context, memory, skills, or parameters | SWE-Bench-CL \citep{joshi2025swebenchcl}; AgentCL \citep{shu2026agentcl}; CL-Bench \citep{asawa2026clbench} |
| **Persistent Competence Consolidation** | Can trajectories be compressed into reusable memories or procedural skills that improve subsequent task execution? | Growth and revision of an agent's persistent competence store | Downstream reuse plus direct memory/skill-quality or mechanism ablations | Evolving memory entries, experience repositories, or skill libraries | MemoryBench \citep{ai2025memorybench}; Evo-Memory \citep{wei2025evomemory}; SkillLearnBench \citep{zhong2026skilllearnbench}; ContinualSkillBench \citep{guan2026continualskillbench} |
| **Evolving Capability-Interface Adaptation** | Can an agent preserve and extend competence as its available action interface changes? | Staged additions of tools, skills, or specialist agents in the harness | Performance across harness stages, harness-induced forgetting, and adaptation gain | Memory-, prompt-, or code-level adaptation of the agent/harness | EvoHarnessBench \citep{ke2026evoharnessbench} |

The proposed taxonomy integrates three dimensions that continual-learning research treats separately: stream structure, update substrate, and evaluation objective \citep{parisi2019continual,vandeven2019three,delange2022survey}. Its categories are defined by the primary temporal object under study and the evidence used to identify learning. This avoids placing memory, stability, lifelong learning, and acquisition at the same conceptual level: memory is mainly an update substrate \citep{ai2025memorybench,wei2025evomemory}, stability and acquisition are complementary objectives in the stability-plasticity trade-off \citep{parisi2019continual,delange2022survey}, and lifelong learning characterizes a temporal protocol \citep{zheng2025lifelongagentbench}. Each benchmark is therefore assigned by its primary research question, while the other dimensions remain secondary descriptors. Online Interaction-Stream Adaptation benchmarks ask whether accumulated interaction history improves later behavior: StreamBench serializes static datasets into feedback streams, whereas LifelongAgentBench adds strict ordering and skill-dependent tasks \citep{wu2024streambench,zheng2025lifelongagentbench}. Transfer-Retention Diagnostics require stronger causal evidence: SWE-Bench-CL uses repository chronology, AgentCL constructs controlled compositional relations, and CL-Bench embeds shared latent structure, enabling explicit analysis of plasticity, forgetting, or transfer rather than only final success \citep{joshi2025swebenchcl,shu2026agentcl,asawa2026clbench}. Persistent Competence Consolidation benchmarks instead ask whether experience becomes a reusable artifact: MemoryBench studies procedural memory from service-time feedback, Evo-Memory continually rewrites experience memory, and the two skill benchmarks evaluate generated procedures through later trajectories and outcomes \citep{ai2025memorybench,wei2025evomemory,zhong2026skilllearnbench,guan2026continualskillbench}. Evolving Capability-Interface Adaptation places temporal change outside the task distribution by expanding the tools, skills, and specialist agents exposed through the harness \citep{ke2026evoharnessbench}. The categories are mutually exclusive only at the level of primary benchmark intent; AgentCL, for example, evaluates memory systems, but its defining contribution is transfer diagnosis \citep{shu2026agentcl}. The resulting coverage indicates that current benchmarks usually isolate online improvement, retention, consolidation, or interface drift rather than jointly identifying them under matched state, cost, and infrastructure controls \citep{shu2026agentcl,asawa2026clbench,ke2026evoharnessbench}.

## 2. Benchmark Comparison

| Benchmark | Dataset and stream design | Protocol and persistent state | Agent-specific evaluation | Representative reported result |
|---|---|---|---|---|
| StreamBench \citep{wu2024streambench} | Seven serialized datasets spanning SQL, code, tools, medicine, and QA | Online binary feedback; prompts or example memories evolve after each item | Continuous improvement under a feedback stream; task-native accuracy/pass metrics | MAM-StreamICL scored 83.50 on DDXPlus versus 52.85 zero-shot; gains were positive across all seven datasets \citep{wu2024streambench} |
| LifelongAgentBench \citep{zheng2025lifelongagentbench} | 1,396 skill-dependent tasks in DB, OS, and KG environments | Fixed serial execution; 0-64 retrieved experiences persist across tasks | Executable actions, automatic verification, task dependency, and context-budget effects | Llama-3.1-8B rose from 0.19 to 0.78 on DB, but OS peaked at 0.50 and KG produced OOM from 16 experiences \citep{zheng2025lifelongagentbench} |
| SWE-Bench-CL \citep{joshi2025swebenchcl} | 273 issues from eight Python repositories ordered chronologically | Independent baseline versus FAISS-backed semantic memory | Code editing, tool efficiency, forgetting, and forward/backward transfer | Standard-harness pass rates were generally below 8.5%; frequent execution failures made the comparison inconclusive \citep{joshi2025swebenchcl} |
| MemoryBench \citep{ai2025memorybench} | About 20k multilingual cases across open, academic, and legal tasks | Simulated verbal/action user feedback; on- and off-policy memory updates | Procedural feedback use, heterogeneous input/output formats, memory and inference time | Embed-S scored 0.6582/0.7185/0.4806 across the three domains; advanced memory systems did not consistently beat simple RAG \citep{ai2025memorybench} |
| Evo-Memory \citep{wei2025evomemory} | Five single-turn benchmark families (with MMLU-Pro split into three domains) and interactive ALFWorld, BabyAI, PDDL, and ScienceWorld streams | Memory is retrieved, integrated, and updated after each interaction | Goal success, partial progress, action-conditioned experience reuse | On four environments with Claude 3.7, baseline mean S/P was 0.24/0.52 and ReMem reached 0.78/0.91 \citep{wei2025evomemory} |
| SkillLearnBench \citep{zhong2026skilllearnbench} | 100 verified instances from 20 tasks and 15 real-world subdomains | One-shot, self-feedback, teacher-feedback, or Skill Creator updates explicit skills | Skill coverage, executability, safety, trajectory alignment/usage, outcome, and tokens | Average accuracy was 10.17% without skills, 31.08% with self-feedback, and 74.50% with human-authored skills \citep{zhong2026skilllearnbench} |
| AgentCL \citep{shu2026agentcl} | Coding, deep research, BabyAI, ScienceWorld, and MMLU-Pro; naive versus controlled compositional streams | Non-parametric memories persist across two passes; separate held-out evaluation | Plasticity Gain (PG), Stability Gain (SG), and Generalization Gain (GG) | On CodeEval-Pro, MemProbe achieved PG +21.9 points but SG -2.1 and held-out GG -0.8 \citep{shu2026agentcl} |
| CL-Bench \citep{asawa2026clbench} | Six expert-validated domains with reusable latent structure and some concept drift | Stateful systems compared with stateless counterparts under matched tasks | Reward gain, stability/plasticity decomposition, and dollar cost | Full-context ICL with Sonnet 4.6 obtained 22.3% normalized reward and 25.4% normalized gain; dedicated memory systems lagged \citep{asawa2026clbench} |
| ContinualSkillBench \citep{guan2026continualskillbench} | Five domains with 100 difficulty- and dependency-ordered tasks each | Sequential versus independent execution; reflection updates a skill repository | Verifier reward, skill quality/count/reuse, and pure-ICL ablation | Sequential execution improved 14 of 15 model-domain pairs by +0.078 mean normalized reward; pure ICL (0.605) matched skills (0.602) \citep{guan2026continualskillbench} |
| EvoHarnessBench \citep{ke2026evoharnessbench} | 17 staged streams containing 802 tasks, 520 tools, 42 skills, and 62 agents | Deployment-only and self-evolving adaptation as the harness expands | Tool selection, skill invocation, delegation, BWT/FWT, time, and tokens | Maximum harness-induced forgetting was -5.3%, -4.0%, and -34.7% for tools, skills, and agents, respectively \citep{ke2026evoharnessbench} |

The benchmarks differ first in how they justify transfer. StreamBench randomizes static datasets, which provides broad coverage but does not guarantee that earlier experience is useful later \citep{wu2024streambench,shu2026agentcl}. LifelongAgentBench imposes skill dependencies, SWE-Bench-CL preserves repository chronology, AgentCL engineers reusable subsolutions, and CL-Bench hides expert-validated latent structure; these designs trade ecological validity against causal control \citep{zheng2025lifelongagentbench,joshi2025swebenchcl,shu2026agentcl,asawa2026clbench}. Persistent state likewise ranges from raw context and replay to structured memory, explicit skill repositories, and harness-level code or prompt adaptation \citep{ai2025memorybench,wei2025evomemory,zhong2026skilllearnbench,guan2026continualskillbench,ke2026evoharnessbench}. Consequently, similar task scores can arise from retrieval, in-context adaptation, procedural consolidation, or retention, a confound exposed by the ablations in ContinualSkillBench and CL-Bench \citep{guan2026continualskillbench,asawa2026clbench}. Agent-specific evaluation also remains uneven: interactive benchmarks measure executable actions, progress, tool use, trajectory alignment, latency, or cost, whereas SWE-Bench-CL, AgentCL, CL-Bench, and EvoHarnessBench explicitly target forgetting or backward transfer \citep{joshi2025swebenchcl,shu2026agentcl,asawa2026clbench,ke2026evoharnessbench}. Experience can produce large gains: Llama-3.1-8B improves from 0.19 to 0.78 on LifelongAgentBench's DB environment, and ReMem raises Claude 3.7's mean interactive success from 0.24 to 0.78 in Evo-Memory \citep{zheng2025lifelongagentbench,wei2025evomemory}. Yet more memory is not monotonically beneficial: larger replay causes degradation or OOM in LifelongAgentBench, advanced memory systems do not consistently beat simple RAG in MemoryBench, and full-context ICL outperforms dedicated memory management on average in CL-Bench \citep{zheng2025lifelongagentbench,ai2025memorybench,asawa2026clbench}. Skill learning is similarly ambiguous: SkillLearnBench remains far below human-authored skills, while ContinualSkillBench reports nearly identical mean scores for explicit skill maintenance and pure ICL \citep{zhong2026skilllearnbench,guan2026continualskillbench}. AgentCL further shows that high plasticity can coincide with negative stability and held-out transfer \citep{shu2026agentcl}. Finally, SWE-Bench-CL's low pass rates are confounded by harness incompatibility, while EvoHarnessBench directly identifies harness expansion as a source of forgetting \citep{joshi2025swebenchcl,ke2026evoharnessbench}. Absolute scores are therefore not comparable across rows; benchmark design studies instead motivate matched models, state budgets, feedback access, infrastructure versions, repeated runs, and separate acquisition, retention, generalization, safety, and cost measures \citep{shu2026agentcl,asawa2026clbench,ke2026evoharnessbench}.

# References

[1] Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. [Continual Lifelong Learning with Neural Networks: A Review](https://doi.org/10.1016/j.neunet.2019.01.012). *Neural Networks*, 2019.

[2] van de Ven, G. M., & Tolias, A. S. [Three Scenarios for Continual Learning](https://arxiv.org/abs/1904.07734). arXiv:1904.07734, 2019.

[3] De Lange, M., Aljundi, R., Masana, M., et al. [A Continual Learning Survey: Defying Forgetting in Classification Tasks](https://doi.org/10.1109/TPAMI.2021.3057446). *IEEE TPAMI*, 2022.

[4] Lopez-Paz, D., & Ranzato, M. [Gradient Episodic Memory for Continual Learning](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html). *NeurIPS*, 2017.

[5] Wang, X., Zhang, Y., Chen, T., et al. [TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](https://arxiv.org/abs/2310.06762). arXiv:2310.06762, 2023.

[6] Anthropic. [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents). 2024.

[7] Cloudflare. [Cloudflare AI Gateway](https://developers.cloudflare.com/ai-gateway/). Documentation, accessed 2026-09-13.

[8] Wu, C.-K., Tam, Z. R., Lin, C.-Y., Chen, Y.-N., & Lee, H.-Y. [StreamBench: Towards Benchmarking Continuous Improvement of Language Agents](https://arxiv.org/abs/2406.08747). *NeurIPS Datasets and Benchmarks*, 2024.

[9] Zheng, J., Cai, X., Li, Q., et al. [LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners](https://arxiv.org/abs/2505.11942). arXiv:2505.11942, 2025.

[10] Joshi, T., Chowdhury, S., & Uysal, F. [SWE-Bench-CL: Continual Learning for Coding Agents](https://arxiv.org/abs/2507.00014). arXiv:2507.00014, 2025.

[11] Ai, Q., Tang, Y., Wang, C., et al. [MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems](https://arxiv.org/abs/2510.17281). arXiv:2510.17281, 2025/2026.

[12] Wei, T., Sachdeva, N., Coleman, B., et al. [Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory](https://arxiv.org/abs/2511.20857). arXiv:2511.20857, 2025/2026.

[13] Zhong, S., Lu, Y., Ning, J., et al. [SkillLearnBench: Benchmarking Continual Learning Methods for Agent Skill Generation on Real-World Tasks](https://arxiv.org/abs/2604.20087). arXiv:2604.20087, 2026.

[14] Shu, Y., Jimenez Gutierrez, B., Jonnalagedda, S. P., et al. [AgentCL: Toward Rigorous Evaluation of Continual Learning in Language Agents](https://arxiv.org/abs/2606.02461). arXiv:2606.02461, 2026.

[15] Asawa, P., Glaze, C. M., Orlanski, G., et al. [Continual Learning Bench: Evaluating Frontier AI Systems in Real-World Stateful Environments](https://arxiv.org/abs/2606.05661). arXiv:2606.05661, 2026.

[16] Guan, T., Wang, Y., Yang, H., et al. [ContinualSkillBench: Can LLM Agents Truly Evolve Their Capabilities?](https://arxiv.org/abs/2608.03874). arXiv:2608.03874, 2026.

[17] Ke, Z., Patil, V., Shi, H., et al. [EVOHARNESSBENCH: Can Your Agents Keep Pace with an Evolving Harness?](https://arxiv.org/abs/2609.04280). arXiv:2609.04280, 2026.
