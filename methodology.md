# 3. Methodology

## 3.1 Review design and objective

This study is designed as a **systematic benchmark review and structured qualitative meta-study** of continual-learning evaluation for large language models (LLMs) and language agents. The unit of analysis is the **benchmark as an evaluation instrument**, including its data, task stream, agent--environment protocol, persistent state, controls, verifiers, and metrics. Individual model runs are treated as evidence about how an instrument behaves, rather than as entries in a cross-benchmark leaderboard.

The review does not conduct new model experiments and is not a statistical meta-analysis. A pooled effect size would not be meaningful because the reviewed benchmarks differ in their evaluation units, task distributions, feedback access, update substrates, agent harnesses, cost budgets, and outcome scales. Instead, we use structured data extraction, benchmark-level critical appraisal, and narrative and tabular synthesis. The review process is reported in a PRISMA-inspired form, while adapting the reporting logic to a rapidly evolving computer-science literature that includes preprints, code repositories, and multiple benchmark versions \citep{page2021prisma}.

The study addresses five research questions:

- **RQ1 -- Inventory:** Which benchmarks have been proposed to evaluate continual learning in LLMs and language agents, and in which studies are they used?
- **RQ2 -- Construction:** How are their data, evaluation units, task streams, feedback, persistent state, reset rules, and verifiers constructed?
- **RQ3 -- Measurement:** Which continual-learning properties can their protocols identify, including outcome performance, learning from experience, plasticity, stability, held-out generalization, efficiency, and robustness?
- **RQ4 -- Validity:** What threats to construct validity, causal attribution, ecological validity, metric validity, reproducibility, and robustness limit the conclusions supported by each benchmark?
- **RQ5 -- Comparability:** Under which conditions can reported results be compared directly, only after controlled re-evaluation, or only at the level of benchmark design?

## 3.2 Scope and operational definitions

We define continual learning as learning from a temporally ordered sequence of experiences while preserving or reusing previously acquired competence. A benchmark is therefore in scope only when order, accumulated experience, changing state, or environmental non-stationarity has an evaluative role. Merely evaluating a model on multiple static tasks is insufficient.

For this review, a language agent is a model-centred system that can select actions or tools in response to observations and environmental feedback over one or more interaction steps. Its evaluated system boundary may include the foundation model, policy or prompt, harness, tools, retrieval components, persistent memory or skills, gateway configuration, and environment. Because some foundational continual-learning benchmarks evaluate LLM parameters without an agent interaction loop, we distinguish a **primary agent-benchmark corpus** from **boundary cases**. TRACE is retained as a boundary case: it provides an important reference for parameter-level sequential learning and forgetting, but it is not treated as equivalent to an interactive agent benchmark \citep{wang2023trace}.

## 3.3 Literature search and corpus construction

The review protocol specifies searches of arXiv, Google Scholar, Semantic Scholar, ACL Anthology, and DBLP. Venue-targeted searches cover NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, AAAI, and AAMAS proceedings. The core query combines one term from each of the following groups:

```text
("continual learning" OR "lifelong learning" OR "continuous improvement"
 OR "test-time learning" OR "self-evolving" OR "continual skill")
AND
("large language model" OR LLM OR "language agent" OR "AI agent")
AND
(benchmark OR evaluation)
```

The query is supplemented with targeted searches for `memory`, `skill`, `tool`, `harness`, `stateful environment`, and `coding agent`. Backward snowballing examines references in included papers, and forward snowballing examines papers that cite them \citep{wohlin2014snowballing}. Benchmark websites and public repositories are consulted to identify revised protocols, data releases, evaluation code, and version information. Search strings, sources, execution dates, and the date of the final search are retained in a review log so that the inventory can be updated.

Corpus construction has two linked stages. The **benchmark-discovery search** identifies papers that introduce or substantially revise evaluation instruments. For every included benchmark, a subsequent **benchmark-usage search** uses its exact name, common abbreviations, repository name, and forward citations to locate papers that apply it. From this second set, we record the types of models and agents evaluated, the learning methods compared, protocol modifications, reported metrics, and independent replications or failures. This separation prevents multiple applications of one benchmark from being miscounted as distinct benchmarks while still answering how each instrument is used in the literature.

The initial seed set contains TRACE and ten agent-oriented benchmarks: StreamBench, LifelongAgentBench, SWE-Bench-CL, MemoryBench, Evo-Memory, SkillLearnBench, AgentCL, Continual Learning Bench (CL-Bench), ContinualSkillBench, and EvoHarnessBench \citep{wang2023trace,wu2024streambench,zheng2025lifelongagentbench,joshi2025swebenchcl,ai2025memorybench,wei2025evomemory,zhong2026skilllearnbench,shu2026agentcl,asawa2026clbench,guan2026continualskillbench,ke2026evoharnessbench}. This seed set initiates, but does not replace, the systematic search.

## 3.4 Eligibility criteria and study selection

A study is included when it satisfies all of the following criteria:

1. It introduces a benchmark, or materially revises an existing benchmark protocol.
2. It evaluates LLMs or language agents under sequential experience, temporally ordered tasks, or a changing environment or interface.
3. It permits or studies a persistent update between evaluation units, such as parameter updates, replay, context accumulation, external memory, a skill repository, executable code, or an adapted harness.
4. It defines an evaluation protocol and at least one outcome or continual-learning metric.
5. Sufficient English-language documentation is available to extract the benchmark construction and evaluation procedure.

We exclude ordinary static agent benchmarks in which order has no evaluative role; long-context recall tests without learning or persistent cross-instance adaptation; method papers that make no benchmark contribution; surveys, position papers, and commentaries; and benchmarks for non-language systems. Papers that only apply an already included benchmark remain part of the **usage evidence set**, but they are not counted as separate benchmark instruments. A usage paper is retained when it reports an empirical evaluation on a named included benchmark and provides enough information to identify the evaluated system, protocol, and outcome; papers that only mention or cite a benchmark are excluded from this set.

Screening will proceed in two stages: title-and-abstract screening followed by full-text assessment. Records will be deduplicated by both paper and benchmark identity. When several papers describe the same benchmark, the benchmark will be retained once and each relevant paper will be linked to it as an evidence source. The extraction will record the paper version, release date, publication status, code or data commit where available, and any material protocol changes. To reduce selection and interpretation bias, two reviewers should independently screen the records and pilot the extraction form on a shared subset. Disagreements should be resolved through discussion, with the resulting decision rule documented and applied to the remaining corpus. The final paper must report the search date, numbers at each screening stage, exclusion reasons at full text, reviewer procedure, and observed agreement; these values must not be inferred from the present seed corpus. If independent double screening is not completed, the actual single-reviewer procedure will be reported as a limitation rather than described as independent review.

## 3.5 Data extraction and benchmark anatomy

For each included benchmark, we extract evidence from the paper, appendices, official code, data cards, and evaluation documentation. Every claim is accompanied in the review database by a source location, such as a section, table, appendix, repository file, or version identifier. Authors' claims are recorded separately from our assessment of what the protocol operationally supports.

We represent benchmark anatomy as

\[
B = \langle D, U, S, F, P, R, V \rangle,
\]

where $D$ is the data and task origin, $U$ the evaluation unit, $S$ the stream construction, $F$ the feedback available to the learner, $P$ the persistent state and update rule, $R$ the reset and re-evaluation policy, and $V$ the verifier and metrics. This representation prevents a sequentially ordered dataset from being treated as equivalent to a benchmark that also tests persistent adaptation, retention, and transfer.

| Extraction domain | Information recorded |
|---|---|
| Identity and scope | Benchmark name and version; paper status; release date; stated purpose; model- or agent-level scope |
| Data and tasks ($D,U$) | Reused, transformed, synthetic, or newly authored data; sample and stream counts; evaluation unit; domains; languages; modalities; interactivity; contamination risk |
| Stream ($S$) | Random, chronological, curriculum, dependency-based, compositional, or environment-driven order; number and length of streams; task-boundary visibility; repeated passes; source of non-stationarity |
| Feedback ($F$) | Labels, correctness, rewards, execution results, critique, user actions, self-generated feedback, or no feedback; timing; oracle access; on- or off-policy collection |
| Persistent state ($P$) | Parameters, context, replay, retrieval memory, skills, prompts, code, or routing policy; permitted write/read operations; capacity and retrieval budget; deletion or consolidation; state initialization |
| Evaluation protocol ($R$) | Online/offline and single-/multi-pass setting; environment resets; old-task retesting; held-out tasks; state freezing; stateful/stateless pairing; model and harness controls |
| Verifier and metrics ($V$) | Exact match, execution success, partial reward, human or LLM judge, BWT/FWT, forgetting, learning-curve measures, aggregation and normalization; reference point and direction of improvement |
| Baselines and evidence | No-learning, independent-task, replay/context, memory/skill, oracle or human, and ablation baselines; backbone and harness; seeds; uncertainty; significance tests; reported failures |
| Resources and openness | Tokens, latency, tool calls, memory, compute, and monetary cost; availability of data, code, prompts, environments, containers, logs, and versioned dependencies |

The continual-learning setting is coded independently along six dimensions: (i) online versus offline updating, (ii) instance- versus task-level streams, (iii) known versus latent task boundaries, (iv) single- versus multi-pass exposure, (v) update substrate, and (vi) the source of non-stationarity, such as data, goals, environments, feedback, tools, or harness capabilities. We do not force agent benchmarks into classical task-, domain-, or class-incremental labels when those labels do not describe the changing object.

## 3.6 Measurement framework

To distinguish a benchmark's headline score from the constructs it can identify, we map its evidence to the following measurement vector:

\[
M(B) = \langle A, L, P, S, G, C, R_b \rangle.
\]

- $A$, **absolute outcome**, is task accuracy, reward, pass rate, progress, or another end-task measure.
- $L$, **attributable learning**, is improvement causally attributable to information retained from earlier experience, ideally through a matched stateful-versus-stateless or learning-versus-no-learning control.
- $P$, **plasticity**, is acquisition of a current or new task, including forward transfer relative to independent learning or a no-history reference.
- $S$, **stability**, is retention of prior competence, measured by old-task re-evaluation, forgetting, or backward transfer.
- $G$, **generalization**, is transfer to held-out tasks, instances, domains, compositions, or environments not used to create the persistent state.
- $C$, **cost**, includes tokens, time, memory, tool calls, compute, and monetary expenditure.
- $R_b$, **robustness**, covers sensitivity to stream order, seeds, erroneous feedback, model choice, prompt or harness variation, and benchmark version.

For every reported metric, we extract its formula or operational definition, unit, range, direction, aggregation level, evaluation time, and reference point. Metrics with the same name are not assumed to be equivalent: for example, backward transfer relative to performance immediately after learning a task differs from retention relative to a stateless baseline. Composite and normalized scores are interpreted only together with their component measures and normalization population.

Two notions of success are kept separate. A **method succeeds on a benchmark** when it improves the benchmark's pre-specified target outcomes under matched controls without unacceptable losses in retained ability, generalization, safety, or cost, to the extent those dimensions are measured. A **benchmark succeeds as an evaluation instrument** when its construction validly operationalizes the claimed construct, isolates learning from confounds, produces reliable and discriminating measurements, and provides enough documentation for independent use. Because acceptable trade-offs are application-dependent, we do not impose a universal numerical threshold; we assess whether the paper defines its success criterion and whether the reported evidence satisfies it. High model scores alone establish neither form of success.

## 3.7 Critical appraisal

Each benchmark is appraised dimension by dimension rather than collapsed into a single quality score. A total score would impose arbitrary weights and could hide a benchmark that is strong for one research question but unsuitable for another. The appraisal covers:

- **Construct validity:** whether the temporal protocol and metrics represent the claimed form of continual learning.
- **Internal validity and attribution:** whether controls isolate persistent learning from base-model ability, additional context, oracle feedback, prompt changes, or unequal budgets.
- **External and ecological validity:** whether tasks, ordering, feedback, and non-stationarity resemble the intended deployment setting.
- **Metric and verifier validity:** whether the verifier is accurate and whether aggregation, normalization, ceiling effects, or judge bias distort conclusions.
- **Reliability and robustness:** whether conclusions persist across stream orders, seeds, backbones, feedback quality, and harness versions.
- **Reproducibility:** whether data, code, prompts, environments, versions, seeds, and resource requirements are available.
- **Bias and leakage risk:** including benchmark contamination, synthetic-data artefacts, language and domain concentration, dependency leakage, and feedback generated by related models.
- **Resource, safety, and fairness coverage:** whether cost is transparently reported and whether relevant safety or subgroup failures are assessed.

For each dimension, the evidence is coded as **clearly demonstrated**, **partially demonstrated**, **not demonstrated**, or **not reported**, followed by a textual justification and source. “Not demonstrated” means that available evidence does not establish the property; it does not assert that the property is absent. Safety and fairness are marked not applicable only when a reason is recorded.

## 3.8 Analysis of baselines and reported results

Baselines are classified by their inferential role: no-learning or stateless controls estimate base capability; independent-task controls estimate learning without transfer; full-context or replay controls test whether dedicated memory management adds value; established memory or skill systems compare mechanisms; oracle or human-authored artefacts estimate an upper bound; and component ablations test causal mechanisms. We assess whether each benchmark contains the controls required for its claimed construct.

Representative numerical results are extracted only to show the benchmark's scale, discriminative behaviour, failure modes, and sensitivity to controls. Wherever possible, we report a matched within-study contrast together with the model, benchmark version, task stream, state and feedback budget, harness, metric, and uncertainty. Missing runs, execution failures, out-of-memory cases, and inconclusive findings are retained rather than treated as zero or omitted. We do not pool effect sizes or rank methods across heterogeneous benchmarks.

## 3.9 Comparability assessment

Comparability is assessed using the configuration tuple

\[
Q = \langle b, t, o, m, h, f, s, v, c, r \rangle,
\]

where $b$ is benchmark version, $t$ the task set, $o$ the stream and order, $m$ the foundation model, $h$ the agent policy and harness, $f$ feedback access, $s$ the persistent-state mechanism and capacity, $v$ the verifier and metric definition, $c$ the resource budget, and $r$ randomness and repetition. We assign one of three comparison levels:

| Level | Decision rule | Permissible conclusion |
|---|---|---|
| **Directly comparable** | Critical elements of $Q$ match, or differences are explicitly controlled within the same experimental design | Numerical method comparison on the specified benchmark configuration |
| **Conditionally comparable** | The studies share tasks or target constructs, but one or more critical elements of $Q$ differ | Comparison requires controlled re-runs or re-analysis under a harmonized protocol |
| **Design-comparable only** | Evaluation units, streams, constructs, or scales differ materially | Qualitative comparison of coverage, validity, and design; no numerical ranking |

This rule is deliberately stricter than matching benchmark names or metric labels. Results obtained with different model versions, task orders, feedback, state budgets, harnesses, or verifier definitions are not treated as direct replications. Cross-benchmark scores are not used to infer which benchmark is harder or which continual-learning method is superior.

## 3.10 Evidence synthesis and reporting

The results are synthesized in four linked outputs. First, an inventory table describes benchmark identity, tasks, evaluated systems, metrics, availability, and usage. Second, an anatomy matrix compares the seven components of $B$. Third, an evidence map shows which elements of $M(B)$ each benchmark measures **directly** (through a dedicated test or matched control), **indirectly** (through a proxy with unresolved alternative explanations), or **not at all**. Fourth, the critical-appraisal and comparability matrices identify validity threats and legitimate comparison groups.

Benchmarks are grouped by their primary evaluative question: online interaction-stream adaptation, transfer--retention diagnosis, persistent competence consolidation, or adaptation to an evolving capability interface. These categories organize the synthesis but do not replace the multi-dimensional coding; a memory benchmark may also measure stability, and an agent benchmark may update parameters. The discussion then examines convergent findings, disagreements, biases, and gaps without converting heterogeneous evidence into a single benchmark or model ranking.

## 3.11 Review transparency and limitations

The extraction form, search log, inclusion decisions, benchmark-version table, coded evidence, and source locations should be released with the paper. Amendments to the protocol should be dated and justified. Where possible, included authors' repositories are archived by commit identifier because agent harnesses and external services can change after publication.

This design has three anticipated limitations. First, fast-moving preprints may change after extraction. Second, inconsistent terminology may cause relevant benchmarks to be missed despite snowballing. Third, most evidence comes from benchmark authors' own baseline experiments, so reproducibility and external usage may remain weak. These limitations are reported as properties of the available literature, not repaired by treating incomparable model scores as a common outcome.

---

### Protocol completion note (remove before submission)

Before converting this protocol into a final, past-tense Methodology section, record: (1) the final search date and exact queries by source; (2) the numbers of retrieved, deduplicated, screened, excluded, and included records; (3) reviewer assignments and observed agreement; and (4) the benchmark paper and repository versions actually analysed. No such values should be reconstructed retrospectively from the current list of eleven seed benchmarks.
