# 近半年 LLM 工程方向与立项建议

**观察窗口**　北京时间 2026-05-23 至 2026-11-23，截至本次查阅。按首次提交、修订、主干合并、正式发行和官方技术报告分别标注。基于一手资料做技术方向筛选，W47 的工程明细见本地周报。本文给出的工期和验收门槛是试验设定，性能数字均注明上游来源，独立 GPU/NPU 验证为 `pending_remote`。 [W47 增强版](../weekly/2026-W47-expanded.md)

## 建议先做什么

**先用一套真实会话回放做底座，在现有引擎上验证前缀复用、排队和投机三者的收益。** 第一轮交付应包括可重复的版本对照、实际可复用的 prefill token 数、会话 P95/P99 和每卡完成任务数。以此结果决定做缓存调度、短请求快路径，还是进一步投入硬件专用算子。

| 顺位 | 工程方向 | 选择条件 | 首轮可交付 |
| --- | --- | --- | --- |
| 1 · 立即 | 真实会话回放、差分正确性与 SLO goodput | 所有在线引擎升级、并行、量化项目均受益 | 固定轨迹、动态任务集、版本清单、阶段时延和首处分歧定位 |
| 2 · 立即 | 缓存感知选路、准入与 host KV | 多轮、代码 Agent 或 RAG 有明显前缀重叠，重算或排队是主要成本 | 负载均衡 → 前缀感知 → 负载和前缀感知 → host KV 的四臂对照 |
| 3 · 立即验证 | 按负载决定投机验证预算 | decode 占主要耗时，模型有可用 draft，流量横跨低并发与饱和 | AR、固定、动态三条曲线，输出回退区间和接受长度分布 |
| 4 · 条件优先 | 小标签集合评分快路径、混合 batch 隔离 | 单 token 固定标签评分占相当流量，warm path 下 head 或通信显著 | 合同测试与阶段 profile，满足条件时做候选投影原型 |
| 5 · 条件优先 | Ascend P/D、PCP/DCP 与狭义内核优化 | 有固定 A3/A5 资源、一个明确目标模型和长上下文工作负载 | 版本兼容表、逐项开关对照、驻留内存与通信账本 |
| 6 · 次优先 | 稀疏索引缓存、低精度 KV 与分离式 I/O | 已定位 KV 或索引层为容量瓶颈，且主存／互联有余量 | 分离索引和主 KV 的内存及读写测量，根据瓶颈选择一条原型 |
| 7 · 次优先 | 推理预算路由与失败升级 | 用量有清晰难度分布，工具任务有客观成功条件 | 固定预算和两档升级策略的成本–成功率 Pareto |
| 8 · 按场景 | 后训练 rollout 权重更新与 sandbox 调度 | 训练流程长期被生成尾部、权重同步或环境创建拖慢 | 同步与有界异步的 time-to-quality、版本和 logprob 追溯 |
| 9 · 观察 | MoE 融合、扩散语言模型、流式多模态 | 有明确任务簇与硬件，能控制基线和验证条件 | 一份窄场景原型和停止判断，复用共同回放底座 |

## 1. 建可复用的会话回放与正确性基线

**为什么现在做。** 5 月 27 日，EvalScope 已提供多轮 trace replay、逐轮 token 上限和工具耗时模拟。9 月 MLPerf Inference v6.1 增加 Edge Agentic 和端到端 RAG。11 月 Endpoints 方法进一步把长会话和负载阶梯带到 API 服务，AIPerf v0.14.0 支持 AgentX、多轮 trace 与部分引擎逐请求投机接受统计。9 月提交的 cache divergence 研究对两种引擎做了路径对照，所测缓存命中与重算可走出不同 token；11 月修订的 The Silent Hyperparameter 报告不同引擎的配置和实现把部分任务分数拉开。功能回归和性能回归可以共享复现材料。 [EvalScope](https://github.com/modelscope/evalscope) · [MLPerf Inference v6.1](https://mlcommons.org/2026/09/mlperf-inference-v6-1-results/) · [Endpoints 方法](https://mlcommons.org/2026/11/mlperf-endpoints-how-to-submit/) · [AIPerf v0.14.0](https://github.com/ai-dynamo/aiperf/releases/tag/v0.14.0) · [Cache Divergence v3](https://arxiv.org/abs/2609.04748v3) · [The Silent Hyperparameter v4](https://arxiv.org/abs/2605.19537v4)

**最小实现。** 复用一个现成 replay harness，提供两个运行模式。固定回放保持 token 化输入、原始到达间隔、工具暂停、分支／汇合与截断边界，衡量服务容量。动态回归让生成结果决定后续调用，衡量任务成功、工具参数、停止条件和失败类目。准备冷／热缓存、重启、抢占、混合长度、结构化输出和一次权重更新的样本。保存模型、tokenizer、模板、量化和引擎 SHA，打通客户端请求与引擎调度 ID。推理服务指标用统一时间戳，与追踪规范或现有系统对接。 [TensorRT-LLM 任务级 trace replay](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog27_Evaluating_Agentic_Serving_with_Trace_Replay_and_Job_Level_Metrics.md) · [inference-perf 指标](https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/metrics.md) · [OpenTelemetry GenAI](https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai)

**可用流量样本。** WEKA/Claude Code 公开 trace 保留会话时间和前缀哈希，适合缓存和容量回放。11 月 Harvard Systems Group 的 FreeInference 样本覆盖约 1.2 万真实编码会话，典型请求 129.1K 输入 token 中未命中的中位数约 491 token。作者在排除用户 idle 并截限工具间隔的模拟里，将工具耗时减半带来的任务提速约 1.38×，prefill、decode 分别约 1.09×、1.16×。因此回放需保留工具依赖与人为空档，还要区分业务负载究竟受模型、工具还是会话 harness 限制。 [WEKA 公开轨迹](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126) · [Harvard 会话研究](https://systems.seas.harvard.edu/blog/agentic-traces-freeinference/) · [llm-d 轨迹回放](https://llm-d.ai/blog/from-agent-traces-to-replay)

**验收与停止。** 首轮用 20–50 个代表性会话建立可诊断的回归样例，再扩大数据规模用于尾部估计。P99 采样不足时给样本量及不确定性。输出首个 token、首个可见答案／有效工具调用、整任务完成三个时间点。确定性契约用 token ID、分段拼接、EOS 和工具语法精确比较，量化或跨后端按业务容差和成对任务成功率比较。续写或切换失败时，先定位差分再进入正式 Agent E2E。固定 trace 可覆盖性能回放，质量结论来自动态任务验收。

**扩展切口。** 如果回归定位到 batch 形状改变了 kernel 数值路径，可先记录具体 dispatch 和 tile 选择再固定最小参数集。11 月 Detent 在两种引擎、两类模型的多卡原型中，用约 1.04–1.17× 默认 TPOT 的代价达到其 batch-invariant 条件。随机采样还需要同 RNG、processor 和一致的状态边界。工具 JSON 与投机同启时，保存草稿被拒绝后 grammar 状态和最终序列。 [Detent v1](https://arxiv.org/abs/2611.10611v1) · [XGrammar 引擎集成](https://github.com/mlc-ai/xgrammar/blob/main/docs/using_xgrammar/engine_integration.md)

## 2. 先做前缀感知调度，再开更深层的 KV

**证据链。** 5 月 25 日 UTC 的 Stateful Inference 研究主动维持会话状态，合成多轮负载表明工具暂停与重复 prefill 值得单测。7 月 Google 在 GKE 澄清多节点 KV 的 RAM/共享 Lustre 路径。9 月 KVSET 把目标命中率下最小容量作为工作集，适合先看复用间隔。10 月 llm-d 1.0 将缓存、负载感知路由和分层 KV 纳入部署方案，Google 的 Ironwood agentic coding 例子相对于普通 Kubernetes Service 报告 TTFT 中位数降逾 85%、吞吐约 2.4×，涉及多个合并改动。11 月 SGLang 继承同一编排的实作说明补了 cache warmup、pod 回收与重启路径。 [Stateful Inference v1](https://arxiv.org/abs/2605.26289v1) · [GKE 与 Lustre](https://cloud.google.com/blog/topics/developers-practitioners/scaling-llm-inference-multi-node-kv-cache-offloading-with-gke-managed-lustre) · [KVSET v1](https://arxiv.org/abs/2609.27746v1) · [llm-d 1.0](https://cloud.google.com/blog/topics/ai-infrastructure/llm-d-1-0-an-open-source-stack-for-distributed-inference) · [llm-d/SGLang 11-23](https://llm-d.ai/blog/serving-glm-5-2-sglang-tiered-kv-cache-on-llm-d)

**切口。** 基线用负载均衡和 HBM 前缀缓存。记录命中 token、可读 tier、预计复用间隔、replica 队列、eviction、模型／adapter 版本。第二臂只改路由，第三臂把队列和可用 slot 作为权重，第四臂再开 host offload。用同抵达序列和同卡数比较，每一层的读回时间包含排队、传输、格式转换与拷贝。比较单租户多轮、重复 system prompt、不同租户长文本、低重复随机短聊。租户和版本边界按已有缓存隔离策略验证。缓存命中用 token 加权，并单独跟踪活跃会话失败后的恢复时间。

**可直接实现的两个策略。** 7 月 OSDI 的 LMetric 以新 prefill token 数和当前 batch size 的乘积为评分，作者报告在生产聊天／编码流量中相对在产调度器 TTFT −39%、TPOT −51%，并做过系统 canary。它可作为简单对照。若热点副本的排队明显，可以再开远端取缓存。8 月 llm-d 的 H200 实验中，P2P 取回与重算的交点对不同模型分别低于 2K、约 8–8.7K，门槛应由模型、可用带宽和接收端负载实测。恢复也是独立收益，LMCache 的 9 月作者实验里，TP4 配置的可恢复 raw-block NVMe 路径平均 TTFT 高于传统缓冲文件方案。 [LMetric OSDI](https://www.usenix.org/conference/osdi26/presentation/zhang-dingyan) · [llm-d P2P](https://llm-d.ai/blog/p2p-kv-cache-sharing-llm-d) · [LMCache NVMe](https://blog.lmcache.ai/en/2026/09/22/raw-block-in-lmcache-building-a-fast-recoverable-nvme-tier-for-kv-cache/)

**负载差异。** 8 月 llm-d 对照 Kubernetes 轮询时，代码生成流量的输入吞吐拐点约 46K 对 16K tok/s，decode-bound reasoning 的输出吞吐约 7.0K 对 6.8K tok/s。匹配前缀对后一类影响有限。选路可让有复用的短任务优先命中热副本，副本逼近饱和时依据新 prefill token 与 active decode 序列降权。分层实现可从已有版本接入，5 月 vLLM v0.22.0 已有 filesystem tier 和 Mooncake 磁盘 offload，6 月 v0.23.0 加对象存储及兼容 connector 的 HMA 配置变化。 [llm-d 路由实验](https://llm-d.ai/blog/sticky-until-saturated-token-aware-routing) · [vLLM v0.22.0](https://github.com/vllm-project/vllm/releases/tag/v0.22.0) · [vLLM v0.23.0](https://github.com/vllm-project/vllm/releases/tag/v0.23.0)

**投资门槛。** 继续投入的起点可设为代表性流量的任务 goodput +10%，或 P95 TTFT 降 15%，且最慢租户与低命中组无有意义回退。百分比为试点目标。若命中高而尾时延恶化，优先看热点副本的排队；若重复 prefill 少，就把工作转到准入和调度；若 HBM 命中稳定且 host 读回大幅慢于重算，结束第二层优化。只有 host 容量溢出且重访概率和可用 I/O 支撑时再开共享存储。11 月 FlatCake、TripleKV、TideCache 分别给了直写 SSD、工具完成前预取和混合状态换页的实现思路，先用实测瓶颈选其中一条。 [FlatCake v2](https://arxiv.org/abs/2610.38355v2) · [TripleKV v1](https://arxiv.org/abs/2611.25112v1) · [TideCache v1](https://arxiv.org/abs/2611.26619v1)

## 3. 把投机解码做成可自适应的运行时功能

**证据链。** 7 月 DSpark 将并行 draft 与按置信度调整验证长度结合。8 月 vLLM 官方 AMD 对比展示不同 drafter、负载和实现路径之间的差异。11 月 SpecSearch 的 RTX 5090 例子中，DFlash 对 AR 的吞吐比分别为低并发 3.50× 和高并发短输出 0.76×；NarrowSpec 在 B200、GB10 的 batch=1 实验有不同幅度，额外证明 draft 宽度本身可优化。W47 的 Ascend #21137 作者测得自适应 eager 改善、同配置自适应 graph 吞吐较低，两个执行模式需要各自的策略。 [DSpark v1](https://arxiv.org/abs/2607.05147v1) · [vLLM ROCm 对照](https://vllm-project.github.io/2026/08/23/speculative-decoding-amd-gpus.html) · [SpecSearch v1](https://arxiv.org/abs/2611.22296v1) · [NarrowSpec v1](https://arxiv.org/abs/2611.23835v1) · [Ascend #21137](https://github.com/vllm-project/vllm-ascend/pull/21137)

**切口。** 先选择一个有官方 draft 的模型和一套硬件，刻画 AR、固定 `k` 和按负载的关／短／长三档。决策信号用运行中 request 数、过去窗口的接受长度、目标 verify 批规模和 decode 队列。细分代码、短输出、长推理与工具 JSON。保留 fallback 到 AR，并观察在业务流量和模式切换瞬间是否振荡。先使用查表和滞后开关，再考虑训练新的策略或 draft。

10 月 vLLM 通过已合并的 #52228 为没有 confidence head 的 draft 从 logits 估计接受概率，作者在 MiMo-V2.5-Pro-FP4 和 DFlash 的七组并发配置中有五组位于 Pareto 前沿。选用变长验证前确认框架实际减少 verify 行数及图内计算。运行时接受率估计应使用能维持目标采样语义的特征，论文或引擎已验证的采样步骤可作参考。 [vLLM 自适应验证](https://vllm.ai/blog/2026-10-14-adaptive-verification-online-estimator) · [vLLM #52228](https://github.com/vllm-project/vllm/pull/52228)

**验收与停止。** 判据以真实输出 token 产出和尾时延为准，同时计 draft 耗时、verify 行数、接受长度、批次形状、图命中率和显存。低并发例可以用 P95 TPOT 至少改善 10%，饱和时 P95 和 goodput 不突破现有 SLO。greedy 模式对齐 token 序列，采样模式用相同 processor、grammar 和 temperature 做分布与任务验证。若提升集中于固定小 subset，就以限定准入发布。若昂贵的是 draft 前向，再考虑窄草稿，若昂贵的是验证，再做 verify kernel 和元数据调度。

## 4. 为固定小标签评分验证 LM head 快路径

**适用业务。** 单请求只判断几个确定标签、单 token 选项或结构化第一步。先确认标签在真实上下文和模板下对应单个 token，这与通用 `prompt_logprobs`、多 token 标签、拒答生成和全词表采样各有不同语义。现有 vLLM `logprob_token_ids` 允许选择返回 token，不能从协议字段推出投影节省。以实际版本的 `compute_logits`、sampler、并行通信和 trace 为证据。 [vLLM completion 协议](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/completion/protocol.py) · [prompt teacher scoring](https://github.com/vllm-project/vllm/blob/main/docs/training/prompt_token_id_logprobs.md)

**近期上游机会。** 7 月合并的接口 PR 与 9 月发起的紧凑 LM head 需求分别涉及返回契约和投影实现。vLLM 的开放 PR #56782 与 11 月仍处 Draft 的 Ascend #21232 采用 V1 runner 的窄入口，后者 PR 的准入限于同一候选顺序、TP1、未量化 head、单 token greedy 且关闭投机。作者的设备投影实验属于原型。另有 9 月 25 日合入 SGLang 的 `/v1/systemone`，复用评分路径，可以作 API 基线。Dynamo DEP 中 `x_label_mass` 是候选标签在全词表中的概率质量，属于需显式保留的数学合同。 [vLLM #43463](https://github.com/vllm-project/vllm/pull/43463) · [vLLM #56782](https://github.com/vllm-project/vllm/pull/56782) · [Ascend #21232](https://github.com/vllm-project/vllm-ascend/pull/21232) · [SGLang #41208](https://github.com/sgl-project/sglang/pull/41208) · [Dynamo DEP #15274](https://github.com/ai-dynamo/dynamo/issues/15274)

**切口。** 先保持数学合同不变，测减少 host 传输、序列化和采样开销的路径。若业务允许候选集归一化，再原型化常驻候选行：分别缓存各 TP rank 持有的候选权重，对候选做投影和聚合，混合 batch 里通用生成保留全 vocab。若合同要求全词表归一化 logprob，仍需全 vocab 的 `logsumexp` 或数学等价实现；对候选 logits 做 softmax 得到的是条件分布。候选排序在普通共同归一化下可等价，但 `top-k/top-p`、约束与不同 processor 顺序可能改变任务语义。权重刷新、量化分组、词表 padding、特殊 token、TP 空 shard 和 fallback 按版本验证。

**验收与停止。** 打点冷／热 prefix、纯评分 batch 和评分混生成、不同 TP 与 token budget，分别报 head 时间、通信、队列、prefill、问题吞吐和常驻内存。先用 Amdahl 上界筛选，例如某阶段只占端到端 5%，理想消除也仅约 1.053×。若 profile 显示热态 head＋相关通信占比不足，优先优化 prefill、跨种类调度或 API 开销。远端 NPU 结果需要实测；本条为对当前工程场景的方向建议。

三条可比较的实现为 **全词表＋小结果**, **原生 score 接口**, **候选条件评分**。第一和第二条通过全词表归一化得到基线，用相同 tokenizer 和模板核对 logits、选择、分数和 `x_label_mass`。第三条单独给出候选内归一化，必须完整保留质量监控所需的字段语义。若不能放宽概率合同，可试把全词表规约留在设备上，减少 D2H 和主机后处理，并据实际内核阶段重新计算收益上界。

## 5. Ascend 长上下文优先选可归因的路径

**证据链。** 9 月 30 日 TileKernels 宣布同 API 的 Ascend 后端。DeepEP-Ascend 展示 950DT、CANN 9.2.0、PoC HDK 和指定 Clos 的 EP 通信数据，dispatch 到 EP32 的物理负载带宽比例有上游证据。该测量绑定特定固件、拓扑、消息和 kernel。vLLM-Ascend 现有文档已经把 PCP/DCP、cache pooling 与 connector 的部分组合逐项标注，W47 还合并了非均匀 PP 与新的 HiSparse/PCP 路径。 [TileKernels](https://github.com/deepseek-ai/TileKernels) · [DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend) · [Context Parallel](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/context_parallel.html) · [非均匀 PP #19284](https://github.com/vllm-project/vllm-ascend/pull/19284) · [HiSparse #21892](https://github.com/vllm-project/vllm-ascend/pull/21892)

**切口。** 以实际可用的一台 A3 或 A5 和一个模型为起点，同卡数做聚合基线、PCP、P/D 三个对照，按文档确认镜像、CANN、插件及已验证的组合。逐项测图模式与 KV 布局，确认传输包括索引与草稿状态，优先验首个 decode token、抢占恢复、数据布局、超时回退。只有 live SKU 上持续 EP 开销主导，才引入异构 EP 融合或低精度 all-reduce。粒度先到一个 layer 与消息档，后续再改体系拓扑。

**选择规则。** 长 prefill 隔离有机会降低 decode 被干扰的程度，跨节点 KV 每次交接却增加传输。采用 P/D 前要求在相同 GPU/NPU 小时约束下获得更高联合 SLO goodput。若一侧持续空转、传输挤压 TTFT，试按长度走聚合与分离两条路。GDDR/HBM 跨供应商或跨 NPU/GPU P/D 归为后续方向，先验证同模型的 cache 格式、元数据、低精度、传输带宽和整服务可用性。6 月 HMA-Serve 的结果来自作者特定模型与组合。SC26 TaiChi 给出了 P/D 合置／分离动态选择及良率对照。 [HMA-Serve v1](https://arxiv.org/abs/2606.29986v1) · [TaiChi SC26](https://sc26.conference-program.com/presentation/?id=pap819&sess=sess235) · [Ascend #21508](https://github.com/vllm-project/vllm-ascend/pull/21508)

7 月的 Prefill Deflection 在 2P2D A100、DeepSeek-V2-Lite 轨迹中发现，prefill 执行只占 P95 TTFT 的 2%–23%，余量来自队列和跨节点传输。9 月 vLLM 的两块 L40S、Qwen2.5-7B、约 8K／256、关闭前缀缓存的 PCIe 对照中，P/D 在 0.4 req/s 将 P99 ITL 从 169 降至 29 ms，在 0.2 req/s 的中位 TTFT 却从 0.7 增至 2.2 s。两个数字对应不同负载点。在自己的部署中把网络交接、设备执行和排队分别打点。D 侧有余量时，再测在约束 decode token 间隔下就地完成短 prefill 的方案。 [Prefill Deflection v1](https://arxiv.org/abs/2607.02043v1) · [EcoServe OSDI](https://www.usenix.org/conference/osdi26/presentation/du) · [vLLM P/D 实测指南](https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide)

**实际版本锚点。** 6 月的 v0.20.2rc1 已涉及 A5 的 MXFP4/MXFP8 MC2 和 DSA 多流。10 月 v0.30.0 正式版可作为一条对照环境。MegaMoE RFC 对 MC2 给出一组作者在 16K/1K、并发 32 的对照，其中 P 侧聚合输入吞吐 +6.53%、平均 TTFT −8.72%、平均 TPOT −1.04%。若自己的目标是低并发短请求，应先做相同形状的现有后端对照。量化通信按 TP all-reduce、EP dispatch/combine 分开，单列 payload、scale、dequant、未掩盖通信占比和低精度质量。 [v0.20.2rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.20.2rc1) · [v0.30.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0) · [MegaMoE RFC](https://github.com/vllm-project/vllm-ascend/issues/17452)

## 6. 有内存压力再做稀疏索引和低精度 KV

**判断顺序。** 把 HBM 拆成权重、图工作区、主 KV、索引缓存、indexer 临时缓冲和未分配余量。区分容量上限与每步读带宽。9 月发布 DeepSeek-V4.1-Flash，10 月提供 1017 新权重。vLLM 的公开配方给出了 1M context 下 H100 indexer logits 缓冲与批 token 上限如何耦合，并注明性能证据绑定 0909 checkpoint。8 月 OasisKV 探索预测稀疏读来做 in-decode 主 KV 卸载。11 月 Twindex 把索引压缩到 HBM 并把原始候选行留在 host，论文所测 1M context 上 indexer 成为 KV 卸载后的主要 HBM 开销。 [DeepSeek 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) · [vLLM 配方](https://github.com/vllm-project/vllm/blob/main/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml) · [OasisKV v1](https://arxiv.org/abs/2608.08097v1) · [Twindex v1](https://arxiv.org/abs/2611.25078v1)

**切口。** 用现有模型的默认路径测 8K／32K／128K 及业务超长档，找第一个在业务形状触顶的资源。容量导向可先验证已支持的 FP8 或 C8 KV，再决定 offload。纯带宽瓶颈则比较是否有相同 layout 的原生低比特 kernel。11 月 FOSCAR 在 B200 上 INT4 kernel 对 TensorRT-LLM NVFP4 约 1.00×，基线从 BF16 换到原生低比特收窄了优势。Twindex 可作为索引上限研究，保留长文本检索、证据引用、工具状态和多轮任务准确率评估。 [FOSCAR v1](https://arxiv.org/abs/2611.09722v1) · [FOSCAR 作者代码](https://github.com/FutureMLS-Lab/FOSCAR) · [XM-Sparse v1](https://arxiv.org/abs/2611.22280v1)

10 月 vLLM 对 GB300 的稀疏 MQA 优化只算固定候选，512K 的单层加速 14–23×，4×GB300 的整模型 decode 加速是 3–6%。第一轮保留相同 top-k 语义，分别优化 scorer、选 top-k、KV gather/dequant 和 attention。若瓶颈落在索引驻留再试压缩或转存。改变检索集合的近似方案额外过质量门槛。 [vLLM V4.1 kernel 拆解](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash) · [LiteTopK v5](https://arxiv.org/abs/2607.11976v5) · [AttentionIndex v1](https://arxiv.org/abs/2611.02508v1)

**停止条件。** 稀疏或量化配置的质量可能随文档长度和任务变化。测命中召回、可驻留并发、读带宽、重算量、TTFT 和 ITL，并对不同长度任务设质量门槛。若只改善 kernel，端到端 TTFT、TPOT 与 goodput 的置信区间还覆盖无收益，先调整调度或保留容量收益为独立目标。HBF、跨机内存池和新格式适合在容量与链路证据齐全之后再讨论采购或系统迁移。

## 7. 推理预算按任务进度调整，先从同模型两档做起

**为什么值得做。** 预算优化的对象是完整任务的成功、等待与成本，和 decode kernel 的每 token 加速可以叠加。7 月 ACL Findings 的 SelfBudgeter 探索给模型学习预算管理，10 月 BOSS 在固定 token 预算下改变多样本与校验开销的分配。11 月 OpenRouter Auto 路由说明将难度、缓存状态、工具结果和失败信号纳入，能够先改 reasoning effort，再评估切换模型带来的冷缓存代价。Qwen4-Max 的厂商测试还展示同任务输出长度本身会随模型改变。上述材料涵盖作者离线实验与供应商线上数据，基线、预算和任务集合各不相同。 [SelfBudgeter](https://aclanthology.org/2026.findings-acl.1063/) · [BOSS v1](https://arxiv.org/abs/2610.26383v1) · [OpenRouter Auto 路由](https://openrouter.ai/blog/announcements/auto-and-auto-code-routers/) · [Qwen4-Max 发布](https://www.alibabagroup.com/en-US/document-2060139246777270272)

**最小实现。** 对同一模型设业务低／高两档，只允许在可恢复的轮次边界升级。信号先取任务类别、剩余上下文、工具已返回的错误、同一失败重复、已有答案能否被客观检查。用固定低、固定高、两档策略和相同成本的简单规则作为基线。将预算控制的行为和服务的瞬时 draft 控制分别记录，方便解释收益。失败分支要能从 token 连续、cache 命中和有效 tool call 三处定位。

**验收与停止。** 报告每次成功任务成本、每轮可见输出等待、任务完成时间、失败后升级率、重复工具调用、硬任务的成功率。可以先以等质量成本下降 10%–15%，或同成本提高成功率作为 pilot 目标，细分代码任务长度和错误类型。若没有可重复的任务级评分，或路由开销与重试抵消生成节省，保持固定策略并把工作放回 trace 基线。

## 8. RL 后训练把 rollout 可追溯和 time-to-quality 放在首位

**证据链。** 7 月 1 日 AReaL 2.0 发布，项目文档讨论异步生成与训练重叠、策略滞后和部分 rollout。10 月 ThunderSyncRL 针对长尾尝试在输入齐备时流式计算梯度，保持原同步目标，作者在 SWE-bench Verified 与 Terminal Bench 4.0 场景最高 1.9× 到达相同水平。11 月 MLPerf Training v6.1 的 Qwen3.5-397B-A17B GRPO 引入质量门槛和多阶段 rollout，DaisyRose-27B 的 144×H200 案例同时给出生成、训练和 sandbox 规模。实际可立项的范围通常是先缩短某一段停顿。 [AReaL](https://github.com/areal-project/AReaL) · [异步算法文档](https://github.com/areal-project/AReaL/blob/main/docs/en/algorithms/async.md) · [ThunderSyncRL v1](https://arxiv.org/abs/2610.05935v1) · [MLPerf Training 6.1](https://mlcommons.org/2026/11/mlperf-training-v6-1-insights/) · [DaisyRose](https://huggingface.co/blog/RadixArk/daisyrose-27b-pushing-the-frontier-of-post-trainin)

**切口。** 用小模型与一组可重跑沙盒任务先建立同步 baseline，分别计时 generate、tool、reward、训练、权重同步和数据等待。每条或每段 trajectory 记录策略版本、生成 logprob、停止条件、截断、重试和工具结果。对照短队列异步或同步流式 overlap，设计 weight update 的提交边界，检查在途请求的 target/draft 一致性及旧 KV 的版本归属。可先用既有框架和小规模 worker，避免实验变量随资源规模扩大。

5 月 Hugging Face 的 TITO 工程文给出 token 边界的实现样例。复用模型真实采样的 token IDs，将工具回包单独追加，loss mask 只标生成 token。6 月 Prime-RL v0.6.0 提供长 rollout 和权重更新的实现入口；9 月 PACE 将生成和队列等待分别纳入策略滞后。训练数据流水的第一个测试可以检查轨迹重建、工具输出插入和逐 token logprob 归属，再衡量异步速度。 [TITO](https://huggingface.co/blog/huggingface/tito) · [Prime-RL v0.6.0](https://github.com/PrimeIntellect-ai/prime-rl/releases/tag/v0.6.0) · [PACE](https://arxiv.org/abs/2609.36830)

**验收与停止。** 用相同初始 checkpoint、质量门槛、任务集合和调度预算比较达到目标的墙钟与 GPU 小时。单独报策略 lag 的 P95／最大值、训练占用、有效 rollout、长轨迹保留率和异常率。若异步改善 GPU 利用率但验证质量或数据分布漂移，优先验证同步 overlap 与 sandbox 尾部，保留同步作为诊断回退。

## 9. 有明确负载后再立项的方向

| 方向 | 何时适合做 | 第一轮对照与证据 |
| --- | --- | --- |
| MoE 小批融合、通信与 GEMM 重叠 | profiler 显示低 batch launch、EP 尾部或小消息规约占比突出，且路由与形状稳定 | 同精度、同 token、同拓扑对比融合与成熟 backend，占用、带宽、rank token 偏度及数值另表。6 月 SGLang/GB300 约 5× 同交互速度的作者结果叠加了 kernel、缓存、图和调度。W47 Weave 原型的 SM90 层级 1.24×，整模型 4K decode 1.08×。 [PyTorch/SGLang 优化](https://pytorch.org/blog/serving-deepseek-v4-on-gb300-with-sglang-5x-higher-throughput-at-the-same-interactivity-since-day-0/) · [Weave v1](https://arxiv.org/abs/2611.27765v1) |
| 算子生成与正确性验证 | 已有可复用的 kernel 基准和硬件位点，能定义输入与误差合同 | 定一个算子及三类形状，先做对照实现和差分用例。候选通过数值、图兼容与饱和吞吐才留存。11 月 EngineTailor 是特化端侧例，TensorLean 在给定 PTX 模型和规范内验证编译实例。 [EngineTailor](https://www.infinigence-ai.com/news-updates/256.html) · [TensorLean](https://www.certik.com/blog/tensorlean-certik-and-columbia-university-researchers-formally-verified-production-ai-kernels-from-deepseek-and-flashattention) |
| 流式多模态阶段调度 | 已有实时音频、视频生成或多路用户并发需求 | 从 vLLM-Omni 正式 release 跑一条会话，对照阶段拆分与单进程，同步测首音频/视频块、中断后的恢复、不同会话 cache 隔离及输出质量。8 月 SGLang 的 Glimmer NVFP4 RTX 5090 decode 例可作为端侧文本基线，整链需加入视觉或音频阶段。 [vLLM-Omni v0.34.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.34.0) · [Glimmer 作者测量](https://www.lmsys.org/blog/2026-08-10-meta-muse-glimmer) |
| 扩散 LLM 的 block 级批处理 | 已有必须部署的 Dream／LLaDA 类模型或固定的离线业务 | 7 月 BlockServe 对 Fast-dLLM 报告作者离线吞吐改善。10 月 PV-Decoding 单／双批在同预算上有收益，多批路径较慢。优先比较同模型、同有效输出质量、同资源下的达标产出，区分并行去噪步和输出 token。 [BlockServe v1](https://arxiv.org/abs/2607.08930v1) · [PV-Decoding v1](https://arxiv.org/abs/2610.24722v1) |
| 边缘的低比特、循环 MoE 与专家缓存 | 尺寸、带宽、功耗或离线要求会约束可用设备 | 在同设备上用同质量档测模型载入、首句、整任务、峰值常驻与热稳定性。Inkling 2 Mini 的 3.4B 活跃与 5.4B 每 token 权重激活反映跨循环执行。MLPerf Edge Agentic 提供质量和任务基准口径。 [Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/) · [MLPerf Inference 6.1](https://mlcommons.org/2026/09/mlperf-inference-v6-1-results/) |

## 先用两周做一次立项筛选

| 时间 | 交付件 | 决策 |
| --- | --- | --- |
| 第 1–2 天 | 两份冻结负载，分别为多轮和短请求／小标签，另有小型动态回归。固定引擎、镜像、权重、并行和采样。拉通请求与阶段时序。 | 用固定样本定位首处分歧，建立低并发和饱和基线。 |
| 第 3–5 天 | 负载阶梯上的冷/热、缓存命中、队列和 head profile。对照普通路由与前缀感知路由。 | 记录排队、prefill、传输、decode 和工具各阶段用时。选出占用最大且可控的环节。 |
| 第 6–8 天 | 只切一个变化，按分型做 host cache 或自适应投机对照。固定标签评分占主流时可替换成 head 路径试验。 | 看同资源、同任务质量下的 Pareto，记录饱和和退化区间。 |
| 第 9–10 天 | 用独立 trace 和多次交叉运行复测最优两臂。包含高优租户突发、重启和混合流量。 | 通过门槛才排期上游合并或部署验证。收益落入噪声区间时，提交可复用回放和瓶颈记录，由下一项接手。 |

### 公用指标与判定顺序

- **完整任务。** `任务 goodput = 在约定 SLO 内完成且通过动态质量校验的任务数 / 测量墙钟时间`。固定回放另报请求级 SLO goodput、累计完成的会话与失败比例。并发分档下分别报每卡或每小时值，避免把输出更短的配置按同 token 数替代成任务收益。
- **交互链路。** 首个生成 token、首个非推理输出或可执行工具调用、TPOT/ITL 与整任务时延。注意单 token 打分请求没有有意义的 TPOT，使用完成时延和问题/s。长会话适合同时报告真实工具耗时与模拟暂停的区分。
- **性能归因。** GPU/NPU 常驻显存、KV/索引、图工作区；排队与调度；通信／数据拷贝；prefill、head、verify 与实际 GPU kernel 分开收数。计算上界可用 `新吞吐 / 原吞吐 ≤ 1 / (1 - 阶段时间占比)` 做串行近似，重叠与批变化需实测。
- **实验统计。** 同输入做配对比较，轮换 A/B 运行顺序，分开冷启动和稳态。报告任务覆盖、样本量、变异和失败，优先观察差异区间及最差任务簇。试点阈值根据既有业务 SLO 与成本目标调整。

## 半年证据索引

| 时间 | 里程碑与阅读目的 |
| --- | --- |
| 05–06 | Stateful Inference 与 EvalScope 将工具轮次及前缀状态带进回放；6 月 SGLang Spec V2＋DFlash 解释 overlap 和验证队列；HMA-Serve 加入异构 P/D 的传输、格式和显存对照。 [Stateful](https://arxiv.org/abs/2605.26289v1) · [EvalScope](https://github.com/modelscope/evalscope) · [SGLang Spec V2，06-15](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/) · [HMA-Serve](https://arxiv.org/abs/2606.29986v1) |
| 07 | DSpark 有论文和 SGLang 集成路径；Google/llm-d 描述共享 KV 存储；AReaL 2.0 提供 rollout 框架对照。 [DSpark](https://arxiv.org/abs/2607.05147v1) · [SGLang 集成](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/) · [GKE Lustre](https://cloud.google.com/blog/topics/developers-practitioners/scaling-llm-inference-multi-node-kv-cache-offloading-with-gke-managed-lustre) · [AReaL](https://github.com/areal-project/AReaL) |
| 08 | OasisKV 的 in-decode 预取、vLLM ROCm 投机实测以及 Glimmer 边缘推理证明实验必须按阶段、架构和并发拆分。 [OasisKV](https://arxiv.org/abs/2608.08097v1) · [ROCm](https://vllm-project.github.io/2026/08/23/speculative-decoding-amd-gpus.html) · [Glimmer](https://www.lmsys.org/blog/2026-08-10-meta-muse-glimmer) |
| 09 | MLPerf 6.1 新增 Agentic/RAG；KVSET 把工作集提到容量规划；TileKernels、FlashMLA 和 DeepEP 对 Ascend 有公开实现和基准。 [MLCommons](https://mlcommons.org/2026/09/mlperf-inference-v6-1-results/) · [KVSET](https://arxiv.org/abs/2609.27746v1) · [TileKernels](https://github.com/deepseek-ai/TileKernels) · [FlashMLA Ascend，09-30](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20260930-ascend-prefill-deep-dive.md) · [DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend) |
| 10 | vLLM 记录分块稀疏 indexer 的显著算子加速和有限端到端增益；llm-d 1.0 让 cache/router 组合可部署；ThunderSyncRL 提供同步训练优化设计。 [vLLM DeepSeek V4.1，10-07](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash) · [llm-d 1.0](https://cloud.google.com/blog/topics/ai-infrastructure/llm-d-1-0-an-open-source-stack-for-distributed-inference) · [ThunderSyncRL](https://arxiv.org/abs/2610.05935v1) |
| 11 | vLLM v0.34.0、LMCache v0.5.8、FlashInfer、AIPerf、Omni 等提供分层落地条件；SC26、Endpoints 和 MLPerf Training 系统化地补充了负载与验证方法。 [vLLM 0.34.0](https://github.com/vllm-project/vllm/releases/tag/v0.34.0) · [LMCache 0.5.8](https://github.com/LMCache/LMCache/releases/tag/v0.5.8) · [Endpoints](https://results.mlcommons.org/endpoints) · [SC26](https://sc26.supercomputing.org/Program/) · [W47 本地报告](../weekly/2026-W47-expanded.md) |

## 信源与验证边界

方向检索沿用雷达信源，按月份补搜官方工程发布、论文、项目代码、会议、云服务与基准。入选证据直接链接 arXiv 版本、厂商作者文档、GitHub tag/PR 或会议页。W47 的核心引擎 REST 分页、并行工程仓库补查及未配置渠道的替代路径由本地周报保存。本文采用定向采样形成投入决策，完整半年的候选总量、未公开线上流量、私有部署和独立硬件结果仍缺数据。优先级以有现成服务、可控模型、可复用评测的小型工程团队为设定。 [信源配置](../../config/sources.yaml) · [W47 信源与边界](../weekly/2026-W47-expanded.md#信源日期和剩余盲区)
