# LLM 工程与推理技术雷达 · 2026-W47

**统计周期**　北京时间 2026-11-16 00:00 至 2026-11-23 00:00，右端不含。核查日期为 2026-11-23。范围涵盖推理引擎、算子与编译、内存与通信、低精度、模型架构、后训练基础设施和服务基准。

**11-24 专题补充**　回查 GLM-5.4 循环层、SpecPilot 对昇腾 DSpark 的可迁移优化，以及 Uno、PARD-2 与扩散投机。W47 之后公开的材料依实际日期标注。 [阅读专题](#专题补充)

**基线报告**　周报分支提交 `c08b11604e193d76d08f4ad17a5df92cf383af9a` 的报告截止于 11-19 16:16，解释了 5 个工作，其余为候选链接。基线文件 blob 为 `84c3e8499bfc47c154328338d0d4cd1c42a15696`。本报告沿用初始排查的 67 项信源清单。 [基线报告](https://github.com/F00L42/llm-inference-radar/blob/c08b11604e193d76d08f4ad17a5df92cf383af9a/reports/weekly/2026-W47.md) · [信源清单](../../config/sources.yaml)

## 工程结论

| 优先级与方向 | 本周出现的判断依据 | 建议先测 |
| --- | --- | --- |
| P0 · 引擎升级与缓存正确性 | vLLM v0.34.0 默认采用 extensible KV，GPU 仅保留 MRV2。混合注意力需要完整 KV、滑窗尾部和递归状态在同一位置可恢复。 | 在线容量、前缀热/冷命中、抢占后逐 token 一致性、P/D 交接与结构化输出。 [v0.34.0](https://github.com/vllm-project/vllm/releases/tag/v0.34.0) · [SGLang #51488](https://github.com/sgl-project/sglang/pull/51488) |
| P0 · 投机解码的最优配置随负载变化 | MRV2 异步深度在特定 DSpark 配置中有收益，昇腾作者对照里自适应 eager 和自适应 graph 的收益方向不同。 | 分别扫并发、有效接受长度、草稿预算、图模式和队列深度，追踪 TPOT 与 SLO 达标产出。 [vLLM #64145](https://github.com/vllm-project/vllm/pull/64145) · [vLLM-Ascend #21137](https://github.com/vllm-project/vllm-ascend/pull/21137) |
| P1 · 长上下文存储分层与选路 | Twindex、TripleKV 与 llm-d 分别改变索引扫描、缓存预取和请求准入。瓶颈可能从 HBM 驻留转到主机带宽、SSD 尾延迟或队列。 | 同时记录索引和 KV 字节量、读取放大、回退率、重新 prefill token、P99 TTFT。 [Twindex](https://arxiv.org/abs/2611.25078v1) · [TripleKV](https://arxiv.org/abs/2611.25112v1) · [llm-d](https://llm-d.ai/blog/multi-tenant-multi-priority-model-serving-llm-d) |
| P1 · 新模型给常驻内存与每 token 成本提出不同约束 | Inkling 2 Mini 复用中间层权重并多次执行，Hy4 flash 把稀疏注意力、低维专家与 Engram 组合，Ripple 选择窄深 dense 和长短窗口。 | 分别记录常驻权重、每 token 权重读取、KV/索引、跨卡通信。以固定成功率下的任务用时比较不同架构。 [Inkling](https://thinkingmachines.ai/news/inkling-2-mini/) · [Hy4 flash](https://huggingface.co/tencent/Hy4-flash) · [Ripple](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) |
| P1 · API 服务与训练 rollout 的瓶颈需要端到端测量 | MLPerf Endpoints 展示负载阶梯与尾延迟。Training v6.1 加入 agentic GRPO，DaisyRose 提供异步 rollout 和 sandbox 运行数据。 | 服务侧测会话成功率和 P90/P99，训练侧测 time-to-quality、policy lag、权重同步停顿与 sandbox 等待。 [Endpoints](https://mlcommons.org/2026/11/mlperf-endpoints-how-to-submit/) · [Training v6.1](https://mlcommons.org/2026/11/mlperf-training-v6-1-insights/) · [DaisyRose-27B](https://huggingface.co/blog/RadixArk/daisyrose-27b-pushing-the-frontier-of-post-trainin) |
| P1 · 周末出现可锁定版本的改动 | TensorRT-LLM v1.3.0、LMCache v0.5.8、AITER v0.1.30、AIPerf v0.14.0、vLLM-Omni v0.34.0 均有正式 release。 | CI 固定轮子和镜像。先检查缓存身份、量化内核选择和压测停止规则，再比较升级前后的业务指标。 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0) · [LMCache](https://github.com/LMCache/LMCache/releases/tag/v0.5.8) · [AITER](https://github.com/ROCm/aiter/releases/tag/v0.1.30) · [AIPerf](https://github.com/ai-dynamo/aiperf/releases/tag/v0.14.0) · [Omni](https://github.com/vllm-project/vllm-omni/releases/tag/v0.34.0) |

## 推理栈升级与远端重点

### vLLM v0.34.0 改变显存规划与 GPU runner 的默认路径

11-17 北京时间正式发布。Extensible KV cache 默认依据 warmup 后可用显存调整 KV 容量，GPU Model Runner V1 移除。升级时应把历史利用率配置、图捕获、模型/LoRA 加载和 connector 作为同一份容量实验的输入。以并发阶梯检查短请求与长请求混跑时的 KV 使用率、抢占次数和 OOM，并在 reload 后再测一次。发布说明给出上游产品状态，实际容量随模型、并行和运行时变化。 [发布 v0.34.0](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)

### 复核远端 5 项深读

| 上游工作 | 远端记录的阶段与机制 | 升级到实验决策所需证据 |
| --- | --- | --- |
| vLLM ROCm split-KV | 11-19 16:11 主干合并。AITER FlyDSL 针对 head dim 64、GQA 8、FP8 KV 形状加入 decode 与小规模投机 verify 专用路径，按 KV 长度划分工作。 | 对照 MI355X、Muse Spark TEP4 图重放，分别检查全注意力和 SWA、单请求及长短混跑，报告 reduction 开销、ITL 与吞吐。作者在限定形状测过 kernel 与解码步。 [vLLM #67133](https://github.com/vllm-project/vllm/pull/67133) |
| SGLang 分页专家拷贝融合 | PR 仍开放，11-22 23:28 有更新。Triton／Marlin 将缺失专家读取与其余专家计算重叠。 | 以专家常驻比例分桶。在同一模型测 PCIe 或高速互联读取、内核占用、专家层停顿和任务级延迟。需要可比作者实测或集成后的服务基准。 [SGLang #51471](https://github.com/sgl-project/sglang/pull/51471) |
| SGLang 混合 GDN 缓存 | PR 仍开放，11-20 10:45 有更新。实验实现可在一次 extend 中保留分支快照和 prompt 尾检查点。 | 检查 cutoff 对齐、共享前缀上的分叉、连续第二次命中和驱逐后恢复。远端证据覆盖指定 CUDA Triton GDN/BF16/eager 组合，其他服务组合待测。 [SGLang #51488](https://github.com/sgl-project/sglang/pull/51488) |
| Prime-RL 上游接口迁移 | RFC 梳理下游补丁涉及的准入、采样、调度、辅助输出、加载和服务接口。 | 按契约逐项迁移。对相同 rollout 流量检查输出 token、logprob、取消请求及权重版本边界。 [vLLM #68616](https://github.com/vllm-project/vllm/issues/68616) |
| Ascend MRV2 非均匀 PP | 远端截点为开放 PR，11-20 17:09 主干合并。连续非空层可按任意大小切分，边界需要同时传激活、索引缓存及附属状态。 | 同权重对照均匀和非均匀分区，用各阶段峰值内存、bubble、边界传输量和 token 连续性衡量。 [vLLM-Ascend #19284](https://github.com/vllm-project/vllm-ascend/pull/19284) |

ROCm split-KV 的 PR 还给出一组同配置的前缀重放。4×MI355X、Muse Spark Open TEP4、FP8 KV、DSpark7 合成接受长度 4.05、60K 共享前缀＋4K 后缀、输出 1,024 token、并发 16。作者每组只跑一次，输出吞吐 532.0 → 572.7 tok/s，ITL p50 40.84 → 32.60 ms，TTFT p50 6,194 → 5,850 ms。可用该结果确定复测形状，下一轮追加实测接受分布及重复样本。 [vLLM #67133](https://github.com/vllm-project/vllm/pull/67133)

### 11-19 前值得保留的工程对照

- **投机解码与异步调度。** vLLM #64145 作者用双节点 GB300、TP8、DeepSeek-V4.1-Flash 与 DSpark 对比异步队列深度，PR 给出同主干单请求 899 → 942 tok/s 的一组数据。加深队列还要预算在途状态和锁页内存。Ascend #21137 的 A3、TP4/DP4、混合 P/D、并发 10 作者对照中，自适应 eager 有改善，自适应 graph 的三轮吞吐均低于固定预算 graph。两个结果对应不同引擎和硬件。 [vLLM #64145](https://github.com/vllm-project/vllm/pull/64145) · [vLLM-Ascend #21137](https://github.com/vllm-project/vllm-ascend/pull/21137)
- **稀疏 decode 优化的测量层级。** vLLM ROCm 作者在 MI355X、TP8、1K/1K、单请求上报告索引 metadata 单次 launch 使 ITL p50 约 4.219 → 4.195 ms，压缩 KV 写回链路 21.78 → 14.26 μs 的另一项改动在作者服务试验中没有可测 ITL 改善。SGLang 用 tile 因数选择 split 数，H200 上受益形状集中于减少 partial 合并的较高并发。长短混跑时应把索引、KV 写入、attention 和 CPU 调度分别计时。 [vLLM #67363](https://github.com/vllm-project/vllm/pull/67363) · [vLLM #67364](https://github.com/vllm-project/vllm/pull/67364) · [SGLang #45274](https://github.com/sgl-project/sglang/pull/45274)
- **P/D 传输需要包含草稿与混合状态。** SGLang #41334 补齐混合注意力 P/D 路径里的 NextN 草稿索引状态，Ascend #21497 让 connector 接受 C8 packed KV，而 LMCache #6626 的 MI355X ROCm VMM IPC 证据来自作者组合补丁。测试按本地跨进程、跨机两档记录格式、数据拷贝次数、注册和 pin 开销、回退率、首个 decode token 及异常重连。Ascend 真机验证为 `pending_remote`。 [SGLang #41334](https://github.com/sgl-project/sglang/pull/41334) · [vLLM-Ascend #21497](https://github.com/vllm-project/vllm-ascend/pull/21497) · [LMCache #6626](https://github.com/LMCache/LMCache/pull/6626)
- **Ascend A5 量化 all-reduce。** #21508 合并路径默认关闭。启用后 A5 默认 MXFP8、A2 默认 INT8，A3 不进入该实现。双块流水需单独启用，图捕获走整张量。PR 列出模拟进程组测试。本地真机收益为 `pending_remote`，对照需要按 TP rank、消息大小和拓扑分桶，记录量化与反量化各自耗时及数值差异。 [vLLM-Ascend #21508](https://github.com/vllm-project/vllm-ascend/pull/21508)

## 三大推理引擎周末补查（11 月 20–22 日）

覆盖口径以北京时间为准。16–19 日沿用周报及 `origin/data` 冻结记录，20–22 日逐日分页核对 GitHub 官方合并查询，共返回 vLLM 195 项、SGLang 276 项、vLLM-Ascend 81 项，含发布分支回移植和维护项。入选条目另用 PR API 核对 `merged_at` 和目标分支。GitHub Releases 上，本周三仓仅核实到 vLLM v0.34.0 正式发布（11 月 17 日）。SGLang 最新可见 release 为 v0.5.23（11 月 3 日），vLLM-Ascend 为 v0.32.0rc1（11 月 11 日，预发布）。下列性能与正确性数据均来自上游，本轮远端硬件验证为 `pending_remote`。 [vLLM Releases](https://github.com/vllm-project/vllm/releases) · [SGLang Releases](https://github.com/sgl-project/sglang/releases) · [vLLM-Ascend Releases](https://github.com/vllm-project/vllm-ascend/releases)

### vLLM

- **11 月 20 日合并，Hopper 的 MoE 后端多了两个低精度路径。** DeepGEMM Mega MoE 在 SM90 直接使用 GLM-5.3-Flash 的块 FP8 权重及 FP32 scale，省去 MXFP8 重量化。作者在 H200、无 MTP/EPLB、llm-d 前缀路由下报告，每卡 KV 池随所选后端变化。把 DP4 各组都固定为 24 GiB KV 池后，在 AgentX 并发 16/64 上，对主干默认 MoE 的吞吐增幅为 12%/16%，对 FlashInfer MoE-EP 为 +10.5%/−2.1%。同日的 FlashInfer MoE-EP 新增 SM90 原生 MXFP4 checkpoint 路径，显式选用。DeepSeek-V4.1-Flash 在 8×H200、DP8/EP8、128K 输入/512 输出、合成 DSpark3 接受率下的 +8% 至 +18% 输出吞吐来自 10 月 28 日的历史对照，合并前更新另有 CPU/API 回归。 [#66812](https://github.com/vllm-project/vllm/pull/66812) · [#64289](https://github.com/vllm-project/vllm/pull/64289)
- **11 月 20–21 日合并，GDN 降低 decode 每步成本。** ROCm 把 conv1d、递推和 gated RMSNorm 交给一个 AITER kernel。在 MI355X、Qwen3.8-27B BF16 TP1、1K/1K、并发 16–64 的作者测试中，总 token 吞吐 +2.2% 至 +3.1%，平均 TPOT −2.3% 至 −2.8%。CUDA 路径则为 BF16/FP16 递推状态增加融合 MTP kernel，GB300 的算子测量约 1.3–1.6×；4×GB300、Qwen3.8-Flash-Next-FP8、DEP4、MTP3、30,720 token 缓存前缀的 decode-only 服务测量中，最高每步约快 5.3%，单请求仍使用原 kernel。 [#61428](https://github.com/vllm-project/vllm/pull/61428) · [#69065](https://github.com/vllm-project/vllm/pull/69065)
- **11 月 21–22 日合并，分层 KV 压缩了副本并修正回收顺序。** 单节点 TP>1 的 HiSparse + OffloadingConnector 让 CPU tier 保存一份 TP rank 共享的 indexer；4×B200、GLM-5.3-NVFP4 TP4 的作者验证包含组合补丁 #68771。CPU→文件层现在可以把超出 CPU tier 容量的请求分批写入；作者用 13 块 CPU tier 落盘了 119 块，重启后读到 7,616 个缓存 token。ARC 把运行中请求已复用的前缀也纳入尾部优先淘汰，GB10 的四交错请求用例中，后续重放的缓存 token 由 0 增至 1,664，单 token 重放延迟中位值 268.3 → 133.9 ms，同组并发生成耗时近似持平。MultiConnector 的分层 offload 同时增加按连接器指定读写方向的配置，可以组合 CPU→文件写和 GDS→GPU 读，PR 未列服务 benchmark。 [#68785](https://github.com/vllm-project/vllm/pull/68785) · [#66719](https://github.com/vllm-project/vllm/pull/66719) · [#69110](https://github.com/vllm-project/vllm/pull/69110) · [#67961](https://github.com/vllm-project/vllm/pull/67961)
- **11 月 22 日合并，RTX 5090 上的 Inkling-2-Mini 小 M GEMM。** 对窄输出使用确定性 split-K，对 loop blend 用批量 GEMM，并把三路独立输入的 Q/K/V 投影合到一次 launch。作者以 FP8 KV、FlashInfer、DSpark7、随机 1K/1K 测试，单并发平均 ITL 约 10.70 → 9.90 ms，输出吞吐约 323 → 347 tok/s；并发 2 的平均 ITL 降约 6.5%，两组草稿接受长度不同。派发覆盖 SM121，DGX Spark 实测待补。 [#69168](https://github.com/vllm-project/vllm/pull/69168)

### SGLang

- **11 月 20–21 日合并，两种可选的 prefill 排队策略改动。** UniBoost 按未命中前缀的 token 数给短任务有界的优先级提升，TP/PP 模式只读取跨 rank 一致的数据。PR 的 8×B300、DeepSeek-V4-Flash 编码 agent trace 上 TTFT p50 155 → 36 s、输出吞吐 314 → 405 tok/s 来自更早的 v0.5.13 移植，合并分支的 GB300 对照仍待发布。Shortest-prefill-first 扩展另外用队列未缓存 token 数动态收紧短作业门槛，重载时长请求按到达顺序，轻载时缩小单批 prefill token 上限；默认关闭，PR 未列公开性能数据。 [#35637](https://github.com/sgl-project/sglang/pull/35637) · [#51957](https://github.com/sgl-project/sglang/pull/51957)
- **11 月 21–22 日合并，P/D 让已拿到 decode 预留的请求先推进。** 开启 optimistic prefill 时，尚无 decode 预留的暂存请求若占满 prefill KV，调度器可释放其中一个的 KV 并重新排队。启用 L3 buffer 模式时也可回收一条尚无预留请求所占的 host staging 或 anchor，让已预约的请求提交其前缀命中；让步方可能重读存储。这组改动针对双向资源等待。另一项 HiCache 改动在本机各 rank 锁页前协调一次可用预算采样，作者以双节点各 4×GB300、Kimi-K3.1 P/D prefill、约 205 GB host pool/rank 验证启动，gfx950 显式配额仍待真机核对。 [#51882](https://github.com/sgl-project/sglang/pull/51882) · [#51892](https://github.com/sgl-project/sglang/pull/51892) · [#52120](https://github.com/sgl-project/sglang/pull/52120)
- **11 月 21–22 日合并，投机验证收拢采样语义，DSA draft 缩减 prefill 工作。** 验证逐行计入草稿树/链上的历史 token，EAGLE、NGRAM、DSpark 与 DFlash 共享惩罚、语法掩码、bias、custom logits processor 的处理顺序，并补齐 min-p。FP32 概率的 block verification 另为小批加词表分块 Triton 路径，GB300 表内效果随词表、批量及接受情况变化。CUDA 的 GLM-5 系列及 DeepSeek-V3.2 还有独立的 draft prefill 优化，满足门槛时仅最后一枚 prompt token 跑 draft attention、DSA indexer 和 MoE，其余 token 只写 draft KV。作者在 GLM-5.3-Flash、H200 上的部分 8K/32K 输入形状报告输入吞吐 +2.8% 至 +3.7%；prefill 图未覆盖的 B300 个别形状约慢 2%，分段额外约 3.5 ms CPU 时间。 [#52054](https://github.com/sgl-project/sglang/pull/52054) · [#52184](https://github.com/sgl-project/sglang/pull/52184) · [#51961](https://github.com/sgl-project/sglang/pull/51961) · [#46784](https://github.com/sgl-project/sglang/pull/46784)
- **11 月 20–21 日合并，昇腾模型适配和 prompt 打分扩展。** Kimi K3.1 接通混合 KDA/DSA 的 NPU KV pool 与索引路由，3 节点各 8×Ascend 950 DT、TP8/PP3 的作者验证报告 GSM8K 200 题 99%。可选 P/D KV 校验另在 A3、A5 做过传输验证，A3 服务验证包含 early-send 修复。独立的 prompt logprob 扩展让部署侧挂接逐位置候选选择器，后续 PR 把候选权重沿 prefill 分块、detokenizer 和响应链返回；默认实现没有提供选择器，二者也未给出 LM head 投影或吞吐收益的实测。 [#48420](https://github.com/sgl-project/sglang/pull/48420) · [#41056](https://github.com/sgl-project/sglang/pull/41056) · [#51899](https://github.com/sgl-project/sglang/pull/51899) · [#51909](https://github.com/sgl-project/sglang/pull/51909)

### vLLM-Ascend

- **11 月 20 日主干合并，词表权重在细粒度 TP 与标准 TP 的乘积网格上切分。** 先前跨 DP 的细粒度切分会复制标准 TP rank 的词表分片。新版减少每卡权重和每步读量，LM head 要再做一次 TP all-gather、embedding 要再做一次 TP all-reduce，DFlash2 候选交换也覆盖组合分片。作者在 16×A3、MiniMax-M3.1-Flash-W8A8、DP4×TP4/EP16 和 DSpark 下测得两表开启后每卡权重 −0.86 GiB，扣掉新 HCCL 组约 0.47 GiB，净增 KV 预算约 0.39 GiB/卡，eager 与图模式通过 E2E。PR 未测得对应 TPOT/吞吐数据。 [#19841](https://github.com/vllm-project/vllm-ascend/pull/19841)
- **11 月 20–21 日主干合并，长上下文的 KV 与并行布局。** DeepSeek V4.1 的实验性 MRV2 流水线支持非空连续层的任意切分，PP 边界传活跃的 KV/索引、Top-K 和 DSpark 状态。作者早期 A3 16 卡、PP2×TP8、23/17 层分割的 W4A8 测试中 GSM8K 1279/1319，缺同权重对照，A5 当前头部 E2E 跳过。Kimi 路径同时接通 PCP prefill、DCP KV 和 DSpark：prefill 仅 gather 引用的 KV 页，draft 查询逻辑复制而物理 KV 仍分片。作者在较早版本、64×A5、约 128K/1、89.09% HBM 命中率的组合对照中（DCP8 + MegaMoE + #22656 对 DCP1 且关闭 MegaMoE），客户端 TTFT 6.576 → 6.232 s，QPS 1.753 → 1.853；另一个 DCP 受控单轮中，逻辑 KV 容量增加而 TTFT 略增、输入吞吐略降；最终合并头真机三轮对照仍待补。 [#19284](https://github.com/vllm-project/vllm-ascend/pull/19284) · [#22181](https://github.com/vllm-project/vllm-ascend/pull/22181)
- **11 月 20 日主干合并，Ascend 950 的 HiSparse 与 PCP 结合。** MRV2 支持 BF16 / native C8 的 prefill PCP 和 decode 按请求切分，历史 KV 跨 PCP rank 复制，适用范围暂不含 DCP、MultiConnector 或外部 P/D。上游 BF16 在 PCP2、GLM-5.3、TP4/EP8、MTP3 + decode graph 下完成了小规模服务及 6/6 次强制 Host→Device 恢复检查。较早的 C8 组合栈在 32K/8K、相同 18.8 GiB KV 池下跑完 16 请求，输出 460.97 → 583.65 tok/s、抢占 2 → 0，两个 arm 的实际 decode 批次有差异。另有 GLM-5.3-Flash P 侧 layerwise KV pool 的 offload；作者在 P 侧 TP8/DP1、FP8 KV、MTP5 的对照中，开启前后每 rank 的 HBM KV 预算同为 23.92 GiB，offload 使池 KV 块由 3,137 增到 5,663。 [#21892](https://github.com/vllm-project/vllm-ascend/pull/21892) · [#21256](https://github.com/vllm-project/vllm-ascend/pull/21256)
- **11 月 20–21 日主干合并，同步与算子融合。** All2AllV dispatch 用锁页缓冲、异步 split D2H 和事件替代预处理中的阻塞路径；作者在 A3、Qwen3.8-Flash-Next-W4A8、TP2+EP、eager、8K prefill 测得主机 dispatch 均值 −7.9%，TTFT −0.63%。Step5-Preview 的 A5 残差/RMSNorm/BF16 融合在 950DT 上算子快 2.20–3.25×，512/128 单并发的平均 TPOT 7.642 → 7.290 ms，20 个随机 prompt 中相同文本 6 个，完整模型评估待补。另一组 head-wise gate 和 indexer 融合在双节点 950DT、DP4×TP4/EP16、MTP3、64K/4K 单独打开时，服务 ITL 改善 0.10% 至 2.00%，作者报告单次波动约 ±1%。 [#21907](https://github.com/vllm-project/vllm-ascend/pull/21907) · [#22336](https://github.com/vllm-project/vllm-ascend/pull/22336) · [#22401](https://github.com/vllm-project/vllm-ascend/pull/22401)
- **11 月 20 日合并的正确性修复，按硬件和分支核对。** RecoverSSM 小范数修复先恢复 A3 的 verify/commit 一致性，随后按原生算子的设备版本分支，A3 保留 epsilon，A5 按原生 L2 归一化并处理零向量。作者以 vLLM 0.33.0、Triton Ascend 3.6.0.post2 在 A3、A5 各跑过 229 项算子测试。另有两个合入 `releases/v0.32.0rc` 的回移植，分别为 Mooncake 同步恢复排序在 NPU block zeroing 之后，以及 DeepSeek-V4.1 bounded replay 在关闭投机时继承 MoE 层游标。A2 下游确认修复后 96 个热读首 token argmax 与冷读对照相同；A5 bounded replay 的下游 GSM8K 每组 3,957 道计划输入，正确数 2,905 → 3,833，两组超时数及首轮重试策略不同。 [#22523](https://github.com/vllm-project/vllm-ascend/pull/22523) · [#22686](https://github.com/vllm-project/vllm-ascend/pull/22686) · [#22491](https://github.com/vllm-project/vllm-ascend/pull/22491) · [#22591](https://github.com/vllm-project/vllm-ascend/pull/22591)

## 其他推理引擎、缓存与算子工程

- **TensorRT-LLM v1.3.0 正式发布（11 月 20 日 10:45，北京时间）。** Rubin SM107 以预览形式提供，发行说明称已在 Vera Rubin 上验证。PyTorch 后端新增 P/D 的 gen-first 调度与会话亲和路由，多数模型默认选用 KV Cache Manager V2，并接入 NVFP4 冷页压缩、磁盘缓存及 LMCache/FlexKV 连接器。投机解码新增基于接受率的自动停用，稀疏注意力新增 DSA FP4/FP8 decode 与 GVR Top-K。升级时需要核对 TensorRT 后端移除后的迁移路径，并检查 CUDA 13.4.1 和 PyTorch 2.14.0 依赖。同周的 `v1.4.0rc4` 是另一条候选分支，包含稀疏 KV offload、共享 drafter KV 池等功能，发行说明列出了在特定 CUDA Graph、低精度和并行配置下的正确性、启动或预热问题。 [v1.3.0](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0) · [v1.4.0rc4](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc4)
- **LMCache v0.5.8 正式发布（11 月 22 日 23:39）。** 多进程实现重整 client、common 和 transport 边界，移除旧 `InProcess` 实现。缓存身份纳入标签、模型版本、LoRA 与 TP 驻留等因素，P2P 新增 RDMA 写入、对等存储 RPC 和分块搬运。NIXL 改成独立 `lmcache-nixl` 包，停止提供 CUDA 12.9 镜像和 wheel。升级验证应覆盖同前缀但模型版本或 LoRA 不同的请求，以及 RDMA 注册、布局、重试与按 TP 组去重。`v0.5.8` 在本周发布，当前发行说明的 GitHub `updated_at` 对应北京时间 11 月 23 日 03:03。 [v0.5.8](https://github.com/LMCache/LMCache/releases/tag/v0.5.8)
- **Dynamo 的 Kimi K3 专用预发行镜像（11 月 18 日 10:29）。** `v1.5.1-kimi-k3-post.1` 基于 Dynamo 1.5.1 和 SGLang `v0.5.18`，含 DCP P/D 下的 DSpark 草稿 KV 传输和 SM100 MoonViT 图像编码修复。GB300 配方包含 16 卡聚合、以及 16 卡 1P1D，后者以 Mooncake 在 MNNVL 上传 KV。官方 trace 对两者采用 36 与 18 并发，且合成草稿接受长度为 4.05。比较架构时需统一并发和接受长度，分别观察 TTFT、用户输出速度与每卡输出吞吐。 [专用镜像与实测条件](https://github.com/ai-dynamo/dynamo/releases/tag/v1.5.1-kimi-k3-post.1)
- **FlashInfer 同周发布 `v0.7.5`（11 月 19 日 07:03）、正式补丁 `v0.7.5.post1`（11 月 21 日 03:34）及候选版 `v0.7.6rc1`（11 月 21 日 04:33）。** `v0.7.5` 将 DeepSeek-V4.1 的 Prims-TS 稀疏 MLA decode 标为实验路径。`v0.7.6rc1` 增加 SM120 FP8 稀疏 MLA prefill 融合、QToken-KvBlock 稀疏注意力的 NVFP4 KV cache，以及 Blackwell/Rubin 批不变 BF16 MegaMoE。按 GPU、输入布局、精度、批形状观察实际后端分派与输出一致性。 [v0.7.5](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.5) · [v0.7.5.post1](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.5.post1) · [v0.7.6rc1](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.6rc1)
- **ROCm AITER v0.1.30 正式发布（11 月 22 日 17:53）。** 将配置、优先级、来源和可观测性引入 kernel provider 接口，已有五种算子经新接口分派，发行预编译 wheel 的 GPU 架构为 `gfx942;gfx950`。HIP MLA decode 增加宽 split-K reduction，适用于少量工作单元、很多 split 的情况。发行说明亦记录 `gfx950` FP8 QH16 单 token MLA decode 修复，以及 `gfx1250` 上 Unified Attention 的 `SPEC_DECODE` 路径。应分别核对 wheel 支持和各架构源码路径。 [v0.1.30](https://github.com/ROCm/aiter/releases/tag/v0.1.30) · [kernel provider #7465](https://github.com/ROCm/aiter/pull/7465) · [MLA split-K #7570](https://github.com/ROCm/aiter/pull/7570)
- **推理基准 AIPerf v0.14.0 正式发布（11 月 21 日 04:48）。** TensorRT-LLM、SGLang 加入逐请求投机接受统计，可按 SPEED-Bench 类别从单次运行输出结果。新增 AgentX harness 和从 Hugging Face 加载 Mooncake trace 的支持。新版基准集默认完整遍历一次，多轮请求在某一轮被 `max_tokens` 截断时终止。跨版本复跑应固定数据集、对话停止规则、压测并发和引擎侧投机统计口径。 [v0.14.0](https://github.com/ai-dynamo/aiperf/releases/tag/v0.14.0)
- **vLLM-Omni v0.34.0 正式发布（11 月 22 日 20:46）。** 自回归和生成阶段迁移至 Model Runner V2。双工服务可把 stage 副本分布到独立运行时，同时把会话固定在副本组。CausVid 在生成期间输出解码分块，Noisy Pipeline Parallelism 可对兼容工作跨请求组批，同时保留各会话 KV 状态。对应测试指标是首音频或首视频块时间、后续块间隔、会话亲和，以及新增组批过程中的 KV 隔离。 [v0.34.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.34.0)
- **编译与量化的版本更新。** Helion `v1.6.0` 于 11 月 16 日 02:22 正式发布，CuTe DSL 包含小行数 gated jagged attention、按 tile 等待的 grouped `tcgen05` 路径和 split-K reduction 工作区改动。Humming `v0.1.20` 于 11 月 17 日 03:53 正式发布，SM90 跳过填充 K 块，SM103 调整 MXFP4×MXFP8 的 MoE 流水和启发式，NVRTC 路径减少编译工作。FlashAttention 于 11 月 18 日 16:12 发布 FA4 `beta40`，包含分页 KV 页表和空查询块正确性修复。ModelOpt `0.49.0rc0` 于 11 月 17 日 01:05 预发行，官方说明仅给出候选 wheel 安装入口。 [Helion](https://github.com/pytorch/helion/releases/tag/v1.6.0) · [Humming](https://github.com/vllm-project/humming/releases/tag/v0.1.20) · [FlashAttention](https://github.com/Dao-AILab/flash-attention/releases/tag/fa4-v4.0.0.beta40) · [ModelOpt](https://github.com/NVIDIA/Model-Optimizer/releases/tag/0.49.0rc0)

### 默认分支上的工程变化

- **Mooncake 数据路径。** 11 月 19 日 22:32 的 `#4540` 为经典 RDMA GPU WRITE 加入显式开启的目的端本地 rail 选择。在共享 NIC 拓扑中以目标 GPU 的首选 HCA 作为发端初始提示，已有显式提示继续优先。11 月 20 日的 `#5644` 接入 UB Transport 的 URMA RM 模式。`#5673` 允许 DrainJob 迁移 hard-pinned 对象。本周查到 94 个默认分支提交，GitHub Release 无周内匹配。 [#4540](https://github.com/kvcache-ai/Mooncake/pull/4540) · [#5644](https://github.com/kvcache-ai/Mooncake/pull/5644) · [#5673](https://github.com/kvcache-ai/Mooncake/pull/5673)
- **Dynamo、NIXL 与 llm-d。** Dynamo `#19236` 在 admission 积压时提高已准备负载视图的可复用性，`#19270` 修复 prefill hop 的输出块跟踪。NIXL `#2666` 将 Mooncake 插件的能力声明移至 backend 构造阶段。llm-d `#3042` 为 KEDA + EPP 的队列信号增加 GKE 夜间 E2E 测试路径，属于队列信号的集成覆盖。 [Dynamo #19236](https://github.com/ai-dynamo/dynamo/pull/19236) · [Dynamo #19270](https://github.com/ai-dynamo/dynamo/pull/19270) · [NIXL #2666](https://github.com/ai-dynamo/nixl/pull/2666) · [llm-d #3042](https://github.com/llm-d/llm-d/pull/3042)
- **跨引擎及传输正确性。** LMDeploy `#5119` 让 DeepEP 容量与 masked FP8 GEMM 对齐，`#4977` 修正 P/D 的 prefill chunk 进度与缓存 owner ID。xLLM `#2805` 封装 Ascend950 分页 KV 写入，`#2803` 对齐调度和缓存的 prefill 上限。UCCL `#1171` 在 poller 关闭前排空接收的 IPC 操作，`#1172` 仅构建选定的默认 EP dispatch/combine 配置。 [LMDeploy #5119](https://github.com/InternLM/lmdeploy/pull/5119) · [LMDeploy #4977](https://github.com/InternLM/lmdeploy/pull/4977) · [xLLM #2805](https://github.com/xLLM-AI/xllm/pull/2805) · [xLLM #2803](https://github.com/xLLM-AI/xllm/pull/2803) · [UCCL #1171](https://github.com/uccl-project/uccl/pull/1171) · [UCCL #1172](https://github.com/uccl-project/uccl/pull/1172)
- **编译与通信。** Triton `#12876` 在 single-warp `tcgen05` MMA 路径省去 warp-zero predicate，`#12864` 修正转置 NVMMA shared layout 的轴顺序。Triton-distributed `a0292af` 在 NVIDIA ring reduce-scatter 中改用对称 per-node buffer 并传递 stream。CUTLASS `#3900` 的 FMHA 性能改动在同周已由 `#3905` 回退。 [Triton #12876](https://github.com/triton-lang/triton/pull/12876) · [Triton #12864](https://github.com/triton-lang/triton/pull/12864) · [Triton-distributed a0292af](https://github.com/ByteDance-Seed/Triton-distributed/commit/a0292af9337b20ebef7a6c8a43ee09b9466569be) · [CUTLASS #3905](https://github.com/NVIDIA/cutlass/pull/3905)

### 工程补查口径

按北京时间 `[2026-11-16 00:00, 2026-11-23 00:00)` 换算 UTC，使用 GitHub REST 按 `published_at` 筛发行，按 `commit.committer.date` 筛默认分支提交。以发行 tag／ID、PR 链接与提交 SHA 标识证据。共核查来源清单中 32 个编号对应的 42 个仓库（含 DeepEP 手动补查，排除由核心采集覆盖的 vLLM、SGLang、vLLM-Ascend），取得 116 条周内发行记录与 3,163 条周内默认分支提交。12 个仓库有发行记录，其中 `llama.cpp` 的 86 条为滚动构建，LMCache 的 13 条包含候选版、多平台 wheel 和 nightly。发行按获取的最近一页（最多 100 条，异常仓库以 15 条分批补查）计数，默认分支提交按时间分页，覆盖范围限于这些仓库与 API 返回数据。工程补查为 11 月 23 日回看，不作为原采集首次观察时间。发行说明和 PR 内作者实验属于上游证据，本轮独立性能复现为 `pending_remote`。

## 模型架构、后训练与部署成本

### Inkling 2 Mini · 26.4B 循环 MoE 开权重，11-20

中间层每 token 执行三遍。作者给出 3.4B 活跃参数、计入重复循环后 5.4B 权重激活，大部分权重为原生 NVFP4、末四层为 MXFP8，报告内存约 20 GB。发布图中的单请求 decode 速度是 614 GB/s、BF16 权重假设下的理论上限。可变循环研究版可单独扫循环次数，在 KV／计算重复、输出长度下降与任务成功率之间对照。实际部署应测冷启动、常驻内存及 KV 增量、不同循环数的实际输出 tok/s 和完整任务用时。 [Thinking Machines 公告和权重入口](https://thinkingmachines.ai/news/inkling-2-mini/)

### Hy4 flash · 稀疏索引、滑窗、LatentMoE 与 Engram，11-18 开权重

官方模型卡给出 155.5B backbone 和 63.5B Engram 等独立组成，文本每 token 激活 11.3B，最大上下文 1,048,576。42 层中有 12 个稀疏层及 30 个 SWA 层，SWA 因果窗为 512，稀疏端按组选择最多 2,048 token。512 个路由专家每 token 选 16 个，另有 1 个共享专家。作者称 KV 容量约为 Hy4 的 1/6，后训练的平均输出 token 也有所下降。部署比较应拆开 Engram 表读、稀疏索引、MLA/SWA KV 字节与 EP 调度，用同样任务完成条件测长上下文。官方提供普通、FP8 与 base 权重，API 与各权重仓单独定版。 [腾讯官方模型卡](https://huggingface.co/tencent/Hy4-flash) · [发布帖](https://x.com/TencentHunyuan/status/2122912575302119554) · [代码与部署](https://github.com/Tencent-Hunyuan/Hy4-flash) · [FP8 权重](https://huggingface.co/tencent/Hy4-flash-FP8)

### Ripple-65B Preview · 窄深 dense 与长短注意力混合，W47 发布

65B dense，80 层、hidden size 5,120，提供 BF16、NVFP4、base 和配套 DFlash2。局部层用滑动窗口和 RoPE，全局层用全注意力和 NoPE，并带门控注意力、FFN 激活裁剪与 value residual。官方模型卡报告 2×B200、batch 8、量化和定制 kernel 之前为 250 tok/s，按该作者口径理解。检查 1Mi 上下文下全局层 KV 增量、长 prefill 的 FFN 临时张量峰值，以及低并发和多卡并发下的实际 DFlash2 接受长度。 [发布](https://www.ricursive.com/ripple) · [模型卡](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) · [模型实现](https://huggingface.co/ricursiveAI/Ripple-65B-Preview/blob/main/modeling_ricursive.py)

### Qwen4-Max · 2.4T 总参数、71B 活跃参数，11-16 官方公告

阿里披露 MoE 改动涉及注意力、残差连接和 embeddings，支持原生视觉和 1M token 上下文。自有 Work360 在 High effort 下报告较 Qwen3.8-Max 少用 35% token、平均任务成本约降 60%。该口径混合模型输出长度、服务策略和价格。服务工程上可以通过固定任务和缓存冷热状态，单独比较请求花费、TTFT、输出 token 与成功率。发布形态为官方 API，11-17 港版公告为同一发布事件。 [阿里集团公告](https://www.alibabagroup.com/en-US/document-2060139246777270272)

### K2-Horizon · W47 技术报告，权重在 9 月已发布

arXiv:2611.21003v1 于 11-16 提交。该家族覆盖 0.9B dense 至 375B-A23B MoE，36B-A4B 采用作者称为 Mixture of Value Attention 的架构。机构提供模型、训练材料与部署入口，375B 模型卡标注 512K context。可作为不同规模和注意力方案的可复现样本。比较时应以相同精度、引擎和 KV 驻留量为对照，并固定训练阶段 checkpoint。 [arXiv v1](https://arxiv.org/abs/2611.21003v1) · [9 月原发布](https://ifm.ai/k2/press-release/) · [375B 模型卡](https://huggingface.co/IFM/K2-Horizon-375B-A23B)

### DaisyRose-27B · 异步 rollout、训练与 sandbox，11-18

从 Qwen3.8-27B 起步，公开 BF16、FP8、NVFP4 三种 checkpoint。作者以 128×H200 做 agent 任务推理、16×H200 做训练，每步下发新权重。最长 session 可达 1M token，通过 harness 摘要形成多条轨迹，单窗口承袭基座 256K。相近训练对照中，作者报告端到端约 16.8K 对 8.3K 训练 token/s，训练端 MFU 37% 对 16%。发布运行使用内部 trainer，Miles 提供公开 RL 框架。复测需要检查 policy lag、每步同步阻塞、sandbox p95 启动时间、最长轨迹裁切和 time-to-quality。 [RadixArk 技术说明](https://huggingface.co/blog/RadixArk/daisyrose-27b-pushing-the-frontier-of-post-trainin) · [Miles 代码](https://github.com/radixark/miles)

## 服务调度、云路由与基准

- **llm-d 多租户、多优先级架构，11-16。** 队列派发与 router 准入形成反馈回路。router 从 vLLM 等引擎等待请求信号判断饱和并发布 Prometheus 指标，异步派发器据此收紧取队列预算。部署验证需同时观测优先级隔离、突发下拒绝和排队、配额回收及缓存命中漂移。路由命中带来的 prefill 节省应和热门副本排队代价一起测。 [llm-d 官方设计](https://llm-d.ai/blog/multi-tenant-multi-priority-model-serving-llm-d)
- **SageMaker AI 跨区域推理路由，11-18 公告、11-19 技术文。** 复用注册的实时、异步 endpoint 和 inference component，优先本区域，对持续错误、限流和超时降健康分，之后渐进回流。响应暴露目的地与理由，允许目的地限额。健康算法以请求失败等信号为主，成功但偏慢的服务仍会接收流量。适合用工程回放区分容量不足、持续慢化、远端冷缓存和重试成本。 [AWS 公告](https://aws.amazon.com/about-aws/whats-new/2026/11/sagemaker-cross-region-routing/) · [机制说明](https://aws.amazon.com/blogs/machine-learning/keep-inference-running-through-regional-capacity-shortfalls-with-cross-region-inference-routing-in-amazon-sagemaker-ai/)
- **Hugging Face Inference Providers 接入 OpenRouter，11-16 技术文、11-17 changelog。** 聊天模型路由可综合价格、速度、可靠性和缓存，并提供固定供应商、`:fastest`、`:cheapest` 等策略。指定供应商时禁用跨供应商 failover。评测应保存每次请求的实际 provider、是否回退、缓存状态与工具调用协议。在回归基线中固定供应商，再单独测默认路由的可用性和成本权衡。 [HF 公告](https://huggingface.co/blog/openrouter) · [Changelog](https://huggingface.co/changelog/inference-by-openrouter) · [路由文档](https://huggingface.co/docs/inference-providers/routing)
- **MLPerf Endpoints v1.0 本周新增正式结果，11-18；方法文 11-20。** v1.0 已有较早批次。11-18 的官方结果包括 8×H200 vLLM 的 `1.0.0.deepseek-v4_1-flash.agentic_combined.1` 与 4-GPU Vera Rubin 的 `1.0.0.deepseek-v4_1-flash.mlperf_agentic_inference_dataset.3`。这两条的 dataset 标识和配置不同。方法文描述的 agentic coding 类有 613 条多轮轨迹，LoadStairs 在同一 endpoint/模型/软件组合中分档增压。读图时同时看系统 tok/s、TTFT P90 和交互速度，比较前先对齐 dataset、规则、模型与容量，区分预公布和正式结果。 [Pareto Explorer](https://results.mlcommons.org/endpoints) · [方法与评审](https://mlcommons.org/2026/11/mlperf-endpoints-how-to-submit/) · [开源客户端](https://github.com/mlcommons/endpoints)
- **MLPerf Training v6.1 引入 agentic GRPO，11-16 美国公告。** Qwen3.5-397B-A17B 采用 R2E-Gym 的 held-out 任务，规则用 pass@4 0.69 作为质量目标。官方工程分析中，Google 的一项方案采用 128 芯片 trainer slice、128 个四芯片 rollout slices，容忍至多两版本策略滞后。NVIDIA 文列出同为 1,024 GPU 的 Rubin 两种 rollout 栈提交 `6.1-0117` 和 `6.1-0118`，分别约 10.44 和 10.94 分钟，是整个训练任务达标时间。该基准把仿真环境、推理、奖励、通信与训练耦合在一起，排查时需让各阶段可独立计时。 [MLCommons 发布](https://mlcommons.org/2026/11/mlperf-training-v6-1-results/) · [提交分析](https://mlcommons.org/2026/11/mlperf-training-v6-1-insights/) · [规则](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc) · [NVIDIA 提交说明](https://developer.nvidia.com/blog/?p=125323)

## SC26 会议信号

会议以芝加哥当地时间排期，下表用北京时间。报告时间与论文首发拆开记录。数字来自作者在会议议程所附摘要中的实验，作为复现线索。

| 报告 | 机制、证据与适用边界 |
| --- | --- |
| TaiChi · 11-19，`pap819` | 以同一份 TTFT、TPOT 联合 SLO 比较 P/D 合置与分离，再用流动 decode 与按长度调度 prefill 给接近违约的请求腾出阶段余量。作者报告 goodput 最高 +40%。复测需扫宽松／严格的两维 SLO、网络占用和序列长度分布。 [SC26](https://sc26.conference-program.com/presentation/?id=pap819&sess=sess235) |
| BidKV · 11-19，`pap1108` | KV 短缺时结合可释放容量和重算代价选择牺牲请求，实现复用 vLLM/SGLang 原生抢占。作者在 Llama-3.1-8B-Instruct、ShareGPT、300 ms 阈值下报告相对各自原生策略达标率分别提升 14.8 和 38.8 个百分点。需额外验证长请求公平性、逐租户 TTFT 和热点前缀寿命。 [SC26](https://sc26.conference-program.com/presentation/?id=pap1108&sess=sess210) · [作者代码](https://github.com/CGCL-codes/BidKV) |
| TIDE · 11-19，`pap577` | 利用已完成推理的 target hidden states 在线适配 draft，依负载开关投机与训练，把推理和训练分派到不同 GPU 类别。作者最高 1.66× 吞吐对应无投机基线。建议与同成本的静态 draft、定期离线训练一起测试，采集漂移恢复时间和数据搬运量。 [SC26](https://sc26.conference-program.com/presentation/?id=pap577&sess=sess230) |
| Spec-K · 11-19，`pap319` | 用草稿对 token 难度的预测动态分配 MoE Top-k，联合改变验证计算和 EP 流量。作者报告分布式推理最高 1.72×，并报告小幅质量变化。与固定模型投机不同，此处改变 target 路由计算，需独立进行任务质量、token 级偏差和 rank 热点检查。 [SC26](https://sc26.conference-program.com/presentation/?id=pap319&sess=sess230) |
| LLMProf · 11-19，`pap579` | 自顶向下逐层取瓶颈指标，作者报告案例定位后应用具体优化最高达 2.09× 吞吐。部署时单独量 profiler 开销以及它定位队列、缓存、内核阻塞和链路空转的粒度。 [SC26](https://sc26.conference-program.com/presentation/?id=pap579&sess=sess230) |
| OpServe · 11-17，`ws_aihpc107` | 高性能计算集群借用短时闲置 GPU 时动态调整 worker 的 prefill/decode 角色，临时 worker 收回后可复用已完成的 prefill KV。作者用三类流量和 8B/14B 样本，报告相对相同资源下固定 P/D 分配 P95 端到端延迟最高改善 15.4×。关注临时资源被回收时的回退开销和最坏延迟。 [SC26 Workshop](https://sc26.conference-program.com/presentation/?id=ws_aihpc107&sess=sess509) |
| RACER · 11-19，`pap128` | 生成式推荐中的 embedding 缓存与 KV 争用 HBM，作者联合动态分配和节点选路，三个数据集相对最佳静态策略 P99 降 24%–38%。适合作为混合缓存池设计参考，迁移前需比较 embedding／KV 命中和更新分布。 [SC26](https://sc26.conference-program.com/presentation/?id=pap128&sess=sess210) |
| CoreFlow · 11-19，`pap104` | 按 workflow 分配资源，再为 agent 按上下文长度分区，缓存放置随分区调整。作者在代表性工作流报告任务延迟下降 57.6%、任务吞吐上升 63.1%。对照时保留原轨迹中各 agent 的前缀、长短请求和工具停顿。 [SC26](https://sc26.conference-program.com/presentation/?id=pap104&sess=sess235) |
| Grape · 11-19，`pap251` | TaskFlow 提供声明式编排，执行层拆出可并行的微任务，以工作流 SLO 约束引擎调度。需要在独立任务多与依赖链长两类流量比较调度、取消、重用和 GPU 利用率。 [SC26](https://sc26.conference-program.com/presentation/?id=pap251&sess=sess210) |
| DiffSpec · 11-19，`pap547` | 依据序列位置调整草稿树、KV 选择与融合验证。作者摘要列出 4×H100、最长 128K 下对 EAGLE-3、TriForce、MagicDec 的端到端加速分别为 7.72×、3.45×、2.36×。复测应先对齐草稿模型、目标质量、长度分布和服务并发。 [SC26](https://sc26.conference-program.com/presentation/?id=pap547&sess=sess230) |

## 论文与版本跟踪

| 版本事件 | 机制与复现要点 |
| --- | --- |
| NarrowSpec · `2611.23835v1`，11-18 提交 | DFlash/EAGLE-3 草稿残差宽度减半或减到四分之一，DFlash 输出 adapter 折叠进词表投影。SGLang、Qwen3-8B NVFP4 与 GPT-OSS-20B MXFP4 的作者对照在 batch 1、温度 0 下报 B200 1.06–1.12×、GB10 1.13–1.44×，较大 batch 的最高值 1.21×。分别扫接受长度、草稿开销和验证占比。 [arXiv v1](https://arxiv.org/abs/2611.23835v1) |
| SpecSearch · `2611.22296v1`，11-17 提交 | 以成本模型和校准联合搜索策略、草稿长度和精度。作者 RTX 5090 示例中 DFlash 在低并发吞吐为 AR 的 3.50×，高并发短输出为 0.76×。用同一卡型的请求分布与内存占用做决策，保留 AR 作为对照。 [arXiv v1](https://arxiv.org/abs/2611.22296v1) |
| XM-Sparse · `2611.22280v1`，11-17 提交 | 动态选择的稀疏 KV 用 tile 内加载与 attention 重叠，可预测的 indexer key 用 GPU 发起 DMA 预取。作者用 SGLang 实现，在仿真的 scale-up 内存池上评测 DeepSeek-V3.2、GLM-5.3。真机需要测突发读、互联争用和索引命中。 [arXiv v1](https://arxiv.org/abs/2611.22280v1) |
| FlatCake · `2610.38355v2`，10-28 首投，W47 修订 | DRAM、NVMe 平级寻址，写入时按会话连续摆放，GDS 直达 GPU 有界窗口，DRAM/SSD 请求各自排队。作者在 DeepSeek-V4-Flash 三组轨迹上报告，接近最优基线排队拐点时，平均 TTFT 为基线的约 1/1.40–1/2.42。在 Claude Code 轨迹上作者报 91.6% 对 71.1% 命中率。受布局、GDS 与负载影响，复测需测冷读和并发预取挤占。 [arXiv v2](https://arxiv.org/abs/2610.38355v2) |
| LaFlash · `2611.22746v1`，11-17 arXiv 记载提交 | 从可收回／休眠的 coding agent 状态出发，将 QLC 高带宽闪存的访问按 session 成组，减少随机恢复对并行读的影响。属候选硬件架构，需检查模拟假设对应的介质延迟、尾部重启成本及写入耐久。 [arXiv v1](https://arxiv.org/abs/2611.22746v1) |
| The Silent Hyperparameter · `2605.19537v4`，5 月首投，W47 修订 | 作者固定权重、解码参数与硬件比较五种后端，报告所测最大分数差 16.6 个百分点，归因涉及前缀缓存、图执行、kernel 和 logit 处理默认值。版本迁移的性能表应附完全相同配置下的逐 token 输出和关键业务成功率。 [arXiv v4](https://arxiv.org/abs/2605.19537v4) |
| SpecPilot · `2611.02501v1`，11-02 首投、11-20 本地回补 | cost model 找 verify 的瓶颈，实验机制包括无损 12-bit 权重格式、混合循环目标的树验证和两级验证，并用 token forcing 对比 stock。论文与 HF 通过 arXiv ID 归并。间接引用和上游讨论单列为 RFC，集成状态待核。 [arXiv v1](https://arxiv.org/abs/2611.02501v1) · [HF 11-04](https://huggingface.co/papers/date/2026-11-04) · [vLLM RFC #65813](https://github.com/vllm-project/vllm/issues/65813) |

### 周中至周末发现的论文

日期为北京时间，提交与修订时间取自 arXiv 版本历史，首次观察取自雷达存档或本次补查。下列量化结果来自论文作者。文中提到的 SGLang、vLLM 实现为论文实验配置，主干合并和正式发布状态待核。

- **TideCache（arXiv:2611.26619v1，11 月 20 日提交，11 月 23 日补查）。** 混合模型在全局注意力 KV、滑窗和循环状态之间按需重分配物理页，同时保持虚拟地址稳定。把全局注意力层的前缀 KV 分层预取到复用的 GPU 槽位，用有界状态层的计算覆盖 CPU→GPU 传输，生成后的 KV 保留在 HBM。GB300、GH200 上七组模型与硬件配置的聊天与编码 agent 混合实验显示，相对调优的 SGLang + HiCache，最多完成 2.04 倍 agent 轮次（GB300 上限 1.44 倍），总体 P50 每 token 延迟下降 20%–59%。相同 batch size 时，单个 decode step 的延迟最多增加 6.3%。 [arXiv v1，§4–5](https://arxiv.org/html/2611.26619v1)
- **Twindex（arXiv:2611.25078v1，11 月 19 日提交，11 月 20 日首次观察）。** 主 KV 卸载后，1M 上下文下 indexer cache 占 Qwen3.8-Flash-Next 和 DeepSeek-V4.1-Flash 单请求 HBM 的 63% 和 75%。Twindex 先用 HBM 中的压缩索引做候选筛选，再用主存里的原始索引行精排。相对仅卸载主 KV，Qwen、DeepSeek-V4.1-Flash、GLM-5.2 的可驻留请求数分别为 1.78、1.62、3.03 倍。在主存不限制 batch 时，decode 吞吐分别为 1.44、1.25、2.54 倍。所测基准平均得分较基线最多低 3.1%。 [arXiv v1](https://arxiv.org/abs/2611.25078v1)
- **TripleKV（arXiv:2611.25112v1，11 月 19 日提交，11 月 20 日首次观察）。** 预测工具调用完成时间，在 agent 恢复推理前把 SSD 上的 KV 预取到 DRAM，并为从 GPU 淘汰的块补充主存缓存。作者以单卡 H100 NVL、Qwen3-30B-A3B、vLLM v0.24 为基座，评测最多 20 个并发 agent。四个数据、浏览与编码任务中，关键路径上的 SSD 读取请求占比从 45%–76% 降至 10% 以下，相对作者扩展为三级缓存的 vLLM，P95 TTFT 最多下降 49%。 [arXiv v1，§6](https://arxiv.org/html/2611.25112v1)
- **TriAxialKV（arXiv:2605.17170v2，5 月 17 日首投，11 月 20 日修订并首次观察该版本）。** 根据距离当前轮次的远近、输入模态和语义角色为 KV 分配 INT2/INT4，并通过融合 Triton kernel 解码。在单卡 B200 上以 Qwen3-VL-32B-Thinking 回放 OSWorld 轨迹，相对同用 FlashInfer prefill 的 SGLang BF16，混合精度有效 KV 容量为 4.51 倍，端到端吞吐从 7,787 增至 10,274 tok/s。作者另行报告实跑 OSWorld 的任务准确率与 BF16 接近。 [arXiv v2，§4](https://arxiv.org/html/2605.17170v2)
- **草稿 KV 量化（arXiv:2611.20565v1，11 月 16 日提交，11 月 19 日首次观察）。** K4V4 对 key 按通道分组、value 按 token 量化，目标模型和其 KV 保持 BF16。六组模型配对中，草稿缓存缩小 66.20%，双缓存合计缩小 23.95%，平均接受率从 57.93% 降至 57.14%。论文表 8 的 batch=1 原型中，1K/16K 上下文均出现端到端延迟上升和峰值显存持平，且 batched verifier 在 BF16 与 K4V4 两条路径上都出现了偏离 target-only greedy 的结果。 [arXiv v1，表 8](https://arxiv.org/html/2611.20565v1)

#### 本周提交、11 月 23 日公告

- **LongCat Recall Sparse Attention（arXiv:2611.27233v1，11 月 20 日提交）。** 将历史键按时间桶做增量索引，用组中心初选页面，再对页内成员精排，论文中的 SGLang 实现可在前缀匹配后复用兼容索引。H20 上 1024K 上下文相对 DSA 的键选择为 4.70 倍，包含索引维护的整模型推理在相同最终选择预算下为 1.865–1.984 倍。 [arXiv v1，表 2](https://arxiv.org/html/2611.27233v1) · [arXiv 公告列表](https://arxiv.org/list/cs.CL/recent)
- **Weave（arXiv:2611.27765v1，11 月 20 日提交）。** 以每卡路由统计选择 MoE megakernel 的 SM/warp 通信与计算配比，并在执行中切换 warp 分工。作者测得 MoE 层相对 DeepGEMM-MegaMoE 在 EP8 SM103、SM90 上的几何平均加速为 1.30、1.24 倍。相对 RCCL+AITER 的 EP16 CDNA3 层测为 1.44 倍。8 卡 SM90 的整模型对照中，4K 上下文的 decode 几何平均加速为 1.08 倍，16K prefill 为 1.01 倍。 [arXiv v1，§IV](https://arxiv.org/html/2611.27765v1) · [arXiv 公告列表](https://arxiv.org/list/cs.DC/recent)
- **CoPyro（arXiv:2611.27217v1，11 月 20 日提交）。** 将检索复制的 n-gram 候选替换并行草稿器输入中的掩码，由草稿器修正候选。Qwen3-4B 上 10 万样本训练的四种草稿器平均接受长度提高 2.19%–5.08%。71 万样本训练的 DFlash 在所测尺寸与温度下增幅为 1.08%–2.00%，Qwen3-8B 在 Transformers 上处理候选增加约 0.24 ms/轮。 [arXiv v1](https://arxiv.org/html/2611.27217v1) · [arXiv 公告列表](https://arxiv.org/list/cs.CL/recent)

## 工程正确性与自动优化

- **TensorLean，在 PTX 层验证 GPU kernel，11-20 团队公告。** 以张量级规范描述计算，在给定输入假设、PTX 模型和浮点误差范围内，用 Lean 检查结果正确性、数据竞争与终止。团队检查 149 个编译实例，其中 83 个通过规范。公司公告称发现 38 个不同问题，后续 GitHub 项目页列 39 项，计数应绑定各自快照。项目页将论文、PTX 语义及证明源码列为后续发布。本轮通过 11-23 机器之心回查到 W47 的官方公告。可将它用于评估未来算子证明与数值回归的覆盖范围。 [CertiK 技术公告](https://www.certik.com/blog/tensorlean-certik-and-columbia-university-researchers-formally-verified-production-ai-kernels-from-deepseek-and-flashattention) · [作者项目](https://github.com/TensorLean/TensorLean)
- **EngineTailor 面向工作负载定制本地引擎，11-17 团队案例。** Agent 围绕同一模型、硬件与应用生成改动，再经独立输出检查和性能测量挑选候选。团队在 16 GB M1 Pro、同一 Clef 权重的视觉游戏决策中报告中位 3.91 → 1.92 s，39 次决策的模型总耗时约 152 → 76 s。场景特化程度很高，可借鉴的是把功能对齐、时延与资源约束作为三项独立验收。 [无问芯穹技术案例](https://www.infinigence-ai.com/news-updates/256.html)

## 专题补充

### GLM-5.4 循环层的已知结构与可复现实验

**发布边界。** GLM-5.4 的官方技术公告发表于 11-23，属于 W48。公告列出 GLM Coding Plan、ZCode 和 Z.ai Chat，独立 API 列为即将推出，权重列为计划发布。截至 11-24，在官方 GLM-5 仓及 Hugging Face 组织未核实到可下载的 GLM-5.4 权重或配置。现阶段可复现的实验应基于有权重的循环模型，GLM-5.4 的细节以公开技术说明为准。 [Z.ai 技术公告](https://z.ai/blog/glm-5.4) · [GLM-5 官方仓](https://github.com/zai-org/GLM-5) · [Z.ai 官方模型组织](https://huggingface.co/zai-org)

| 结构 | GLM-5.4 披露的做法 | 推理侧需验证的成本 |
| --- | --- | --- |
| 循环计算 | 从已预训练的约 750B GLM-5 基座继续训练，将中间 28 层的同一组参数执行两遍。第二遍以第一遍产生的隐状态为起点。 | 按实际 layer visit 记计算、权重读取和图节点。参数可复用，每 token 仍多执行一遍循环块。 |
| 渐进写入 | 每遍有独立的逐 token 门控，对候选更新加权后写入残差流。作者观察到第一遍通常写得较少，第二遍进一步更新，第一遍的写入幅度也随 token 难度变化。 | 门控控制残差更新量。公告把按 token 跳过或增加循环次数列为后续工作。测 residual/gate 融合对中间张量和数值误差的影响。 |
| 跨循环 KV | 第二遍读第一遍稀疏选出的历史 KV，同时维护第二遍自己的局部滑窗。两个流允许覆盖同一位置，因此历史表示与更新后的近邻表示都可参与注意力。 | 额外逻辑缓存由一份全长缓存变为局部窗口。验证双流寻址、重复位置的语义、滑窗驱逐和前缀恢复，分开计 HBM 驻留及每步读量。 |
| 稀疏索引 | 第二遍复用第一遍 indexer 给出的索引。作者称所测结果接近每遍重新索引。 | 同时记录索引构造与搬运、稀疏 gather、注意力计算和图填充。第二遍仍需读 KV 并做注意力。 |

上表依据 Z.ai 的架构披露。其 30B 同参数对照使用约多 10% 的训练计算，在 GPQA-Diamond、MMLU-Pro 等多数所测任务报告平均输出 token 减少 10%–14%。旗舰线上峰谷速度是厂商披露，公告缺少复现实验所需的硬件、并发及对照参数。开放实验应固定任务成功标准，分别报每 token 时间、输出长度和完成任务的总耗时。 [Z.ai 技术公告](https://z.ai/blog/glm-5.4)

| 相关工作 | 与 GLM-5.4 的可比点 | 可验证方向 |
| --- | --- | --- |
| Universal Transformer、Huginn、Ouro | 从共享深度到可调 recurrent depth。Huginn 与 Ouro 提供可查的开权重研究模型；Ouro 用深度分配相关的训练目标。 | 选择开放研究权重做固定循环数扫描，分别对齐参数数目和推理算力，比长程推理质量、TTFO 和任务耗时。 [Universal Transformer](https://arxiv.org/abs/1807.03819) · [Huginn](https://arxiv.org/abs/2502.05171) · [Ouro](https://arxiv.org/abs/2510.25741) |
| W47 Inkling 2 Mini 与可变循环研究版 | 固定版的 12 层共享 core 运行三遍，每遍独立 KV 与卷积状态，DSpark 读取固定的多遍 hidden-state taps。可变循环版的共享 KV 是后遍覆盖写，卷积状态仍按遍分开，与 GLM 的双流不同。 | vLLM 工程文的单 GB200、TP1、4K 仅前向例中，研究版三遍全独立与三遍共享的 KV 容量为 175K 和 194K token，batch 1 decode step 约 3.85 和 3.86 ms。测共享、降低深度两条不同的容量／质量曲线。 [Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/) · [可变版权重](https://huggingface.co/thinkingmachines/TML-d24-varloop-research) · [vLLM 实现与对照](https://vllm.ai/blog/2026-11-20-inkling2-mini) |
| Recurrent KV Replay、LooM | 前者训练时构造只在指定迭代写 KV 的布局。LooM 在 causal encoder-decoder 中让全局内存每 token 写一次，并让重复的 decoder 层只在首次访问时发布 KV。 | 对照每遍独立、训练时共享与共享历史加局部状态。压测时对齐长上下文检索质量和额外存储，检查同一 token 在不同迭代的可恢复性。 [KV Replay v1](https://arxiv.org/abs/2611.19446v1) · [LooM v1](https://arxiv.org/abs/2611.06835v1) |
| LLA、LoopKV、LoopVQ | LLA 用跨循环低秩表示并在读时重构；LoopKV 按循环与层选近似共用或量化；LoopVQ 对多个循环联合向量量化。 | 从独立 KV 基线开始，固定最大 HBM 对照可驻留并发、每步读取、解码端解包和任务成功率。小模型原型结果适合作设计依据。 [LLA v1](https://arxiv.org/abs/2607.15456v1) · [LoopKV v1](https://arxiv.org/abs/2611.04137v1) · [LoopVQ v1](https://arxiv.org/abs/2611.05304v1) |
| LoopDuet、LoopFold、W47 LoopCache | LoopDuet 分配 prefill 与 decode 深度，LoopFold 学习将多个递归深度并入平行批，LoopCache 在循环模型内自投机，用循环 KV 排程减少长上下文的读取放大。 | 先测分阶段深度。需要准确保持目标模型时，用浅遍提案、完整深度验证，分别测草稿依赖的隐藏层、状态提交和真正的 KV 读流量。 [LoopDuet v1](https://arxiv.org/abs/2611.07211v1) · [LoopFold v1](https://arxiv.org/abs/2610.35556v1) · [LoopCache v1](https://arxiv.org/abs/2611.24301v1) |

**先做一个可归因的实验。** 用 Inkling 可变循环研究版比较单遍、两遍、三遍和三遍共享，先关闭投机并走固定版本。以 4K／32K／业务长上下文，覆盖低并发、饱和以及热前缀。记录每遍和每种缓存的峰值、每轮读取、可分配 token、TTFT、TPOT、首个有效输出与成功任务成本。第二轮在固定循环权重上用支持的 DSpark 路径，与逐 token 基线做对照。GLM 式“第一遍全局／第二遍局部”需要匹配架构和训练，等官方配置或有等价语义的可用权重再验证。 [vLLM 实作](https://vllm.ai/blog/2026-11-20-inkling2-mini)

### SpecPilot 在 910B4、vLLM-Ascend v0.32.0rc1 的可尝试方向

**证据口径。** 公共 SpecPilot 实验使用 SGLang 和单卡 H100 或 RTX PRO 6000，包含 greedy 与 thinking。其项目页中 Qwen3.8-27B DSpark 的 H100 单请求 3.74 → 9.41×、FP8 批处理 1.94 → 3.93×，都是开发集相对于 AR 的整套结果；比值约 2.52 和 2.03。项目另外用未见样本给出相对 stock 的封存对照 2.38× 和 1.94×。迁移所指是机制，下面的 910B4 加速均为 `pending_remote`。 [SpecPilot 项目结果](https://stargazerx0.github.io/specpilot-project-page/) · [论文 v1](https://arxiv.org/html/2611.02501v1) · [实验代码](https://github.com/StargazerX0/SpecPilot)

| 优先级 | 动作 | 最小对照与通过条件 |
| --- | --- | --- |
| P0 · 先 profile | 冻结 Qwen3.8-27B target 与 DSpark checkpoint、权重精度、TP、runner、CANN 和图模式。对照普通 AR 与 stock DSpark，分开测 draft backbone、base LM head、串行 Markov、元数据、verify 和状态 commit。 | 用设备 trace 确认真实路径和实际 padded shape。v0.32.0rc1 的 MRV1 DSpark proposer 显式关闭自身的图执行，目标侧启图和草稿侧仍需分别确认。 [MRV1 proposer](https://github.com/vllm-project/vllm-ascend/blob/v0.32.0rc1/vllm_ascend/worker/v1/spec_decode/dspark_proposer.py) |
| P0 · 语义等价的元数据和搬运 | 检查 slot／page 映射、位置与长度物化、D2H 同步、logits/state copy 和验证结果提交。已经融合的输入构造与主机长度镜像沿用。对余下同 shape 的重复构造先做 buffer 复用，再判断是否融合。 | 用相同 token、相同分支和相同 slot 对照，记录 launch、同步、字节和峰值工作区，加入抢占、并发异长和全部／部分拒绝。纯布局与搬运改动适用严格 token 对照。 [SpecPilot 元数据实现](https://github.com/StargazerX0/SpecPilot/blob/main/settings/h100_batch_qwen38-27b-fp8_dspark/files/python/sglang/srt/speculative/specpilot_d4r3_host.py) · [Ascend proposer](https://github.com/vllm-project/vllm-ascend/blob/v0.32.0rc1/vllm_ascend/worker/v1/spec_decode/dspark_proposer.py) |
| P0 · 候选集 Markov 打分 | 同 tag 的 MRV1 先算 base logits，沿草稿位置串行加前驱 token 的 Markov bias。对照完整 bias、base top-k 和 `base top-k ∪ 前驱条件候选` 三条草稿，保留一条链及原目标验证。上游有 gathered bias 接口，其具体 kernel 需按 NPU 构建实现。 | 先用 greedy，按 K 测候选构建、召回完整草稿 top-1、接受长度与每提交 token 耗时。仅在完整投影后 gather 不节省 base LM head GEMM。缩投影或采样时另验词表映射、实际 proposal 概率和目标采样分布。 [Ascend MRV1 路径](https://github.com/vllm-project/vllm-ascend/blob/v0.32.0rc1/vllm_ascend/worker/v1/spec_decode/llm_base_proposer.py) · [vLLM Qwen3 DSpark](https://github.com/vllm-project/vllm/blob/v0.32.0rc1/vllm/model_executor/models/qwen3_dspark.py) · [SpecPilot 候选实现](https://github.com/StargazerX0/SpecPilot/blob/main/settings/h100_batch_qwen38-27b-fp8_dspark/files/python/sglang/srt/speculative/specpilot_d1_ld.py) |
| P0 · 实测共享验证预算 | 复用已存在的 MRV1 `DynamicSpecScheduler`，按累计接受概率与实测验证成本分配批内预算。MRV2 `enable_adaptive_verification` 是另一条 runner 路径，按 tag 文档确认 confidence head 和图约束。 | 扫请求数、总验证行、GDN／attention、TP 及执行模式，记录逻辑行、实际执行行、padding 和 `.item()` 同步。只在实测成本平台附近调整预算。H100 配置中的 192／256 属作者选点。 [MRV1 scheduler](https://github.com/vllm-project/vllm-ascend/blob/v0.32.0rc1/vllm_ascend/worker/v1/spec_decode/utils.py) · [tag 文档](https://github.com/vllm-project/vllm-ascend/blob/v0.32.0rc1/docs/source/user_guide/feature_guide/speculative_decoding.md) · [SpecPilot §4.4](https://arxiv.org/html/2611.02501v1#S4.SS4) |
| P1 · 真正缩短草稿 | 先比较固定长度预生成后截断验证、按请求减少 drafter 的实际执行位置。后一条需要 ragged packing 或可覆盖的形状分桶，处理位置、采样 index 和 KV slot。混合流量中若出现验证批空余，再判断拼入短 prompt chunk 是否优于现有 chunked prefill。 | 只扫 checkpoint 支持的长度，记录实际位置与 GEMM 形状，确认 draft 与 verify 两端都节省。独立对照混合 prefill 的 TTFT、decode P99 ITL 与低优请求完成率。 [SpecPilot 请求级 draft 代码](https://github.com/StargazerX0/SpecPilot/blob/main/settings/h100_batch_qwen38-27b-fp8_dspark/files/python/sglang/srt/speculative/specpilot_d8r3_rblock.py) · [SpecPilot §4.4](https://arxiv.org/html/2611.02501v1#S4.SS4) |
| P2 · 树验证、目标 head、W12、TLV | 树验证处理父 token 条件打分、GDN／卷积分支隔离和接受路径提交。Certified LM head 用保守界筛掉无望成为 argmax 的行，对剩余行用原精度重算并保留全量回退。W12 把 BF16 压缩并在 GEMM 内恢复，TLV 增加一次便宜的中间检查及再次起草。 | 仅在低并发且查清主瓶颈后逐项做。论文的 certified head 在 H100 较快、在所测 RTX PRO 6000 较慢；作者的 BF16 批量 TLV 比对照慢约 13%。tag 文档中 FIA 路径的 `num_speculative_tokens + 1 ≤ 16` 要单独验证，较宽的树不能照搬参数。 [论文 §4](https://arxiv.org/html/2611.02501v1#S4) · [作者反例](https://stargazerx0.github.io/specpilot-project-page/) · [Ascend tag 文档](https://github.com/vllm-project/vllm-ascend/blob/v0.32.0rc1/docs/source/user_guide/feature_guide/speculative_decoding.md) |

**首轮交付与退出条件。** 同权重、同 prompt token IDs 和停止规则，分别扫并发 1、低并发与接近饱和的点。先按 trace 选 Markov、元数据或预算中最大的一项，单变量优化后检查每提交 token 的耗时、P95 TPOT／ITL、达标吞吐、实际 verify 行和接受长度分布。保存首次 token 分歧及 margin，分开比较 stock DSpark 和目标逐 token 基线。SpecPilot 的 teacher-forced 检查以共同前缀比较逐位置偏差，并用 stock 自身的漂移校准；对要求严格一致的改动另做自由生成全序列检查。采样态还需验证目标分布，候选裁剪不应继续用原 proposal 概率。若减少逻辑行后设备仍执行原规模，回到图形状和分派阶段。 [SpecPilot §3.3](https://arxiv.org/html/2611.02501v1#S3.SS3) · [W47 Ascend 自适应验证 #21137](https://github.com/vllm-project/vllm-ascend/pull/21137)

### 扩散模型与投机解码的三条路线

这组工作中的验证对象有三种：独立并行草稿交给 AR target 验证，共享 AR 模型内部起草和验证，以及直接加速扩散 target 的去噪轨迹。跑分应分别和同一目标权重、同一采样语义下的强基线对照。

| 工作 | 机制与公开状态 | 值得试的方向 |
| --- | --- | --- |
| **Uno** · `2609.04010v1`，09-03 | AR 权重维持原模型分布，加入经 diffusion distillation 训练的轻量扩散参数，一遍并行起草；Ψ-Spec 由 AR 权重验证，公开实现支持线性与树采样。作者报所测中最高约 3× 原 AR 模型，代码和权重可用，推理由 `nano_vllm_uno` 样例引擎支撑。 | 先跑 Uno-Qwen3-8B 对相同底座 AR，再跑外置 DFlash／PARD-2 作为工程对照。确认共享权重、KV 布局、草稿驻留开销和高 batch 拐点。 [Uno v1](https://arxiv.org/abs/2609.04010v1) · [作者解释](https://kempnerinstitute.harvard.edu/research/deeper-learning/unlocking-lossless-speedups-in-llms-via-discrete-diffusion/) · [代码](https://github.com/ifm-ai/uno) |
| **PARD-2** · `2605.08632`，05-09 首投，06-10 作者发布代码和权重 | 官方拼写为 **PARD-2**。这是 target-aligned 并行 drafter，以 CAT 权重调整训练目标以改善连续接受长度，通过随机注入 target 特征支持依赖和独立两种模式。论文报告最高 6.94× 的测试点。W47 SGLang #51465 为 HIP 集成开放 PR，复用 DFlash verify/accept 路径，不能从作者代码发布推断此 PR 已正式发行。 | 对同一可用 checkpoint 分别试两种模式，比较 target 特征传输、草稿时间、连续接受和吞吐。优先用发布的 Qwen3-8B drafter 做单机试验，910B4 需要另做后端兼容性核验。 [PARD-2 论文](https://arxiv.org/abs/2605.08632) · [作者代码及日期](https://github.com/AMD-AGI/PARD) · [Qwen3-8B 权重](https://huggingface.co/amd/PARD2-Qwen3-8B) · [SGLang #51465](https://github.com/sgl-project/sglang/pull/51465) |
| **DFlash／DFlash2／DSpark** | DFlash 用 target 层特征做一次块扩散起草；DFlash2 加组内动态因果卷积和可选候选选择器；DSpark 在并行骨干后引入局部的串行 Markov 依赖与置信度。三者需按目标 checkpoint 和后端支持配对。 | 固定 target 与排序／停止规则比较块长度、候选质量、草稿开销和同等 SLO 下的输出 tok/s。记录预测步骤内的串行部分及其占比。 [DFlash v1](https://arxiv.org/abs/2602.06036v1) · [DFlash2 工程文档](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash2/) · [DSpark v1](https://arxiv.org/abs/2607.05147v1) |
| **DDTree、DCOCO、Parallel Reconsideration** | DDTree 从块扩散概率构造候选树。DCOCO 用上下文修正并行块，作者报告并发 4–32 对最强比较项有 6%–23% 的提升。Parallel Reconsideration 给不确定位置并行二次猜测，作者报告平均接受长度与解码速度的增量。 | 分开做多树分支、一次修正两条对照。按相同 verify 行预算或等 p95 时延测接受与状态提交，判断是继续花预算在 draft，还是在 target 多验证节点。 [DDTree](https://arxiv.org/abs/2604.12989) · [DCOCO](https://arxiv.org/abs/2610.12570) · [Parallel Reconsideration v1](https://arxiv.org/abs/2611.05724v1) |
| **D3D、DNeo** | D3D 减少草稿对刚产生的 target hidden state 的依赖，验证当前块时并行准备可能的后继，目标是重叠两段执行。W47 提交的 DNeo 用 MoFA、历史拒绝信息和非均匀特征 tap 改善并行 drafter 编码，论文相对 DFlash2 报约 1.3×。 | 先记录 verify 后等待 target 特征和 draft 前向的空档。并行起草时同时记无效分支工作、竞争带宽以及接受收益，避免只统计重叠时间。 [D3D](https://arxiv.org/abs/2611.06425) · [DNeo v1](https://arxiv.org/abs/2611.21925v1) |
| **LibraSpec、HybridSpec、NarrowSpec** | LibraSpec 用置信度与边际验证成本决定是否多验证一位；HybridSpec 针对 GDN-MoE 的验证成本与状态恢复；NarrowSpec 减小草稿残差维度并重新适配 draft，在 B200 和 GB10 效果不同。 | 优先以现有配置做成本驱动验证长度，只有 draft 实测占比高再尝试缩宽和重新训练。验证预算按模型、shape、图和精度标定。 [LibraSpec](https://arxiv.org/abs/2608.08721) · [HybridSpec](https://arxiv.org/abs/2611.04055) · [NarrowSpec v1](https://arxiv.org/abs/2611.23835v1) |
| **以 dLLM 为目标的 SimSD、SPOT、Trajectory-Level SD、PV-Decoding** | dLLM 要验证的是掩码更新的**位置、token 与顺序**。SimSD 用掩码构造合法上下文，SPOT 学预测 commit order 后让 target 验证状态，Trajectory-Level SD 验证整段去噪轨迹，PV-Decoding 测 progressive verification 与 gate。SPOT 作者所测最高 3.2× 标准解码；PV-Decoding 的多 batch 实现存在退化。 | 先跑按指定提交次序逐步重放的基线，再测目标 pass 数、正确状态相等、每步提交 token、输出质量和并发曲线。若有近似步骤，另报其任务质量与失败。同一模型的 block continuous batching 可做强服务对照。 [SimSD](https://arxiv.org/abs/2606.02544) · [SPOT](https://arxiv.org/abs/2610.22543) · [Trajectory-Level](https://arxiv.org/abs/2608.27514) · [PV-Decoding](https://arxiv.org/abs/2610.24722) · [BlockServe](https://arxiv.org/abs/2607.08930) |
| **图像／视频 DiT 的近邻路线** | SpeCa 借鉴提案和验证框架预测跨 denoising step 的特征，再判定是否回退到完整计算。论文以 FLUX、DiT 和 HunyuanVideo 等质量和速度指标衡量。 | 有图像／视频业务才开实验。固定分辨率、步数／采样器和 seeds 对比近似特征缓存、质量阈值和每请求 GPU 秒。 [SpeCa](https://arxiv.org/abs/2509.11628) |

**立项顺序。** 现有 910B4 有 DSpark 时先做不换权重的三项 profile 驱动优化。另有 8B GPU 环境时，用同一 Qwen3-8B 目标跑普通 AR、DFlash、PARD-2 及 Uno。把相同 harness 的对照与作者原版复现分开，注明引擎、精度、停止、温度和并发，报告平均接受长度、draft/verify 行数、语法正确性、尾延迟和任务成功率。要探索循环模型与投机的交点，可先复现已经公开的 Inkling + DSpark，或用 LoopCache 检验浅层提案与完整循环验证的投入产出。此处硬件与性能实验均为 `pending_remote`。 [Inkling 引擎支持](https://vllm.ai/blog/2026-11-20-inkling2-mini) · [LoopCache v1](https://arxiv.org/abs/2611.24301v1)

## 可复现的工程验证矩阵

下表为由上游结果推导的实验设计。性能、精度和 GPU/NPU 联调均为 `pending_remote`。每条运行记录保存容器或构建 SHA、依赖及驱动、权重与 tokenizer revision、硬件型号和拓扑、运行参数、原始负载、样本数与失败数。冷启动和热态分别采样。

| 要验证的假设 | 对照与压力档 | 指标与验收 |
| --- | --- | --- |
| 默认 KV 扩容可以提高容量且保持输出连续 | 引擎升级前后各测短请求、热前缀、多分支和混合长度请求。加入长 prompt、SWA 对齐边界、中途取消、驱逐／抢占、跨 P/D 重连。 | 首先检查输入 token ID、路由后的 model/draft 版本、EOS／grammar 状态、每轮输出文本和逐 token logprob。随后测可驻留请求、有效缓存 token、内存余量、队列 P95/P99、TTFT。 |
| 投机验证能在真实流量下改善 TPOT | AR、固定预算与动态预算，分别在 eager／graph 下按低并发至饱和、短／长输出、温度 0／业务采样做对照。控制相同 target、draft 与输出停止条件。 | 先检查 greedy token 一致性，随机采样检查分布和结构化约束。分开报草稿接受长度、verify 耗时、每步输入行、TPOT、ITL 尾部、端到端 goodput。 |
| 稀疏索引与缓存卸载能释放 HBM | 无卸载、KV-only、KV＋indexer，分别对比 host／SSD／跨机目标，覆盖高命中、低命中、突发、tool idle 与不同步长。 | GPU/host 常驻、PCIe／网络／SSD 字节、读放大、排队、失败回退、重算 token、P99 TTFT。用同一质量集检查稀疏召回或量化误差。 |
| 小 M GEMM、稀疏 attention 和量化通信能产生端到端收益 | 固定算法和 token 数，扫 prefill／decode 形状、消息尺寸、EP 热点、CPU 提交和 dtype。每次确认实际命中后端，记录 fallback。 | kernel 时延、CPU launch、HBM 和互联占用、MoE rank 负载、端到端 TTFT/TPOT 与输出误差分开。性能门槛用实测负载设定。 |
| 多租户与跨节点路由可以提高达标产出 | 均衡、前缀感知和队列感知，加入高优租户突发、热点副本、连接器故障、跨区容量不足与恢复。 | 定义 goodput 口径为同时满足 TTFT/TPOT 约束的成功请求或 token 每秒，并明确统计单位。记录拒绝、重试、实际目的地、冷命中惩罚和各租户尾延迟。 |
| rollout 扩大后仍受控 | 固定任务、质量目标和权重同步频率，改变 train/serve 资源分配、最大策略滞后和 sandbox 并发。 | time-to-quality、训练端 MFU、rollout 输出 tok/s、每任务成功率、权重字节和同步时长、sandbox 等待与长轨迹截断率。 |

## 信源、日期和剩余盲区

| 信源组 | 本报告使用的覆盖与边界 |
| --- | --- |
| 核心采集 | 复用 `origin/data` 快照 `d39ac3a98748a0be021ca5cddeb8edbdeecfd977` 的历史事实。arXiv、HF 日榜与三大引擎 API 在 11-20 15:52 的快照均记录成功。周末工程补查按 GitHub Search 的合并日分片并分页，页面记录包含回移植和维护项。入选工作再按实际 PR API 的目标分支与 `merged_at` 核对。 |
| 无配置或关闭的工程渠道 | 对清单内 32 个来源编号对应的 42 仓库走官方 Releases 与默认分支 REST。统计是源记录数，关联到同一项目时按 PR、tag、SHA 各自保留阶段。滚动构建、平台包和候选版均会贡献记录。每仓 release 首屏范围与默认分支视野存在限制。 [GitHub REST Releases 文档](https://docs.github.com/en/rest/releases/releases) · [GitHub REST Commits 文档](https://docs.github.com/en/rest/commits/commits) |
| 论文与会议 | 通过 arXiv 版本、公告和全文，以及 HF 入口复核入选论文。SC26 用官方排期和作者摘要补回会议。复查 MLSys、OSDI、SOSP 和 EuroSys 的公开目录作为历史索引。各会作者 PDF 的评测配置和部署状态随可取得材料深度而异。 [SC26](https://sc26.supercomputing.org/Program/) · [MLSys](https://proceedings.mlsys.org/) · [OSDI](https://www.usenix.org/conference/osdi26/technical-sessions) · [SOSP](https://www.sigops.org/s/conferences/sosp/2026/accepted.html) · [EuroSys](https://2026.eurosys.org/papers.html) |
| 研究机构、厂商、服务商和基准 | 人工按官方 model card、博文、提交页和公开文档交叉核对。模型发布、权重可获取、框架主干支持与正式发行均按各自时间记录。先前 67 项逐源覆盖包含 Academic API、机构站和 RSS 的检查。 [信源清单](../../config/sources.yaml) |
| 中文媒体与参考项目 | 本地 11-23 机器之心 RSS 试跑存档读取 10 条、保留 9 条，保留内容的发布时间落在 W48。经 TensorLean 线索追回 W47 一手公告。量子位通过站点索引补查。额外比对 GitHub 上同类雷达的 arXiv API/OAI-PMH、RSS 和 Releases 路线，作为信源发现参考。一手证据位于各正文条目。 [量子位线索](https://www.qbitai.com/2026/11/511679.html) · [llm-paper-radar](https://github.com/zhaolin-amd/llm-paper-radar) · [agents-radar](https://github.com/sikm-lqs/agents-radar) |

有时间戳的事件按北京时间归周。只标日的博客、模型公告保留发布方日历日期。arXiv 以明确 ID 和版本去重，另列首次提交、修订或公告越过周界的记录。PR 以仓库和编号追溯，正式发行以 tag 和 `published_at` 追溯。档案能提供首次观察时间时沿用历史记录，通过本轮补查取得的证据统一记为 11-23 查阅。数据分支快照、REST 原始记录、性能实验、集成状态的口径各自独立。

本次直接 SSH 与默认沙箱内的 GitHub CLI 遇到连接限制，远端报告使用 GitHub 连接器确认 blob，公共工程补查通过可用的官方 REST 路径完成。未配置站点靠公开页面和索引做定向查询，登录、付费、限流、接口分页和更新频率仍会造成遗漏。可见候选数与入选数量都作为实际查阅范围。整份报告中的上游数字用于形成复现顺序，保留了条件和验证阶段。
