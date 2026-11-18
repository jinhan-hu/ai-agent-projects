# 多智能体路由：一份可运行的最小实现

> 这是「从零实现 Agent」系列的第四篇。项目地址：[05-multi-agent-router](https://github.com/jinhan-hu/ai-agent-projects/tree/main/05-multi-agent-router)

## 为什么要再写一个「多智能体」项目

过去一年里，"多智能体"这个词被用得很泛。各种框架给的示例通常是：三个角色互相聊天，然后神奇地产出结果。

**这类 demo 能跑，但回答不了一个工程问题：**

> **多智能体系统里，哪些判断该交给模型，哪些必须写死？**

这个问题的答案决定了系统能不能上生产。**所以我写了一个不带框架、零依赖、52 个测试的最小实现，把这个问题拆开讲。**

---

## 它做什么

一张工单进来，系统判定类型，交给对应的专家处理：

| 类型 | 专家 |
| --- | --- |
| `install` | 安装部署 |
| `usage` | 功能使用 |
| `troubleshoot` | 故障排查 |
| `fallback` | 通用兜底 |

编排图：

```
load_memory ──► route ──(条件边)──┬─► expert_install ──────┐
                                  ├─► expert_usage ────────┤
                                  ├─► expert_troubleshoot ─┤
                                  └─► expert_fallback ─────┤
                                                           ▼
                                                       verify ──► remember ──► END
```

**五个节点，其中只有两个用 LLM。**

---

## 取舍一：只有 `route` 一处让模型做决策

先看整张图的职责划分：

| 节点 | 职责 | 用 LLM？ |
| --- | --- | :---: |
| `load_memory` | 取客户档案 + 会话历史 | ❌ |
| **`route`** | ★ **判定类别 + 置信度** | ✅ |
| `expert_*` | 检索 → 生成回答 | ✅（生成） |
| `verify` | 检查引用/澄清/异常 | ❌ |
| `remember` | 写入两级记忆 | ❌ |

**为什么这样分？**

> **把「判断」交给模型：能处理没见过的说法、灵活。**
> **把「流程」写死：可控、可测、可审计。**

**纯自主 Agent 不可控**（不知道下一步要干嘛、贵、难调试）；
**纯规则系统不灵活**（用户换个说法就失效）。

**两者之间的这个点，才是生产可用的。**

---

## 取舍二：置信度低于阈值，就不硬分派

```python
CONFIDENCE_THRESHOLD = 0.7

# 低置信度不硬分派
if confidence < CONFIDENCE_THRESHOLD and route != ROUTE_FALLBACK:
    return {
        "route": ROUTE_FALLBACK,
        "route_confidence": confidence,
        "route_reason": f"置信度 {confidence:.2f} 低于阈值，转兜底。原判：{reason}",
        "need_clarify": True,
    }
```

**为什么？**

> **分错类的代价，比承认"我不确定"的代价大得多。**

分错类 → 专家拿到不相关的知识库片段 → 生成一个看起来合理但完全不对的答案 → 用户被误导。

**而转兜底 → 向用户索取补充信息 → 下一次能分对。**

**这里有个反直觉的点**：`fallback` 分支不是"降级"，是**主动承认信息不足**。在可靠性上，这比硬猜好。

---

## 取舍三：`verify` 是独立节点，不塞进专家里

**最容易被写进专家内部的一步是"检查回答质量"。** 我刻意把它拆出来：

```python
def _verify(self, state: TicketState) -> dict:
    problems = []
    if state.error:
        problems.append(f"上游异常：{state.error}")
    if not state.answer.strip():
        problems.append("专家没有产出回答")

    # 兜底专家允许没有引用（它本来就没检索）
    if state.route != "fallback":
        if len(state.citations) < self.config.verify_min_citations:
            problems.append("回答没有标注引用来源，可能是编造")

    # 需要澄清的工单：回答里必须真的向用户提问
    if state.need_clarify:
        asked = any(k in state.answer for k in ("？", "?", "请", "补充", "确认"))
        if not asked:
            problems.append("路由判定信息不足，但回答没有向用户索取补充信息")

    verified = not problems
    return {"verified": verified, "verify_note": "通过" if verified else "；".join(problems)}
```

**为什么不塞进专家？**

> **生成和检查是两种不同的能力。**
> **混在一起时，检查会被"我刚才生成的东西看起来没问题"这种惯性带偏。**

**这个判断在开发过程中被验证过一次**：

我最初让 `_extract_citations` 在模型没写引用时**回落到"检索到的文档"**，看起来更宽容。结果测试挂了两条——因为它把"没标注引用"这个信号掩盖了，`verify` 就失效了。

**改成只从回答正文里抽引用之后，校验才真正起作用。**

---

## 为什么不用 LangGraph

这是个绕不开的问题。我的回答是：**我手写了一个 87 行的状态图引擎，因为想看清编排的本质。**

**LangGraph 的核心其实三样东西：**

1. **节点** = 一个 `(state) -> update` 的函数
2. **边** = 固定跳转或条件跳转
3. **一个带步数上限的循环**

**对应关系：**

| LangGraph | 本项目 |
| --- | --- |
| `StateGraph` | `Graph` |
| `add_node` / `add_edge` | `add_node` / `add_edge` |
| `add_conditional_edges` | `add_conditional` |
| `compile()` | 无需编译 |
| `invoke()` | `run(state, max_steps=20)` |

## ★ 步数上限是必需的

```python
for _ in range(max_steps):
    started = time.perf_counter()
    update = self._nodes[current](state) or {}
    state.apply(update)
    state.record(current, started, sorted(update.keys()))

    nxt = self._next(current, state)
    if nxt is None or nxt == self._end:
        return state
    current = nxt

state.error = state.error or f"达到步数上限 {max_steps}，可能是条件边成环"
return state
```

**条件边可以形成环。没有上限就会挂死。**

所以测试里专门覆盖了这一条：

```python
g3 = Graph()
g3.add_node("loop", lambda s: {})
g3.set_entry("loop")
g3.add_edge("loop", "loop")
s3 = g3.run(TicketState(ticket=Ticket("t", "x")), max_steps=5)
check("成环时被步数上限截断", "步数上限" in s3.error)
```

---

## 顺手解决的一个问题：状态字段拼错会静默失败

`State` 如果用 `dict`，节点返回 `{"answr": "..."}` 这种拼写错误**不会报错**——它只是塞进一个没人读的 key。

**我用 dataclass + 显式校验：**

```python
def apply(self, update: dict[str, Any]) -> None:
    for k, v in update.items():
        if not hasattr(self, k):
            raise AttributeError(f"状态里没有字段 {k!r}，检查节点返回值")
        setattr(self, k, v)
```

**多写这 4 行，省掉的是"为什么这个字段一直是空"的半小时。**

---

## 检索：为什么是简化版

真实的生产 RAG 用「向量 + BM25 混合检索 + Rerank」。本项目用的是**字符二元组 + IDF 加权**。

**这不是偷懒，是明确的取舍：**

1. **零依赖** —— clone 下来零配置能跑，测试不需要向量库
2. **结果可解释** —— 每一分都能说清是哪几个 bigram 贡献的，便于写断言

**升级路径很明确**：把 `KnowledgeBase.search()` 换成混合检索即可，**上层专家 Agent 的接口不需要改**。

## ★ 一个容易被忽略的收益：路由先于检索

```python
def search(self, query, k=3, route=None):
    pool = [d for d in self.docs if route is None or d.route == route]
```

**`route` 参数让检索空间从"全库 N 篇"缩到"该类的几篇"。**

**这就是路由的第一层收益**——不只是"交给对的专家"，还是**"缩小检索范围"**。

**这也是为什么"路由 + 每个专家独立知识库"比"一个 Agent 全库检索"更准。**

---

## 两级记忆：为什么不能塞进一个容器

| 层 | 生命周期 | 用途 |
| --- | --- | --- |
| **会话记忆** | 随 `session_id` | 多轮对话上下文 |
| **客户档案** | 跨会话（按 `customer`） | 「这个客户上次问过同样的问题」 |

**塞进同一个容器的后果是二选一：**

- 短期记忆被长期污染（上一轮的话混进长期档案）
- 长期记忆被会话清空（一关会话，档案没了）

**持久化留了 `Store` 协议，默认内存实现。接 SQLite / Redis 只是换一个实现，上层不动。**

---

## 一个诚实的设计局限

**这个项目目前的检索是「专家内部的固定步骤」，不是「模型自主调用的工具」。**

也就是说：**专家会检索，但它不"决定"要不要检索。**

这是刻意的边界——**这个项目的主题是编排，不是工具调用**。工具化（`run_tool_loop`）和 MCP 接入是明确的下一步扩展点，README 的「扩展点」一节写了改哪里。

**我觉得把边界写清楚，比假装它已经完备要好。**

---

## 测试覆盖什么

**52 个测试，全部离线**（`MockLLM`，不需要 API Key）。重点在**编排逻辑**，不是模型质量：

| 组 | 覆盖 |
| --- | --- |
| 图引擎 | 固定边、条件边、成环截断、未知节点报错、未知状态字段报错 |
| 路由 | 正常分派、低置信度兜底、非法 JSON 降级、未知类别降级、置信度夹紧、围栏/夹带文字解析 |
| 检索 | 分类约束、排序正确、空查询 |
| 记忆 | 长期档案、会话隔离、上限截断 |
| 端到端 | 节点顺序、引用抽取、校验失败场景、多轮历史、乱答不崩溃 |

```bash
$ python tests/run_all.py
通过 52 / 52
```

**为什么坚持离线测试？**

因为要验证的是「**低置信度会不会兜底**」「**模型输出非法 JSON 会不会崩**」这类**编排分支**——它们必须能确定性复现。挂上真实 API 之后，这些测试会变成"网络抖一下就红"。

---

## 三个可以带走的结论

1. **多智能体系统的设计核心不是"几个 Agent"，是"哪里用模型、哪里用代码"**
2. **低置信度兜底不是降级，是可靠性设计**——承认不确定，比硬猜好
3. **生成和检查必须分开**——否则检查会被生成带偏

---

**相关代码**：[05-multi-agent-router](https://github.com/jinhan-hu/ai-agent-projects/tree/main/05-multi-agent-router)

**系列其他文章**：

- [Agent 为什么停不下来？——从零实现 Agent Loop 的终止设计](./01-agent-loop-termination.md)
- [RAG 混合检索：为什么纯向量不够](./02-hybrid-retrieval.md)
- [给 Boss 装一个 LLM：延迟、公平性与成本的三角](./03-llm-boss-design.md)
