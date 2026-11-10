# Agent 为什么停不下来？——从零实现 Agent Loop 的终止设计

> 这是「从零实现 Agent」系列的第一篇。项目地址：[mini-agent-harness](https://github.com/jinhan-hu/ai-agent-projects/tree/main/04-agent-harness)

## 一个看起来没问题的循环

如果你搜「怎么实现一个 Agent」，大概率会看到这段代码：

```python
while True:
    resp = llm.chat(messages, tools)
    if not resp.tool_calls:
        return resp.content
    for call in resp.tool_calls:
        messages.append(run_tool(call).to_message())
```

逻辑很清晰：**模型不请求工具调用，就说明它想好了，退出。**

**这段代码能跑通 demo，但上不了生产环境。**

## 它会怎么卡住

我在自己实现时就遇到了三种情况：

### ① 模型反复调用同一个工具

```
模型：我要调 calculator 算 120+80
工具：600
模型：我要调 calculator 算 120+80     ← 又来一次
工具：600
模型：我要调 calculator 算 120+80     ← 还来
...
```

**模型"觉得"自己没算对，于是重试。** 参数完全没变，结果也不会变，但它就是不停。

### ② 工具持续失败，模型持续重试

某个工具的参数格式要求很严，模型每次都差一点，每次都报错，每次都改一点再试。**这个过程可以无限长。**

### ③ 工具输出被截断，模型以为没拿到

长输出被截断后，模型看到的是不完整的结果，于是**再调一次**——但它不知道截断规则，所以每次都拿到同样被截断的内容。

**这三种情况的共同点是：循环本身没有"该停"的信号。**

---

## 三个出口

我给循环设了三个必须满足的终止条件：

| 出口 | 触发条件 | 处理方式 |
| --- | --- | --- |
| **① 正常收敛** | 模型不再请求工具调用 | 直接返回答案 |
| **② 步数上限** | `step >= max_steps` | 软着陆 |
| **③ 循环检测** | 同一组调用重复超过阈值 | 软着陆，标记原因 |

前两个很容易理解，**第三个需要解释一下"什么叫同一组调用"**。

### 循环检测：给工具调用算个签名

```python
def _signature(calls: list[ToolCall]) -> tuple:
    """把一组工具调用压成可比较的签名。"""
    return tuple(
        sorted((c.name, tuple(sorted(c.arguments.items()))) for c in calls)
    )
```

**为什么用 `sorted`**：`{"a": 1, "b": 2}` 和 `{"b": 2, "a": 1}` 是同一个调用，排序后才能正确判等。

然后维护一个计数器：

```python
seen[sig] += 1
if seen[sig] > self.max_repeat:
    # 同一组调用重复太多次 → 判定为卡住
```

**阈值取 2（即重复第 3 次触发）**——一次重复可能是模型在确认，两次以上基本就是卡住了。

---

## 关键取舍：硬超时 vs 软着陆

**发现问题很容易，怎么"停"才是设计点。**

### 最直觉的做法：直接终止

```python
raise TimeoutError("Agent 步数超限")
```

**问题**：用户拿到的是一个**异常**，不是答案。

**而这个 Agent 可能已经调用了一堆工具、收集了大量信息——这些工作全部白费。**

### 我的做法：软着陆（soft landing）

```python
def _soft_landing(self, messages: list[Message], *, hint: str) -> str:
    """不带 tools 再问一次，让模型基于已有信息总结。"""
    resp = self.llm.chat(
        list(messages) + [Message(role="user", content=hint)],
        [],          # ★ 关键：tools 传空列表
    )
    return resp.content
```

**核心就是那个 `[]`。**

**不给工具，模型就只能做一件事：说话。**

它手里已经有前面所有工具调用的结果，所以它能给出**基于事实的部分答案**，而不是一个异常。

### 两种终止的对比

| | 硬超时 | 软着陆 |
| --- | --- | --- |
| 用户拿到 | 异常 / 空结果 | 部分答案 |
| 已收集的信息 | 浪费 | 用上了 |
| 成本 | 省一次调用 | 多一次调用 |
| 可调试性 | 只有堆栈 | 有 `stop_reason` + 完整 trace |

**多一次调用的成本，换用户拿到可用结果——这笔账是划算的。**

---

## 完整的主循环

把三个出口放进去，就得到（简化版）：

```python
def run(self, task: str) -> Trace:
    messages = [
        Message(role="system", content=self.system_prompt),
        Message(role="user", content=task),
    ]
    trace = Trace(task=task)
    seen = Counter()

    for step_idx in range(self.max_steps):
        # ① 调模型
        resp = self.llm.chat(messages, self.registry.to_openai_schema())
        messages.append(resp)

        # ② 出口：正常收敛
        if not resp.wants_tools:
            trace.final_answer = resp.content
            trace.stop_reason = "converged"
            return trace

        # ③ 出口：循环检测
        sig = _signature(resp.tool_calls)
        seen[sig] += 1
        if seen[sig] > self.max_repeat:
            trace.stop_reason = "loop_detected"
            trace.final_answer = self._soft_landing(
                messages,
                hint="你已多次重复相同的工具调用，请不要再调用工具，直接基于现有信息给出最终答案。",
            )
            return trace

        # ④ 执行工具
        step = Step(index=step_idx, thought=resp.content, calls=list(resp.tool_calls))
        for call in resp.tool_calls:
            result = run_in_sandbox(call, self.registry, timeout_s=self.timeout_s)
            step.results.append(result)
            messages.append(result.to_message())
        trace.steps.append(step)

    # ⑤ 出口：步数上限
    trace.stop_reason = "max_steps"
    trace.final_answer = self._soft_landing(
        messages,
        hint="已达到工具调用步数上限，请不要再调用工具，直接基于现有信息给出最终答案。",
    )
    return trace
```

**注意 `stop_reason` 这个字段**——它不是装饰，是**事后分析的唯一线索**。

线上看到 80% 的请求都是 `max_steps`，就说明 `max_steps` 设小了；如果大量 `loop_detected`，说明工具设计有问题（返回的信息不够模型判断）。

---

## 别顺手撒谎：什么时候"不软着陆"

**有一种情况软着陆是有害的：模型正在假装自己有权限。**

比如 Agent 可以调 `delete_file`，但当前用户没权限。工具返回了 `ERROR: permission denied`，模型却可能总结成"已为您删除"。

**所以软着陆的提示词里必须加一条**：

```python
hint = (
    "请基于以上信息给出最终答案。"
    "如果因为权限、错误或信息不足导致任务未完成，请如实说明「未能完成」及原因，"
    "不要假设任何未实际执行成功的操作。"     # ★ 这条最重要
)
```

**没有这条约束的软着陆，等于教模型编造。**

---

## 小结

Agent Loop 的骨架只有三行：**调模型 → 看要不要用工具 → 把结果塞回去再调**。

**但它必须有能停下来的机制，而且"怎么停"决定了系统是否可用：**

1. **三个出口**：正常收敛 / 步数上限 / 循环检测
2. **软着陆优先于硬终止**：多一次调用，换用户拿到可用结果
3. **`stop_reason` 必须记录**：它是定位问题的唯一线索
4. **软着陆必须约束"不许撒谎"**：否则等于教模型编造

---

**代码实现**：[mini-agent-harness](https://github.com/jinhan-hu/ai-agent-projects/tree/main/04-agent-harness)（56 个测试，零第三方依赖）

**Demo**：[17 秒真机演示](https://github.com/jinhan-hu/ai-agent-projects/releases/download/v0.1.0-agent-harness/agent-harness-demo.mp4)——真实模型自主完成三步任务
