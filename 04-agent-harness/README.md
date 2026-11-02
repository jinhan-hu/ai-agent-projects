# 04 · Mini Agent Harness

从零实现的**最小 Agent 执行框架**，纯 Python 标准库，零第三方依赖。

## 为什么写这个

RAG 解决"模型不知道"的问题，Agent 要解决"模型能不能动手"。
本项目把 Agent 的七个核心环节拆开实现，看清每一层在解决什么、代价是什么。

## 架构

![架构](docs/architecture.png)

**依赖方向：上层 → 下层。** `types.py` 被所有模块依赖；`loop.py` 是唯一调度中枢。

## 模块清单

| 模块 | 职责 | 状态 |
| --- | --- | :---: |
| `registry` | 工具注册与发现：统一 schema、装饰器注册、按名查找 | ✅ |
| `mock_llm` | 假 LLM 客户端：让 loop 可以离线测试 | ✅ |
| `loop` | Agent 主循环：思考 → 调用 → 观察 → 收敛 | 🚧 |
| `sandbox` | 工具执行隔离：超时、异常隔离、输出截断 | 🚧 |
| `memory` | 会话记忆：滑动窗口 + 摘要压缩 | 🚧 |
| `retry` | 错误分类、指数退避重试、降级兜底 | 🚧 |
| `tracing` | 每次调用的可回放记录 | 🚧 |
| `eval` | 对比不同策略的成功率与 token 成本 | 🚧 |

## Agent Loop 如何工作

![Agent Loop](docs/agent-loop.png)

主循环只有三步：**调模型 → 看它要不要用工具 → 把结果塞回去再调**。

循环必须有三个出口，否则会跑飞：

| 出口 | 触发条件 | 处理方式 |
| --- | --- | --- |
| **① 正常收敛** | 模型不再请求工具调用 | 直接返回答案 |
| **② 步数上限** | `step >= max_steps`（默认 10） | **软着陆**：再发一次不带 tools 的请求，让模型基于已有信息总结 |
| **③ 循环检测** | 同一 `(工具名, 参数)` 重复 3 次 | 打断并在 messages 里插入提示，要求模型换方法 |

**两个设计取舍**：

- **为什么不用硬超时**：直接 kill 会让用户拿到空结果。软着陆至少能给出部分答案。
- **messages 只追加不删除**：这是上下文膨胀的根源，也正是 `memory.py` 存在的理由。

## 设计目标

- **零依赖**：clone 即跑，只用标准库
- **可替换**：LLM 客户端抽象成接口，可接任意模型（OpenAI / DeepSeek / 本地）
- **可观测**：每次执行都有完整 trace，可回放定位问题
- **可测试**：`MockLLM` 让全流程无需 API key 即可跑单测

## 快速开始

```bash
git clone https://github.com/jinhan-hu/ai-agent-projects
cd ai-agent-projects/04-agent-harness

python tests/test_registry.py     # 12 个测试
python tests/test_mock_llm.py     # 5 个测试
python examples/demo.py           # 注册两个工具并打印 schema
```

**当前测试状态：17 / 17 通过。**

## 目录结构

```
04-agent-harness/
├── docs/                 架构图 / 流程图
├── harness/
│   ├── types.py          核心数据结构
│   ├── registry.py       工具注册（已完成 · 12 测试）
│   ├── mock_llm.py       假 LLM（已完成 · 5 测试）
│   ├── loop.py           Agent 主循环
│   ├── sandbox.py        沙箱执行
│   ├── memory.py         记忆管理
│   ├── retry.py          重试与兜底
│   ├── tracing.py        调用追踪
│   └── eval.py           评测
├── tools/                示例工具
├── tests/                单元测试
└── examples/             可运行示例
```

## 开发进度

| 日期 | 任务 | 状态 |
| --- | --- | :---: |
| 11-02 | 架构定稿 + `types` 补 `tool_calls` + `MockLLM` + 测试 5→12 | ✅ |
| 11-03 | `loop.py` 主循环（用 MockLLM 验证） | ⬜ |
| 11-04 | `sandbox.py` | ⬜ |
| 11-05 | `memory.py` | ⬜ |
| 11-06 | `retry.py` | ⬜ |
| 11-07 | `tracing.py` + `eval.py` | ⬜ |
| 11-08 | 串联 + 接真实 API + 1 分钟 demo | ⬜ |

> 建设中，2026-11 阅读周集中开发。
