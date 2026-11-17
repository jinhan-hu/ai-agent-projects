"""两级记忆：会话记忆（短期）+ 客户档案（长期）。

为什么要分开：
    两者的**生命周期**不同。
    会话记忆随对话结束而失效；客户档案要跨会话保留（"这个客户上周问过同样的问题"）。
    塞进同一个容器会导致：要么短期记忆被长期污染，要么长期记忆被会话清空。

为什么内存版而不是数据库：
    本项目要能在 clone 之后零配置跑起来。持久化留了接口（Store Protocol），
    接 SQLite / Redis / 文件只是换一个实现。
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Protocol

MAX_HISTORY = 20  # 会话记忆最多保留多少轮


class Store(Protocol):
    """持久化接口。默认有内存实现，接数据库时替换即可。"""

    def load(self, key: str) -> list[dict]: ...
    def save(self, key: str, entries: list[dict]) -> None: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, list[dict]] = {}

    def load(self, key: str) -> list[dict]:
        return [dict(e) for e in self._data.get(key, [])]

    def save(self, key: str, entries: list[dict]) -> None:
        self._data[key] = [dict(e) for e in entries]


class Memory:
    """两级记忆的统一入口。"""

    def __init__(self, store: Store | None = None) -> None:
        self.store = store or InMemoryStore()

    # ---------- 长期：按客户归档 ----------
    def append_profile(self, customer: str, entry: dict) -> list[dict]:
        key = f"profile::{customer}"
        entries = self.store.load(key)
        entries.append({**entry, "time": datetime.now().isoformat(timespec="seconds")})
        self.store.save(key, entries)
        return entries

    def profile_context(self, customer: str, limit: int = 5) -> str:
        """把客户历史档案渲染成注入 prompt 的文本。"""
        if not customer:
            return ""
        entries = self.store.load(f"profile::{customer}")
        if not entries:
            return ""
        lines = []
        for e in entries[-limit:]:
            parts = [f"[{e.get('time', '')[:10]}]"]
            if e.get("route_label"):
                parts.append(f"处理类型：{e['route_label']}")
            if e.get("summary"):
                parts.append(f"结论：{e['summary']}")
            lines.append("　".join(parts))
        return "\n".join(lines)

    # ---------- 短期：按会话 ----------
    def append_turn(self, session_id: str, turn: dict) -> list[dict]:
        key = f"session::{session_id}"
        turns = self.store.load(key)
        turns.append(turn)
        turns = turns[-MAX_HISTORY:]
        self.store.save(key, turns)
        return turns

    def session_history(self, session_id: str) -> list[dict]:
        return self.store.load(f"session::{session_id}")

    def session_context(self, session_id: str, limit: int = 3) -> str:
        turns = self.session_history(session_id)[-limit:]
        if not turns:
            return ""
        return "\n".join(
            f"[第{len(self.session_history(session_id)) - len(turns) + i + 1}轮] "
            f"用户：{t.get('text', '')[:60]} → 类型：{t.get('route_label', '-')}"
            for i, t in enumerate(turns)
        )

    # ---------- 调试 ----------
    def dump(self) -> str:
        data = getattr(self.store, "_data", {})
        return json.dumps(data, ensure_ascii=False, indent=2)
