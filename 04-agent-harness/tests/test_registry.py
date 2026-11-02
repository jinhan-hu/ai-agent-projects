"""工具注册中心的测试（12 个）。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.registry import ToolRegistry  # noqa: E402


# ---------- 基础功能 ----------
def test_register_and_get() -> None:
    reg = ToolRegistry()

    @reg.tool(description="两数相加")
    def add(a: int, b: int) -> int:
        return a + b

    assert len(reg) == 1
    assert reg.get("add").func(1, 2) == 3
    assert reg.get("add").description == "两数相加"


def test_duplicate_name_raises() -> None:
    reg = ToolRegistry()

    @reg.tool(name="dup")
    def a() -> None:
        pass

    try:

        @reg.tool(name="dup")
        def b() -> None:
            pass

    except ValueError as e:
        assert "重复" in str(e)
    else:
        raise AssertionError("重复注册应当抛 ValueError")


def test_missing_tool_raises() -> None:
    reg = ToolRegistry()
    try:
        reg.get("not_exist")
    except KeyError as e:
        assert "未注册" in str(e)
    else:
        raise AssertionError("取不存在的工具应当抛 KeyError")


def test_empty_registry() -> None:
    reg = ToolRegistry()
    assert len(reg) == 0
    assert reg.list_tools() == []
    assert reg.to_openai_schema() == []


def test_custom_name() -> None:
    reg = ToolRegistry()

    @reg.tool(name="my_add", description="加法")
    def add(a: int, b: int) -> int:
        return a + b

    assert "my_add" in [t.name for t in reg.list_tools()]
    assert reg.get("my_add").func(2, 3) == 5


def test_description_defaults_to_docstring() -> None:
    reg = ToolRegistry()

    @reg.tool()
    def greet(name: str) -> str:
        """打个招呼。"""
        return f"hi {name}"

    assert reg.get("greet").description == "打个招呼。"


def test_list_tools_preserves_order() -> None:
    reg = ToolRegistry()

    @reg.tool()
    def z_tool() -> None:
        pass

    @reg.tool()
    def a_tool() -> None:
        pass

    # 注册顺序即返回顺序（不做字母排序）
    assert [t.name for t in reg.list_tools()] == ["z_tool", "a_tool"]


# ---------- Schema 推导 ----------
def test_schema_generation() -> None:
    reg = ToolRegistry()

    @reg.tool()
    def greet(name: str, times: int = 1) -> str:
        return name * times

    params = reg.get("greet").parameters
    assert params["type"] == "object"
    assert params["properties"]["name"]["type"] == "string"
    assert params["properties"]["times"]["type"] == "integer"
    assert params["required"] == ["name"]


def test_schema_no_params() -> None:
    reg = ToolRegistry()

    @reg.tool()
    def ping() -> str:
        return "pong"

    params = reg.get("ping").parameters
    assert params["properties"] == {}
    assert params["required"] == []


def test_schema_float_and_bool() -> None:
    reg = ToolRegistry()

    @reg.tool()
    def set_temp(celsius: float, enabled: bool) -> None:
        pass

    props = reg.get("set_temp").parameters["properties"]
    assert props["celsius"]["type"] == "number"
    assert props["enabled"]["type"] == "boolean"


def test_schema_all_optional() -> None:
    reg = ToolRegistry()

    @reg.tool()
    def maybe(a: int = 1, b: str = "x") -> None:
        pass

    assert reg.get("maybe").parameters["required"] == []


# ---------- OpenAI 格式导出 ----------
def test_openai_schema() -> None:
    reg = ToolRegistry()

    @reg.tool(description="测试工具")
    def ping() -> str:
        return "pong"

    schema = reg.to_openai_schema()
    assert schema[0]["type"] == "function"
    assert schema[0]["function"]["name"] == "ping"
    assert schema[0]["function"]["description"] == "测试工具"
    assert "parameters" in schema[0]["function"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 通过")
