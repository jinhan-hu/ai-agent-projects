"""工具注册中心的测试 —— 项目里第一个能跑通的测试。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.registry import ToolRegistry  # noqa: E402


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


def test_openai_schema() -> None:
    reg = ToolRegistry()

    @reg.tool(description="测试工具")
    def ping() -> str:
        return "pong"

    schema = reg.to_openai_schema()
    assert schema[0]["type"] == "function"
    assert schema[0]["function"]["name"] == "ping"


def test_missing_tool_raises() -> None:
    reg = ToolRegistry()
    try:
        reg.get("not_exist")
    except KeyError as e:
        assert "未注册" in str(e)
    else:
        raise AssertionError("取不存在的工具应当抛 KeyError")


if __name__ == "__main__":
    # 不依赖 pytest，直接跑
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
