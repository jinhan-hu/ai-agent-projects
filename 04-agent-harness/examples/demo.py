"""最小可运行示例：注册两个工具并查看 schema。

（Agent Loop 尚未实现，所以这里只演示注册部分。）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.registry import ToolRegistry  # noqa: E402
from tools.example_tools import calculator, read_file  # noqa: E402

reg = ToolRegistry()
reg.tool(description="计算数学表达式")(calculator)
reg.tool(description="读取文本文件")(read_file)

print(f"已注册 {len(reg)} 个工具：")
for t in reg.list_tools():
    print(f"  - {t.name}: {t.description}")
    print(f"    参数: {t.parameters}")

print("\n直接调用 calculator('2 * (3 + 4)') =", calculator("2 * (3 + 4)"))
