"""一次性跑完全部测试。

    python tests/test_all.py
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ["run_all.py", "test_tools_and_eval.py", "test_retrieval.py", "test_biz_tools.py", "test_mcp.py"]


def main() -> int:
    total_pass = 0
    total_fail = 0
    failed_suites = []

    for suite in SUITES:
        path = os.path.join(HERE, suite)
        if not os.path.exists(path):
            print(f"⚠ 跳过不存在的测试文件：{suite}")
            continue
        print(f"\n{'=' * 62}\n▶ {suite}\n{'=' * 62}")
        proc = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, encoding="utf-8"
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        # 只回显汇总行，避免刷屏
        for line in out.splitlines():
            if line.startswith("通过 ") or line.startswith("  - "):
                print("  " + line)
        # ★ 用正则找「通过 X / Y」这一行（取最后一个，避免被正文里的字样干扰）
        import re as _re
        hits = _re.findall(r"通过\s*(\d+)\s*/\s*(\d+)", out)
        if hits:
            p, t = int(hits[-1][0]), int(hits[-1][1])
            total_pass += p
            total_fail += t - p
        else:
            print(f"  ⚠ 未能从 {suite} 的输出里解析「通过 X / Y」")
            failed_suites.append(suite)
        if proc.returncode != 0:
            failed_suites.append(suite)
            print(out[-1500:])

    print(f"\n{'=' * 62}")
    print(f"总计通过 {total_pass} / {total_pass + total_fail}")
    if failed_suites:
        print("失败的测试文件：" + ", ".join(failed_suites))
    print("=" * 62)
    return 0 if not failed_suites and total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


