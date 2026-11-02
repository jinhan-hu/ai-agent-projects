"""跑全部测试。"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

MODULES = [
    "tests.test_registry",
    "tests.test_mock_llm",
    "tests.test_sandbox",
    "tests.test_memory",
    "tests.test_retry",
    "tests.test_tracing",
    "tests.test_eval",
    "tests.test_loop",
]


def main() -> int:
    total = passed = 0
    for modname in MODULES:
        mod = __import__(modname, fromlist=["_"])
        tests = sorted(
            (n, getattr(mod, n)) for n in dir(mod) if n.startswith("test_")
        )
        if not tests:
            continue
        print(f"\n[{modname}]  {len(tests)} 个")
        for name, fn in tests:
            total += 1
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL  {name}: {e}")
    print("\n" + "=" * 50)
    print(f"总计 {passed}/{total} 通过")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
