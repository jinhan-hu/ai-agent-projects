import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.eval import EvalCase, EvalReport, check, evaluate  # noqa: E402
from harness.types import Trace  # noqa: E402


def test_check_contains() -> None:
    c = EvalCase(task="t", expected="42", checker="contains")
    assert check("答案是 42 哦", c) is True
    assert check("答案是 43", c) is False


def test_check_exact() -> None:
    c = EvalCase(task="t", expected="42", checker="exact")
    assert check("42", c) is True
    assert check(" 42 ", c) is True
    assert check("答案是 42", c) is False


def test_check_regex() -> None:
    c = EvalCase(task="t", expected=r"\d+", checker="regex")
    assert check("有 42 个", c) is True
    assert check("没有数字", c) is False


class _FakeLoop:
    def __init__(self, answers):
        self.answers = list(answers)
        self.i = 0

    def run(self, task):
        a = self.answers[self.i]
        self.i += 1
        return Trace(task=task, final_answer=a, stop_reason="converged")


def test_evaluate_all_pass() -> None:
    cases = [EvalCase(task="a", expected="1"), EvalCase(task="b", expected="2")]
    r = evaluate(cases, _FakeLoop(["答案是 1", "答案是 2"]))
    assert r.total == 2
    assert r.passed == 2
    assert r.accuracy == 1.0


def test_evaluate_mixed() -> None:
    cases = [EvalCase(task="a", expected="1"), EvalCase(task="b", expected="2")]
    r = evaluate(cases, _FakeLoop(["答案是 1", "答案是 999"]))
    assert r.passed == 1
    assert abs(r.accuracy - 0.5) < 1e-9
    assert "1/2" in r.summary()


def test_report_properties_empty() -> None:
    r = EvalReport()
    assert r.total == 0
    assert r.accuracy == 0.0

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
