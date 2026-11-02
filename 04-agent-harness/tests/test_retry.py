import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.retry import ErrorKind, classify_error, retry_call  # noqa: E402


def test_classify_retryable() -> None:
    assert classify_error(TimeoutError("t")) is ErrorKind.RETRYABLE
    assert classify_error(ConnectionError("c")) is ErrorKind.RETRYABLE
    assert classify_error(OSError("o")) is ErrorKind.RETRYABLE


def test_classify_fixable() -> None:
    assert classify_error(ValueError("v")) is ErrorKind.FIXABLE
    assert classify_error(TypeError("t")) is ErrorKind.FIXABLE
    assert classify_error(KeyError("k")) is ErrorKind.FIXABLE


def test_classify_fatal() -> None:
    assert classify_error(RuntimeError("r")) is ErrorKind.FATAL
    assert classify_error(Exception("e")) is ErrorKind.FATAL


def test_retry_succeeds_after_failures() -> None:
    state = {"n": 0}
    delays = []

    def flaky() -> str:
        state["n"] += 1
        if state["n"] < 3:
            raise ConnectionError("网络抖动")
        return "ok"

    out = retry_call(flaky, max_attempts=3, base_delay_s=0.1,
                     sleep=lambda s: delays.append(s))
    assert out == "ok"
    assert state["n"] == 3
    assert delays == [0.1, 0.2]     # 指数退避


def test_gives_up_after_max_attempts() -> None:
    state = {"n": 0}

    def always_fail() -> None:
        state["n"] += 1
        raise ConnectionError("一直失败")

    try:
        retry_call(always_fail, max_attempts=3, sleep=lambda s: None)
    except ConnectionError:
        pass
    else:
        raise AssertionError("应当在用完次数后抛出")
    assert state["n"] == 3


def test_no_retry_on_fixable() -> None:
    state = {"n": 0}

    def bad() -> None:
        state["n"] += 1
        raise ValueError("参数错")

    try:
        retry_call(bad, max_attempts=3, sleep=lambda s: None)
    except ValueError:
        pass
    else:
        raise AssertionError("FIXABLE 错误应当立即抛出")
    assert state["n"] == 1, "不应重试"

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
