from lethe.memory.llm import AsyncLLMClient


def test_ssl_bad_record_mac_is_transient():
    msg = "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] sslv3 alert bad record mac (_ssl.c:2580)".lower()
    assert AsyncLLMClient._is_transient_error(msg)


def test_rate_limit_classification():
    assert AsyncLLMClient._is_rate_limit_error("429 too many requests")
    assert not AsyncLLMClient._is_rate_limit_error("connection reset by peer")
