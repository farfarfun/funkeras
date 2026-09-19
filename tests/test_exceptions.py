"""funkeras.exceptions 的正常路径与边界测试。"""

from funkeras.exceptions import FeatureConfigError, FunkerasError


def test_feature_config_error_is_funkeras_error():
    """FeatureConfigError 必须能被基类 FunkerasError 捕获。"""
    err = FeatureConfigError("参数不合法")
    assert isinstance(err, FunkerasError)
    assert isinstance(err, Exception)


def test_feature_config_error_message_without_context():
    """不传 context 时，错误信息就是原始 message，不附加任何内容。"""
    err = FeatureConfigError("params error")
    assert str(err) == "params error"
    assert err.context is None


def test_feature_config_error_message_with_context():
    """传入 context 时，错误信息里必须包含具体的配置片段，便于定位问题。"""
    context = {"key": "user_id", "type": "CateEmbeddingColumn"}
    err = FeatureConfigError("缺少 vocabulary/bucket_size", context=context)

    message = str(err)
    assert "缺少 vocabulary/bucket_size" in message
    assert "user_id" in message
    assert err.context == context


def test_feature_config_error_can_be_caught_specifically():
    """调用方应该能只捕获 FeatureConfigError，而不必用裸 except Exception。"""
    try:
        raise FeatureConfigError("boom", context={"a": 1})
    except FeatureConfigError as err:
        assert err.context == {"a": 1}
    else:
        raise AssertionError("FeatureConfigError 应该被捕获到")
