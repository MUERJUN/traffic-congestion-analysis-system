import pytest
from src.app.risk import classify_risk

@pytest.mark.parametrize(("value", "label"), [(0, "低风险"), (.3999, "低风险"), (.40, "中风险"), (.6999, "中风险"), (.70, "高风险"), (1, "高风险")])
def test_boundaries(value, label):
    assert classify_risk(value).label == label

@pytest.mark.parametrize("value", [-.01, 1.01])
def test_invalid(value):
    with pytest.raises(ValueError):
        classify_risk(value)
