from dataclasses import dataclass

@dataclass(frozen=True)
class RiskLevel:
    label: str
    icon: str
    color: str

LOW = RiskLevel("低风险", "●", "#16865c")
MEDIUM = RiskLevel("中风险", "▲", "#d97706")
HIGH = RiskLevel("高风险", "◆", "#c73e3a")

def classify_risk(probability: float) -> RiskLevel:
    if not 0 <= probability <= 1:
        raise ValueError("拥堵概率必须位于 0 到 1 之间")
    if probability < 0.40:
        return LOW
    if probability < 0.70:
        return MEDIUM
    return HIGH
