from __future__ import annotations

import unittest

from app.dashboard_utils import risk_level


class DashboardUtilityTests(unittest.TestCase):
    def test_risk_level_boundaries(self) -> None:
        self.assertEqual(risk_level(0.0), "低风险")
        self.assertEqual(risk_level(0.3999), "低风险")
        self.assertEqual(risk_level(0.40), "中风险")
        self.assertEqual(risk_level(0.6999), "中风险")
        self.assertEqual(risk_level(0.70), "高风险")


if __name__ == "__main__":
    unittest.main()

