import tempfile
import unittest
from pathlib import Path

import pandas as pd

from export_market_context import export_market_context


class MarketContextExportTests(unittest.TestCase):
    def test_latest_dates_sectors_and_leading_zero_codes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            pd.DataFrame({"date": ["2026-10-23", "2026-10-26"], "stock_code": ["005930", "005930"],
                          "sector": ["old", "old"], "return_1d": [0.01, -0.02]}).to_csv(directory / "feature_data.csv", index=False)
            pd.DataFrame({"code": ["005930"], "sector": ["Semiconductor"]}).to_csv(directory / "stock_info.csv", index=False)
            result = export_market_context(directory)
            frame = pd.read_csv(directory / "market_context_daily.csv", dtype={"stock_code": str})
            self.assertEqual(result["latest_date"], "2026-10-26")
            self.assertEqual(frame.stock_code.tolist(), ["005930", "005930"])
            self.assertEqual(frame.sector.tolist(), ["Semiconductor", "Semiconductor"])
            self.assertEqual(frame.return_1d.tolist(), [0.01, -0.02])
            self.assertFalse((directory / "market_context_daily.tmp").exists())


if __name__ == "__main__":
    unittest.main()
