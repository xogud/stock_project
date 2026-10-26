import io
import tempfile
import tracemalloc
import unittest
from pathlib import Path

import pandas as pd

import app as api


class MemoryLoadingTests(unittest.TestCase):
    def test_xml_keeps_only_listed_companies_with_bounded_memory(self):
        unlisted = b'<list><corp_code>00000001</corp_code><corp_name>Unlisted</corp_name><stock_code> </stock_code></list>'
        listed = b'<list><corp_code>00126380</corp_code><corp_name>Samsung</corp_name><stock_code>005930</stock_code></list>'
        stream = io.BytesIO(b'<result>' + unlisted * 30000 + listed + b'</result>')
        tracemalloc.start()
        try:
            result = api.parse_dart_corp_codes(stream)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(result.stock_code.tolist(), ['005930'])
        self.assertEqual(result.corp_code.tolist(), ['00126380'])
        self.assertLess(peak, 4 * 1024 * 1024)

    def test_chunked_csv_matches_full_read_across_unsorted_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.csv'
            frame = pd.DataFrame({
                'trade_date': ['2026-10-26'] * 5 + ['2026-10-23'] * 10010 + ['2026-10-26'] * 3,
                'stock_code': ['005930'] * 10018,
                'risk_probability': [0.5] * 10018,
            })
            frame.to_csv(path, index=False)
            signature = api.file_signature(path)
            columns = tuple(frame.columns)
            result = api.csv_day_rows(*signature, columns, 'trade_date')
            pd.testing.assert_frame_equal(result, frame[frame.trade_date == '2026-10-26'].reset_index(drop=True))
            older = api.csv_day_rows(*signature, columns, 'trade_date', '2026-10-23')
            self.assertEqual(len(older), 10010)
            self.assertTrue(api.csv_day_rows(*signature, columns, 'trade_date', '2020-01-01').empty)

    def test_bounded_prediction_cache(self):
        self.assertEqual(api.build_prediction_frame.__wrapped__.cache_info().maxsize, 4)


if __name__ == '__main__':
    unittest.main()
