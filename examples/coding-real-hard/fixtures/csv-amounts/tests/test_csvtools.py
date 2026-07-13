import unittest

from app.csvtools import total_amount


class CsvToolsTests(unittest.TestCase):
    def test_sums_amount_column_after_header(self):
        csv_text = "name,amount\nAda,10.50\nGrace,2.25\n"
        self.assertEqual(total_amount(csv_text), 12.75)


if __name__ == "__main__":
    unittest.main()

