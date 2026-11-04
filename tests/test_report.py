import csv
import os
import tempfile
import unittest
from core.report import report_rows, summary, write_report

SCRIPT = [
    {"start": 0.0, "end": 2.0, "max_end": 2.5, "speaker": "A", "text_en": "Hello there."},
    {"start": 65.5, "end": 69.5, "speaker": "B", "text_en": "And the blue square!"},
]
RESULTS = [
    {
        "index": 1,
        "duration": 2.0,
        "status": "VALID_FALLBACK",
        "final_text": "I niebieski kwadrat!",
        "voice_type": "GOLDEN",
        "similarity": 0.91,
        "heard": "I niebieski kwadrat",
        "ctc_heard": "i niebieski kwadra",
        "attempts": 2,
    },
    {"index": 0, "duration": 1.8, "status": "ACCEPTED", "final_text": "Cześć.", "voice_type": "DYNAMIC", "attempts": 1},
]


class TestReport(unittest.TestCase):
    def test_rows_follow_the_script(self):
        rows = report_rows(SCRIPT, RESULTS)
        self.assertEqual([r["index"] for r in rows], [0, 1])
        self.assertEqual(rows[0]["limit"], "2.50")
        self.assertEqual(rows[0]["fill"], "90%")
        self.assertEqual(rows[1]["start"], "01:05.50")
        self.assertEqual(rows[1]["similarity"], "0.91")
        self.assertEqual(rows[0]["similarity"], "")
        self.assertEqual(rows[1]["ctc_heard"], "i niebieski kwadra")
        self.assertEqual(rows[0]["ctc_heard"], "")

    def test_summary(self):
        text = summary(report_rows(SCRIPT, RESULTS))
        self.assertIn("2 lines", text)
        self.assertIn("fill 63%", text)
        self.assertIn("VALID_FALLBACK 1", text)
        self.assertIn("similarity min 0.91", text)

    def test_write_report_is_tab_separated(self):
        path = os.path.join(tempfile.mkdtemp(), "report_pl.tsv")
        write_report(path, SCRIPT, RESULTS)
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        self.assertEqual(rows[1]["dub"], "I niebieski kwadrat!")
        self.assertEqual(rows[1]["voice"], "GOLDEN")


if __name__ == "__main__":
    unittest.main()
