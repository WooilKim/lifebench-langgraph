"""Tests for server.data_loader — reads output dir, produces viewer payload."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_fixture(root: Path) -> None:
    persona_dir = root / "홍길동"

    _write(persona_dir / "persona.json", {
        "name": "홍길동", "age": 30, "gender": "남", "job": "개발자",
        "home_city": "서울", "mbti": "ENTJ",
        "hobbies": ["게임", "독서"],
        "economic_level": "중",
        "relationships": [{"name": "김철수", "relation": "친구"}],
    })

    calls = [
        {"identifier": f"call_{i}_{i*1000}", "timestamp": i * 1000,
         "event_source": "CALL_LOG",
         "payload": {"name": "김철수", "transcriptDialog": "대화 내용" if i % 2 == 0 else ""}}
        for i in range(1, 5)  # 4 calls, 2 with transcript
    ]
    sms = [{"identifier": "sms_1_10", "timestamp": 10, "event_source": "SMS",
            "payload": {"body": "안녕"}}]
    noti = [{"identifier": "noti_1_20", "timestamp": 20, "event_source": "NOTIFICATION",
             "payload": {"appName": "카카오톡"}}]
    gmail = [{"identifier": "gmail_1_30", "timestamp": 30, "event_source": "GMAIL",
              "payload": {"subject": "안내"}}]
    combined = calls + sms + noti + gmail

    _write(persona_dir / "behavior_events.json", combined)
    _write(persona_dir / "behavior_events_call.json", calls)
    _write(persona_dir / "behavior_events_sms.json", sms)
    _write(persona_dir / "behavior_events_noti.json", noti)
    _write(persona_dir / "behavior_events_gmail.json", gmail)

    _write(root / "metadata.json", {"provider": "claude", "elapsed_seconds": 1.23})


class LoadViewerPayloadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _make_fixture(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_entries_with_persona_stats_and_samples(self):
        from server.data_loader import load_viewer_payload

        entries = load_viewer_payload(self.root)
        self.assertEqual(len(entries), 1)
        entry = entries[0]

        self.assertEqual(entry["persona"]["name"], "홍길동")
        self.assertEqual(entry["stats"]["calls"], 4)
        self.assertEqual(entry["stats"]["sms"], 1)
        self.assertEqual(entry["stats"]["noti"], 1)
        self.assertEqual(entry["stats"]["gmail"], 1)
        self.assertEqual(entry["stats"]["total"], 7)
        self.assertEqual(entry["stats"]["transcript_count"], 2)

        for key in ("calls", "sms", "noti", "gmail"):
            self.assertIn(key, entry["samples"])
        self.assertTrue(len(entry["samples"]["calls"]) <= 10)
        self.assertTrue(len(entry["samples"]["calls"]) >= 1)

    def test_missing_output_dir_returns_empty_list(self):
        from server.data_loader import load_viewer_payload

        entries = load_viewer_payload(self.root / "does-not-exist")
        self.assertEqual(entries, [])

    def test_load_metadata_returns_dict(self):
        from server.data_loader import load_metadata

        meta = load_metadata(self.root)
        self.assertEqual(meta["provider"], "claude")

    def test_load_metadata_missing_returns_empty(self):
        from server.data_loader import load_metadata

        meta = load_metadata(self.root / "missing")
        self.assertEqual(meta, {})


if __name__ == "__main__":
    unittest.main()
