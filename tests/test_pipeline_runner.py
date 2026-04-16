"""Tests for server.pipeline_runner — shared pipeline execution helpers."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class BuildInitialStateTests(unittest.TestCase):
    def test_contains_required_keys(self):
        from server.pipeline_runner import build_initial_state

        state = build_initial_state(count=2, provider="claude", test_mode=True)
        for key in (
            "count", "provider", "test_mode", "persons", "personas",
            "daily_drafts", "daily_events_map",
            "raw_calls", "raw_sms", "raw_push", "raw_emails",
            "persona_event_id_map",
            "generated_calls", "generated_sms", "generated_noti", "generated_gmail",
            "behavior_events", "behavior_events_map",
            "metadata",
        ):
            self.assertIn(key, state)
        self.assertEqual(state["count"], 2)
        self.assertEqual(state["provider"], "claude")
        self.assertTrue(state["test_mode"])
        self.assertEqual(state["personas"], [])
        self.assertIn("start_time", state["metadata"])
        self.assertEqual(state["metadata"]["provider"], "claude")


class SavePipelineOutputTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _fake_result(self):
        personas = [
            {"name": "홍길동", "age": 30, "job": "개발자"},
            {"name": "김유정", "age": 27, "job": "교사"},
        ]
        return {
            "personas": personas,
            "behavior_events_map": {
                "홍길동": [{"identifier": "call_1_1000", "timestamp": 1000,
                            "event_source": "CALL_LOG", "payload": {}}],
                "김유정": [{"identifier": "sms_2_2000", "timestamp": 2000,
                            "event_source": "SMS", "payload": {}}],
            },
            "behavior_events_by_type": {
                "홍길동": {
                    "call": [{"identifier": "call_1_1000", "timestamp": 1000,
                              "event_source": "CALL_LOG", "payload": {}}],
                    "sms": [],
                    "noti": [],
                    "gmail": [],
                },
                "김유정": {
                    "call": [],
                    "sms": [{"identifier": "sms_2_2000", "timestamp": 2000,
                             "event_source": "SMS", "payload": {}}],
                    "noti": [],
                    "gmail": [],
                },
            },
            "metadata": {"start_time": 1.0, "provider": "claude"},
        }

    def test_save_writes_per_persona_files(self):
        from server.pipeline_runner import save_pipeline_output

        summary = save_pipeline_output(
            self._fake_result(), self.out_dir, elapsed_seconds=1.5,
        )

        for name in ("홍길동", "김유정"):
            persona_dir = self.out_dir / name
            self.assertTrue((persona_dir / "persona.json").exists())
            self.assertTrue((persona_dir / "behavior_events.json").exists())
            self.assertTrue((persona_dir / "behavior_events_call.json").exists())
            self.assertTrue((persona_dir / "behavior_events_sms.json").exists())

        metadata_path = self.out_dir / "metadata.json"
        self.assertTrue(metadata_path.exists())
        meta = json.loads(metadata_path.read_text())
        self.assertEqual(meta["provider"], "claude")
        self.assertAlmostEqual(meta["elapsed_seconds"], 1.5)

        self.assertEqual(summary["output_dir"], str(self.out_dir))
        self.assertEqual(summary["elapsed_seconds"], 1.5)
        self.assertEqual(len(summary["personas"]), 2)

        hong = next(p for p in summary["personas"] if p["name"] == "홍길동")
        self.assertEqual(hong["event_count"], 1)
        self.assertEqual(hong["counts"], {"call": 1, "sms": 0, "noti": 0, "gmail": 0})
        self.assertTrue(hong["output_dir"].endswith("홍길동"))

    def test_save_falls_back_when_behavior_events_map_empty(self):
        from server.pipeline_runner import save_pipeline_output

        result = self._fake_result()
        result["behavior_events_map"] = {}
        result["behavior_events"] = [
            {"identifier": "n_1", "timestamp": 1, "event_source": "NOTIFICATION", "payload": {}}
        ]
        summary = save_pipeline_output(result, self.out_dir, elapsed_seconds=0.1)
        # personas still get their dirs written, just with empty events
        self.assertTrue((self.out_dir / "metadata.json").exists())
        self.assertEqual(len(summary["personas"]), 2)


if __name__ == "__main__":
    unittest.main()
