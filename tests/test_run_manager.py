"""Tests for server.run_manager."""
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class RunManagerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out_root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_start_returns_run_id_and_status_progresses_to_success(self):
        from server.run_manager import RunManager

        def fake_runner(*, count, provider, output_dir, test_mode, log):
            log("step 1")
            log("step 2")
            return {
                "output_dir": str(output_dir),
                "personas": [{"name": "홍길동", "event_count": 3,
                              "counts": {"call": 1, "sms": 1, "noti": 1, "gmail": 0},
                              "output_dir": str(output_dir / "홍길동")}],
                "elapsed_seconds": 0.01,
                "count": count,
                "provider": provider,
                "test_mode": test_mode,
            }

        mgr = RunManager(default_output_dir=self.out_root, runner_fn=fake_runner)
        run_id = mgr.start(count=1, provider="claude", test_mode=True)
        self.assertIsInstance(run_id, str)

        # Wait (bounded) for the background thread to finish.
        for _ in range(200):
            run = mgr.get(run_id)
            if run["status"] in ("success", "failed"):
                break
            time.sleep(0.01)

        run = mgr.get(run_id)
        self.assertEqual(run["status"], "success")
        self.assertIn("step 1", "\n".join(run["logs"]))
        self.assertEqual(run["summary"]["personas"][0]["name"], "홍길동")
        self.assertIsNone(run["error"])

    def test_runner_exception_marks_failed(self):
        from server.run_manager import RunManager

        def broken_runner(**_kwargs):
            raise RuntimeError("boom")

        mgr = RunManager(default_output_dir=self.out_root, runner_fn=broken_runner)
        run_id = mgr.start(count=1, provider="claude", test_mode=True)
        for _ in range(200):
            run = mgr.get(run_id)
            if run["status"] in ("success", "failed"):
                break
            time.sleep(0.01)

        run = mgr.get(run_id)
        self.assertEqual(run["status"], "failed")
        self.assertIn("boom", run["error"])

    def test_get_unknown_run_returns_none(self):
        from server.run_manager import RunManager

        mgr = RunManager(default_output_dir=self.out_root, runner_fn=lambda **_: None)
        self.assertIsNone(mgr.get("no-such-id"))

    def test_list_recent_runs(self):
        from server.run_manager import RunManager

        mgr = RunManager(default_output_dir=self.out_root,
                         runner_fn=lambda **kw: {"output_dir": str(kw["output_dir"]),
                                                 "personas": []})
        ids = [mgr.start(count=1, provider="claude", test_mode=True) for _ in range(3)]
        for _ in range(200):
            if all(mgr.get(i)["status"] in ("success", "failed") for i in ids):
                break
            time.sleep(0.01)
        runs = mgr.list()
        self.assertEqual(len(runs), 3)
        self.assertTrue(all("run_id" in r and "status" in r for r in runs))

    def test_attach_metadata_updates_existing_run(self):
        from server.run_manager import RunManager

        mgr = RunManager(default_output_dir=self.out_root,
                         runner_fn=lambda **kw: {"output_dir": str(kw["output_dir"]), "personas": []})
        run_id = mgr.start(count=1, provider="claude", test_mode=True)
        mgr.attach_metadata(run_id, previous_output_dir='prev/out')
        for _ in range(200):
            run = mgr.get(run_id)
            if run["status"] in ("success", "failed"):
                break
            time.sleep(0.01)
        self.assertEqual(mgr.get(run_id)["previous_output_dir"], 'prev/out')


if __name__ == "__main__":
    unittest.main()
