"""Tests for the HTTP API exposed by server.app."""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _http(url: str, *, method: str = "GET", body: dict | None = None,
          timeout: float = 5.0):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, method=method, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


class ServerApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from server.app import start_background_server
        from server.run_manager import RunManager

        cls._tmp = tempfile.TemporaryDirectory()
        cls.out_dir = Path(cls._tmp.name) / "output_web"

        def fake_runner(*, count, provider, output_dir, test_mode, log):
            log("hello")
            # Write a fake persona so /api/data has something to return
            out = Path(output_dir)
            persona_dir = out / "홍길동"
            persona_dir.mkdir(parents=True, exist_ok=True)
            (persona_dir / "persona.json").write_text(
                json.dumps({"name": "홍길동", "age": 30, "job": "개발자",
                            "home_city": "서울", "mbti": "ENTJ", "gender": "남",
                            "hobbies": [], "economic_level": "중",
                            "relationships": []}, ensure_ascii=False),
                encoding="utf-8",
            )
            for name in ("call", "sms", "noti", "gmail"):
                (persona_dir / f"behavior_events_{name}.json").write_text("[]", encoding="utf-8")
            (persona_dir / "behavior_events.json").write_text("[]", encoding="utf-8")
            (out / "metadata.json").write_text(
                json.dumps({"provider": provider, "elapsed_seconds": 0.01}),
                encoding="utf-8",
            )
            return {"output_dir": str(out), "personas": [{"name": "홍길동", "event_count": 0,
                                                         "counts": {"call": 0, "sms": 0, "noti": 0, "gmail": 0},
                                                         "output_dir": str(persona_dir)}],
                    "elapsed_seconds": 0.01, "count": count, "provider": provider,
                    "test_mode": test_mode}

        cls.run_manager = RunManager(default_output_dir=cls.out_dir, runner_fn=fake_runner)
        cls.httpd, cls.thread, cls.port, cls.state = start_background_server(
            output_dir=cls.out_dir,
            run_manager=cls.run_manager,
        )
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls._tmp.cleanup()

    def test_health(self):
        status, body = _http(f"{self.base}/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["output_dir"].endswith("output_web"))

    def test_data_initially_empty_list(self):
        status, body = _http(f"{self.base}/api/data")
        self.assertEqual(status, 200)
        self.assertIn("personas", body)
        self.assertIsInstance(body["personas"], list)

    def test_start_run_and_poll_until_success(self):
        status, body = _http(f"{self.base}/api/run", method="POST",
                             body={"count": 1, "provider": "claude", "test_mode": True})
        self.assertEqual(status, 202)
        run_id = body["run_id"]
        self.assertIsInstance(run_id, str)

        for _ in range(200):
            s, rec = _http(f"{self.base}/api/run/{run_id}")
            if rec["status"] in ("success", "failed"):
                break
            time.sleep(0.01)
        self.assertEqual(rec["status"], "success")
        self.assertTrue(any("hello" in line for line in rec["logs"]))

        # /api/data should now include the faked persona.
        s, data = _http(f"{self.base}/api/data")
        self.assertEqual(s, 200)
        names = [p["persona"]["name"] for p in data["personas"]]
        self.assertIn("홍길동", names)

    def test_invalid_provider_rejected(self):
        status, body = _http(f"{self.base}/api/run", method="POST",
                             body={"count": 1, "provider": "nope", "test_mode": True})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_invalid_count_rejected(self):
        status, body = _http(f"{self.base}/api/run", method="POST",
                             body={"count": 0, "provider": "claude", "test_mode": True})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_output_override_becomes_active_data_source(self):
        alt_out = Path(self._tmp.name) / "alt_output"
        status, body = _http(
            f"{self.base}/api/run",
            method="POST",
            body={"count": 1, "provider": "claude", "test_mode": True, "output": str(alt_out)},
        )
        self.assertEqual(status, 202)
        run_id = body["run_id"]

        for _ in range(200):
            _, rec = _http(f"{self.base}/api/run/{run_id}")
            if rec["status"] in ("success", "failed"):
                break
            time.sleep(0.01)
        self.assertEqual(rec["status"], "success")

        status, data = _http(f"{self.base}/api/data")
        self.assertEqual(status, 200)
        self.assertEqual(Path(data["output_dir"]).resolve(), alt_out.resolve())

    def test_failed_override_restores_previous_output_dir(self):
        status, body = _http(
            f"{self.base}/api/run",
            method="POST",
            body={"count": 1, "provider": "claude", "test_mode": True},
        )
        self.assertEqual(status, 202)
        base_run_id = body["run_id"]
        for _ in range(200):
            _, rec = _http(f"{self.base}/api/run/{base_run_id}")
            if rec["status"] in ("success", "failed"):
                break
            time.sleep(0.01)
        self.assertEqual(rec["status"], "success")
        self.assertEqual(_http(f"{self.base}/api/data")[1]["output_dir"], str(self.out_dir))

        broken_out = Path(self._tmp.name) / "broken_output"
        def broken_runner(*, count, provider, output_dir, test_mode, log):
            raise RuntimeError('boom')

        original_runner = self.run_manager._runner_fn
        self.run_manager._runner_fn = broken_runner
        try:
            status, body = _http(
                f"{self.base}/api/run",
                method="POST",
                body={"count": 1, "provider": "claude", "test_mode": True, "output": str(broken_out)},
            )
            self.assertEqual(status, 202)
            failed_run_id = body["run_id"]
            for _ in range(200):
                _, rec = _http(f"{self.base}/api/run/{failed_run_id}")
                if rec["status"] in ("success", "failed"):
                    break
                time.sleep(0.01)
            self.assertEqual(rec["status"], "failed")
            self.assertEqual(_http(f"{self.base}/api/data")[1]["output_dir"], str(self.out_dir))
        finally:
            self.run_manager._runner_fn = original_runner

    def test_output_override_outside_allowed_root_rejected(self):
        status, body = _http(
            f"{self.base}/api/run",
            method="POST",
            body={"count": 1, "provider": "claude", "test_mode": True, "output": "/tmp/outside-root"},
        )
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_static_repo_files_not_exposed(self):
        status, body = _http(f"{self.base}/.git/config")
        self.assertIn(status, (403, 404))
        self.assertIn("error", body)

    def test_unknown_run_returns_404(self):
        status, _ = _http(f"{self.base}/api/run/does-not-exist")
        self.assertEqual(status, 404)

    def test_list_runs_includes_started_run(self):
        # Start one run so the list is guaranteed non-empty regardless of
        # test execution order.
        _http(f"{self.base}/api/run", method="POST",
              body={"count": 1, "provider": "claude", "test_mode": True})
        status, body = _http(f"{self.base}/api/runs")
        self.assertEqual(status, 200)
        self.assertIn("runs", body)
        self.assertTrue(len(body["runs"]) >= 1)


if __name__ == "__main__":
    unittest.main()
