import http.client
import datetime
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer

from experiment_agent import Experiments, job_timing
from experiment_control import read_control, set_paused
from experiment_dashboard import ArchiveBridge, Dashboard, SSHBridge, handler
from archive.run_replay_followups import Queue


class FakeBridge:
    def __init__(self):
        self.paused = False
        self.calls = []

    def call(self, method, **arguments):
        self.calls.append(method)
        if method == "snapshot":
            return {"control": {"paused": self.paused}}
        raise ValueError("Unknown method")


class TimingTests(unittest.TestCase):
    def test_training_uses_job_start_and_published_result_not_last_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log, result = root / 'train.log', root / 'result.json'
            log.write_text('launcher output\n'
                           '\x1b[32m2026-09-06T10:00:00Z\x1b[0m INFO loaded replay source\n'
                           '2026-09-06T10:01:00Z INFO starting fixed replay training pass epoch=0\n'
                           '2026-09-06T11:55:00Z INFO starting fixed replay training pass epoch=19\n')
            result.write_text('{"completed_epochs":20}')
            end = datetime.datetime(2026, 9, 6, 12, tzinfo=datetime.timezone.utc).timestamp()
            os.utime(result, (end, end))
            # Log modification time may reflect later inspection or append activity.
            os.utime(log, (end + 1000, end + 1000))
            agent = Experiments(root, root)
            phase = agent.training('train', 'Train', root, log, 'pending', 'kata-v1')
            self.assertEqual(phase['started_at'], '2026-09-06T10:00:00+00:00')
            self.assertEqual(phase['ended_at'], '2026-09-06T12:00:00+00:00')
            self.assertEqual(phase['duration_seconds'], 7200)
            self.assertEqual(phase['current_pass_started_at'], '2026-09-06T11:55:00+00:00')

    def test_appended_resume_logs_preserve_original_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log, result = root / 'train.log', root / 'result.json'
            log.write_text('2026-09-06T10:00:00Z INFO first attempt\n' + 'output\n' * 12000 +
                           '2026-09-06T11:00:00Z INFO resumed training\n')
            result.write_text('{}')
            end = datetime.datetime(2026, 9, 6, 12, tzinfo=datetime.timezone.utc).timestamp()
            os.utime(result, (end, end))
            self.assertEqual(job_timing(log, result, True)['duration_seconds'], 7200)

    def test_match_timing_includes_setup_and_has_documented_missing_log_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log, result = root / 'battle.log', root / 'battle.json'
            log.write_text('2026-09-06T10:00:00Z INFO loading models\n')
            result.write_text('{}')
            end = datetime.datetime(2026, 9, 6, 10, 2, tzinfo=datetime.timezone.utc).timestamp()
            os.utime(result, (end, end))
            timing = job_timing(log, result, True, reported_duration=110)
            self.assertEqual(timing['duration_seconds'], 120)
            log.unlink()
            timing = job_timing(log, result, True, reported_duration=110)
            self.assertEqual(timing['duration_seconds'], 110)
            self.assertEqual(timing['started_at'], '2026-09-06T10:00:10+00:00')
            self.assertEqual(timing['timing_source'], 'report_duration_and_result')

    def test_pending_and_running_jobs_never_invent_an_end_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log, result = root / 'train.log', root / 'result.json'
            self.assertEqual(job_timing(log, result, False), {
                'started_at': None, 'ended_at': None, 'duration_seconds': None, 'timing_source': None})
            start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=90)
            log.write_text(start.isoformat() + ' INFO starting\n')
            timing = job_timing(log, result, False, running=True)
            self.assertIsNone(timing['ended_at'])
            self.assertGreaterEqual(timing['duration_seconds'], 90)
            self.assertIsNone(job_timing(log, result, False)['duration_seconds'])


class ControlTests(unittest.TestCase):
    def test_archive_serves_saved_results_and_only_known_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'archive.json'
            path.write_text(json.dumps({'snapshot': {'phases': []},
                                       'logs': {'job': {'lines': ['saved']}}}))
            bridge = ArchiveBridge(path)
            self.assertTrue(bridge.call('snapshot')['archived'])
            self.assertEqual(bridge.call('logs', phase_id='job')['lines'], ['saved'])
            for method, arguments in [('pause', {}), ('exec', {}),
                                      ('logs', {'phase_id': '../../etc/passwd'})]:
                with self.assertRaises(ValueError): bridge.call(method, **arguments)

    def test_training_reports_actual_legacy_and_cached_backends(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'activation/kata-v1'
            directory.mkdir(parents=True)
            (directory / 'replay-config.json').write_text('{"model":{"architecture":"kata_v1"}}')
            agent = Experiments(root / 'activation', root / 'queue')
            performance = agent.snapshot()['phases'][0]['training_performance']
            self.assertEqual(performance, {'adam_backend': 'standard', 'replay_cache': 'none', 'prefetch_batches': 0})
            (directory / 'batch-cache.json').write_text('{"mode":"device","prefetch_batches":0}')
            self.assertEqual(agent.snapshot()['phases'][0]['training_performance']['replay_cache'], 'device')

    def test_pause_persists_and_worker_waits_before_next_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            set_paused(root, True)
            self.assertTrue(read_control(root)["paused"])
            queue = Queue(SimpleNamespace(output_dir=root, binary=Path('/tmp/test')))
            with patch.object(queue, "status") as status:
                with patch("archive.run_replay_followups.time.sleep", side_effect=lambda _: set_paused(root, False)) as sleep:
                    queue.wait_until_resumed("next-match")
                sleep.assert_called_once()
                status.assert_called_once_with("paused", next_stage="next-match")
            self.assertFalse(read_control(root)["paused"])

    def test_agent_is_read_only_and_exposes_only_known_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = Experiments(root / "activation", root / "queue")
            snapshot = agent.dispatch({"method": "snapshot"})
            self.assertEqual(len(snapshot["phases"]), 9)
            self.assertFalse(snapshot["worker"]["alive"])
            for request in [{"method": "pause"}, {"method": "resume"},
                            {"method": "shutdown"}, {"method": "exec", "command": "anything"},
                            {"method": "logs", "phase_id": "../../etc/passwd"}]:
                with self.assertRaises(ValueError):
                    agent.dispatch(request)
            self.assertFalse((root / "queue").exists())

    def test_bridge_rejects_mutations_before_connecting(self):
        bridge = SSHBridge({})
        with patch.object(bridge, "connect") as connect:
            for method in ("pause", "resume", "shutdown", "exec"):
                with self.assertRaisesRegex(ValueError, "read-only"):
                    bridge.call(method)
            connect.assert_not_called()

    def test_invalid_control_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "control.json").write_text('{"paused":"false"}')
            with self.assertRaises(ValueError):
                read_control(root)
            with self.assertRaises(ValueError):
                set_paused(root, "false")


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.assets = Path(self.temp.name)
        (self.assets / 'index.html').write_text('<html>dashboard</html>')
        self.bridge = FakeBridge()
        self.dashboard = Dashboard(self.bridge)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler(self.dashboard, self.assets))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    def test_snapshot_and_head_are_available(self):
        self.dashboard.refresh()
        status, data = self.request('GET', '/api/experiments')
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(data)['snapshot']['control']['paused'])
        self.assertEqual(self.request('HEAD', '/api/experiments'), (200, b''))
        self.assertEqual(self.request('HEAD', '/experiments'), (200, b''))

    def test_all_mutating_http_methods_are_rejected_without_rpc(self):
        for method in ('POST', 'PUT', 'PATCH', 'DELETE'):
            for path in ('/api/experiments/queue', '/api/experiments', '/experiments'):
                status, data = self.request(method, path, '{"paused":true}',
                    {'Content-Type': 'application/json', 'Origin': 'https://dashboard.example'})
                self.assertEqual(status, 405)
                self.assertIn('read-only', json.loads(data)['error'])
        self.assertEqual(self.bridge.calls, [])
        self.assertFalse(self.bridge.paused)
        self.assertEqual(self.request('GET', '/api/experiments/queue?paused=true')[0], 404)

    def test_public_proxy_headers_and_asset_boundaries(self):
        self.assertEqual(self.request('GET', '/api/experiments', headers={
            'Host': 'dashboard.example', 'Origin': 'https://dashboard.example'})[0], 200)
        self.assertEqual(self.request('GET', '/../../etc/passwd')[0], 404)
        self.assertEqual(self.request('GET', '/api/arbitrary-command')[0], 404)


if __name__ == '__main__':
    unittest.main()
