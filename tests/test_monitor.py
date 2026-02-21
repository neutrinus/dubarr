import unittest
from unittest.mock import patch
from infrastructure.monitor import ResourceMonitor


class TestMonitor(unittest.TestCase):
    @patch("psutil.cpu_percent", return_value=10.0)
    @patch("psutil.virtual_memory")
    @patch("subprocess.run")
    @patch("time.sleep")
    def test_monitor_flow(self, mock_sleep, mock_run, mock_vm, mock_cpu):
        mock_vm.return_value.percent = 50.0
        mock_run.return_value.returncode = 0
        # Format: utilization.gpu, memory.used, memory.free, memory.total
        mock_run.return_value.stdout = "10, 1000, 7000, 8000\n20, 2000, 6000, 8000"

        # We want it to run exactly once
        mock_sleep.side_effect = Exception("StopLoop")

        monitor = ResourceMonitor(interval=0.1)

        with self.assertLogs(level="INFO") as cm:
            try:
                monitor.run()
            except Exception as e:
                if str(e) != "StopLoop":
                    raise e

        # Verify if log contains expected strings
        log_text = "".join(cm.output)
        self.assertIn("CPU:10.0%", log_text)
        self.assertIn("RAM:50.0%", log_text)
        self.assertIn("GPU0:10%", log_text)
        self.assertIn("GPU1:20%", log_text)

    def test_monitor_stop(self):
        monitor = ResourceMonitor()
        self.assertFalse(monitor.stop_event.is_set())
        monitor.stop()
        self.assertTrue(monitor.stop_event.is_set())
