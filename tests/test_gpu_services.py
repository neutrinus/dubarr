import unittest
import time
import concurrent.futures
from src.core.gpu_services import GPUService

class TestGPUService(unittest.TestCase):
    def test_priority_and_futures(self):
        service = GPUService("Test")
        service.start()

        execution_order = []

        def task_fn(val, delay=0.1):
            time.sleep(delay)
            execution_order.append(val)
            return val

        # 1. Test basic future return
        f1 = service.submit(task_fn, "A", priority=10)
        self.assertEqual(f1.result(), "A")

        # 2. Test priority ordering
        # Submit a long blocker to fill the queue
        blocker = service.submit(task_fn, "Blocker", delay=0.3, priority=100)
        
        # Give the worker a moment to definitely pick up the blocker
        time.sleep(0.05)
        
        # Submit Low priority first, then High priority while blocker is running
        f_low = service.submit(task_fn, "Low", priority=20)
        f_high = service.submit(task_fn, "High", priority=1)
        
        # Wait for all to complete
        blocker.result()
        f_low.result()
        f_high.result()
        
        # The first task after Blocker should be High, then Low
        self.assertEqual(execution_order, ["A", "Blocker", "High", "Low"])

        service.stop()
        service.join()

if __name__ == "__main__":
    unittest.main()
