import unittest
from archive.run_board_mask_comparisons import overlap_ready, SETTINGS


class ComparisonScheduleTests(unittest.TestCase):
    def test_overlap_threshold_and_serial_fallback(self):
        self.assertFalse(overlap_ready(599,False,False))
        self.assertTrue(overlap_ready(600,False,False))
        self.assertFalse(overlap_ready(999,False,True))
        self.assertTrue(overlap_ready(1000,True,True))
        self.assertEqual(SETTINGS,{'games':1000,'simulations':4000,'temperature':.7,'parallelism':500,'inference_batch_size':128})

if __name__=='__main__':
    unittest.main()
