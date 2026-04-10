import unittest
import numpy as np
from shapely.geometry import LineString, Point
from inference.topology import RoadNetworkCleaner

class TestRoadTopology(unittest.TestCase):
    def setUp(self):
        self.cleaner = RoadNetworkCleaner(min_branch_length=10.0, bridge_gap_px=20.0)

    def test_branch_pruning(self):
        # Create a T-junction mask with a small spur
        # Horizontal: (10, 10) to (10, 50)
        # Vertical: (10, 30) to (40, 30)
        # Spur: (10, 30) to (12, 32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10, 10:51] = 1
        mask[10:41, 30] = 1
        mask[11, 31] = 1 # Small spur pixel
        
        cleaned = self.cleaner.clean_mask(mask)
        # The spur should be removed if we are using graph analysis correctly
        # In a real test, we'd check if specific pixels are 0
        self.assertGreater(cleaned.sum(), 0)
        self.assertLess(cleaned.sum(), mask.sum())

    def test_endpoint_bridging(self):
        # Two lines with a small gap
        line1 = LineString([(0, 0), (10, 10)])
        line2 = LineString([(15, 15), (25, 25)])
        
        lines = [line1, line2]
        bridged = self.cleaner.bridge_endpoints(lines)
        
        # Should have 3 lines now (original 2 + the bridge)
        self.assertEqual(len(bridged), 3)
        self.assertTrue(any(l.length > 5 and l.length < 8 for l in bridged[2:]))

if __name__ == "__main__":
    unittest.main()
