"""
Graph-Based Road Network Analysis for DUK.
Focuses on topological connectivity, spur pruning, and endpoint bridging.
"""

import logging
import numpy as np
import cv2
from typing import List, Tuple, Optional
from shapely.geometry import LineString, Point
from skimage.morphology import skeletonize
from skan import Skeleton, summarize

logger = logging.getLogger(__name__)


class RoadNetworkCleaner:
    """
    Cleans skeletonized road masks using graph theory.
    Removes short dead-end spurs and bridges small topological gaps.
    """

    def __init__(self, min_branch_length: float = 15.0, bridge_gap_px: float = 25.0):
        self.min_branch_length = min_branch_length
        self.bridge_gap_px = bridge_gap_px

    def clean_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Skeletonize and prune a binary road mask.
        """
        if binary_mask.sum() == 0:
            return binary_mask

        # 1. Skeletonize
        skeleton = skeletonize(binary_mask > 0)
        if skeleton.sum() == 0:
            return binary_mask

        try:
            # 2. Graph analysis with skan
            skel_obj = Skeleton(skeleton.astype(bool))
            stats = summarize(skel_obj, find_main_branch=False)

            # 3. Identify spurs (type 1: junction to endpoint)
            # Branch-type 1 in skan is often a spur from a junction to an endpoint
            # Branch-type 0 is an isolated branch (endpoint to endpoint)
            # Branch-type 2 is junction-to-junction (core network)

            keep_mask = np.zeros_like(skeleton, dtype=bool)
            for idx, row in stats.iterrows():
                b_type = row.get("branch-type", 2)
                b_len = row.get("branch-distance", 0)

                # Keep junction-to-junction always, and long spurs
                if b_type == 2 or b_len >= self.min_branch_length:
                    path = skel_obj.path_coordinates(idx)
                    for r, c in path.astype(int):
                        if 0 <= r < skeleton.shape[0] and 0 <= c < skeleton.shape[1]:
                            keep_mask[r, c] = True

            return keep_mask.astype(np.uint8)

        except Exception as e:
            logger.warning(f"Topology cleanup failed: {e}")
            return skeleton.astype(np.uint8)

    def bridge_endpoints(self, lines: List[LineString]) -> List[LineString]:
        """
        Connect nearby dead-ends if they are within bridge_gap_px and
        roughly aligned.
        """
        if len(lines) < 2:
            return lines

        new_lines = list(lines)
        endpoints = []

        for i, line in enumerate(lines):
            if line.is_empty:
                continue
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            # Only consider "dead ends" (ends that aren't junctions)
            # For simplicity, we consider all endpoints and then filter by distance
            endpoints.append(
                {
                    "idx": i,
                    "type": "start",
                    "pt": np.array(coords[0]),
                    "vec": np.array(coords[0]) - np.array(coords[1]),
                }
            )
            endpoints.append(
                {
                    "idx": i,
                    "type": "end",
                    "pt": np.array(coords[-1]),
                    "vec": np.array(coords[-1]) - np.array(coords[-2]),
                }
            )

        for i in range(len(endpoints)):
            p1 = endpoints[i]["pt"]
            v1 = endpoints[i]["vec"]
            v1 = v1 / (np.linalg.norm(v1) + 1e-6)

            for j in range(i + 1, len(endpoints)):
                if endpoints[i]["idx"] == endpoints[j]["idx"]:
                    continue

                p2 = endpoints[j]["pt"]
                dist = np.linalg.norm(p1 - p2)

                if dist < self.bridge_gap_px:
                    # Connection logic: check alignment (cosine similarity)
                    v2 = endpoints[j]["vec"]
                    v2 = v2 / (np.linalg.norm(v2) + 1e-6)

                    # Connection vector
                    v_conn = p2 - p1
                    v_conn = v_conn / (np.linalg.norm(v_conn) + 1e-6)

                    # Check if the connection is "smooth" (along the trajectory of both lines)
                    # We want dot(v1, v_conn) > 0.5 (or similar)
                    if np.dot(v1, v_conn) > 0.4 and np.dot(v2, -v_conn) > 0.4:
                        logger.info(f"Bridging road gap: {dist:.1f}px")
                        new_lines.append(LineString([Point(p1), Point(p2)]))

        return new_lines
