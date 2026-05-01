# Contributions to the Munich Motorsport Driverless Team

This repository contains selected work performed while developing the autonomous vehicle software stack for the **driverless** project. The focus was on **state estimation**, **perception analysis**, and **real‑time performance optimisation**.

## 1. Perception Distance Measurement from ROS Bags

**File:** `max_perception_distance_node.py`

Developed a ROS 2 node that subscribes to `/local_cones` while replaying a recorded test bag. The node:
- Computes the Euclidean distance of each detected cone (in vehicle frame).
- Tracks the maximum distance encountered across all frames.
- Outputs the result upon shutdown.

This value was used to update the `perception_distance` parameter in the simulation, aligning the virtual sensor range with real‑world performance.

## 3. Ground Removal Latency Analysis

**File:** `plot_ground_removal_latency_time.py`

To verify that the LiDAR ground removal step (C++ with PCL) met the real‑time requirement (<100 ms):
- Added internal timing logs to `PreprocessingNode.cpp` (`GROUND_REMOVAL_LATENCY: XX ms`).
- Captured logs while replaying a bag.
- Wrote a Python script to parse the logs and plot latency over time.

The resulting plots confirmed that ground removal averaged ~25 ms and never exceeded 65 ms – proving it was not the cause of end‑to‑end perception delays.

---

### [!! 'perception_mock_node.py' and 'PreprocessingNode.cpp' were already written by the increadible **Munich Motorsport Driverless Team** who built an impressive autonomous racing system from the ground up.
/Original file copyright municHMotorsport e.V. ]

 
 
*These contributions helped improve the reliability and realism of the autonomous driving simulation and validated the performance of the perception pipeline on the Jetson AGX Orin.*
