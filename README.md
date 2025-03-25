<div align="center">
  
# Learning Orientation Field for OSM-Guided Autonomous Navigation
# Abstract

OpenStreetMap (OSM) has gained popularity recently in autonomous navigation due to its public accessibility, lower maintenance costs, and broader geographical coverage. However, existing methods often struggle with noisy OSM data and incomplete sensor observations, leading to inaccuracies in trajectory planning. These challenges are particularly evident in complex driving scenarios, such as at intersections or facing occlusions. To address these challenges, we propose a robust and explainable two-stage framework to learn an Orientation Field (OrField) for robot navigation by integrating LiDAR scans and OSM routes. In the first stage, we introduce the novel representation, OrField, which can provide orientations for each grid on the map, reasoning jointly from noisy LiDAR scans and OSM routes. To generate a robust OrField, we train a deep neural network by encoding a versatile initial OrField and output an optimized OrField. Based on OrField, we propose two trajectory planners for OSM-guided robot navigation, called Field-RRT* and Field-Bezier, respectively, in the second stage by improving the Rapidly Exploring Random Tree (RRT) algorithm and Bezier curve to estimate the trajectories. Thanks to the robustness of OrField which captures both global and local information, Field-RRT* and Field-Bezier can generate accurate and reliable trajectories even in challenging conditions. We validate our approach through experiments on the SemanticKITTI dataset and our own campus dataset. The results demonstrate the effectiveness of our method, achieving superior performance in complex and noisy conditions. The code for network training and real-world deployment will be released.

<a href="https://"><img src='https://img.shields.io/badge/PDF-IEEE%20Xplore-purple' alt='PDF'></a>
<a href="https://arxiv.org/abs/2503.18276"><img src='https://img.shields.io/badge/PDF-arXiv-lightgreen' alt='PDF'></a>
<a href="https://"><img src='https://img.shields.io/badge/Video-YouTube-blue' alt='Video'></a>
<a href="https://"><img src='https://img.shields.io/badge/Dataset-red' alt='Dataset'></a>
<a href="https://"><img src='https://img.shields.io/badge/Supplementary%20Material-pink' alt='Supplementary Material'></a>

</div>

[![Learning Orientation Field for OSM-Guided Autonomous Navigation](cover.jpg)](https://youtu.be/nGqufhbP2NQ "Learning Orientation Field for OSM-Guided Autonomous Navigation")

## News
- **2025-03-18**: The paper was submitted to the Journal of Field Robotics. 
