# 2D Object Detection and Tracking using Kalman Filter (ROS 1 Melodic)

## Overview
This project implements a multi-object detection and tracking system using **2D LiDAR** data, **Kalman filtering**, and **ROS 1 Melodic**. Objects are detected using simple clustering techniques from LiDAR scan data and tracked over time using individual Kalman filters. The results, including object IDs, positions, and orientations, are visualized in real-time using **RViz**.

## [Project File with built workspace](https://drive.google.com/file/d/1FDIc_OrW5SLWYXw_oZuOyq7VyVDa-VZB/view?usp=sharing)
Do not fork the repository, just download the above file which has all the executables. The dataset should be downloaded separately

## System Requirements
   ```
python == 2.7
numpy == 1.16.6
filterpy == 1.4.5
   ```


## Docker and ROS Setup

Allow GUI Display on Host
   ```bash
xhost +local:root
   ```
Run Docker Container
   ```bash
docker run -it --rm \
  --name ros1_melodic \
  --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/rosbags:/rosbags \
  ros:melodic-robot \
  bash
   ```
Install Necessary Packages (Inside Docker)
```bash
apt update
apt install python-pip ros-melodic-rviz ros-melodic-cv-bridge python-opencv -y
pip install filterpy numpy
   ```

## Build and Run Instructions
Build the Workspace and launch roscore
   ```bash
cd /rosbags
catkin_make
source devel/setup.bash
source /opt/ros/melodic/setup.bash
roscore
   ```
Play ROS Bag (In a new terminal)
   ```bash
docker exec -it ros1_melodic bash
source /opt/ros/melodic/setup.bash
rosbag play /rosbags/your_bag.bag
   ```
Run Object Tracker Node (In a new terminal)
   ```bash
docker exec -it ros1_melodic bash
source /opt/ros/melodic/setup.bash
source /rosbags/devel/setup.bash
rosrun object_tracking object_tracker.py
   ```
Launch RViz for Visualization (In a new terminal)
```bash
docker exec -it ros1_melodic bash
source /opt/ros/melodic/setup.bash
rviz
```
In RViz:
1. Set “Fixed Frame” to base_link.
2. Add “MarkerArray” and subscribe to /tracked_objects.

## Implementation Details

### What I Did
- **LiDAR Pre-processing**  
  Subscribed to the `/scan` topic (2-D LiDAR) and converted raw range measurements into Cartesian `(x, y)` points in the robot’s body frame.
- **Clustering**  
  Wrote a simple distance-threshold clustering algorithm to group nearby points into object candidates.
- **Tracking Manager**  
  - Created and removed tracks dynamically.  
  - Ran **nearest-neighbour data association** between cluster centroids and existing tracks.  
  - Integrated a constant-velocity **Kalman filter** (`state = [x, y, vx, vy]`) for each track.  
  - Tuned process / measurement covariances and implemented lost-track pruning.
- **Yaw Estimation**  
  Computed each object’s orientation from recent motion history and applied it to marker orientation.
- **RViz Publishing**  
  Published coloured cube markers and overlaid text labels (object IDs) via `visualization_msgs/MarkerArray` for real-time visualisation in RViz.


