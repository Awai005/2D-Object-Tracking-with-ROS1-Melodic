#ifndef OBJECT_TRACKER_H
#define OBJECT_TRACKER_H

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Quaternion.h>
#include <tf/transform_datatypes.h>
#include <opencv2/video/tracking.hpp>

struct TrackState {
  int id;
  cv::KalmanFilter kf;
  int missed_frames;
  std::vector<cv::Point2f> history;
  std_msgs::ColorRGBA color;
};

class ObjectTracker {
public:
  ObjectTracker(ros::NodeHandle& nh);
  void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan_msg);

private:
  void clusterPoints(const std::vector<cv::Point2f>& pts,
                     std::vector<cv::Point2f>& centroids);
  void predictAll();
  void associateAndUpdate(const std::vector<cv::Point2f>& centroids);
  void removeLostTracks();
  void publishMarkers(const std::string& frame_id);

  ros::Subscriber scan_sub_;
  ros::Publisher marker_pub_;

  std::map<int, TrackState> tracks_;
  int next_id_;
  float dist_thresh_;
  int max_missed_;  
};

#endif // OBJECT_TRACKER_H