#include "object_tracking_cpp/object_tracker.h"
#include <cmath>
#include <opencv2/core.hpp>
#include <visualization_msgs/Marker.h>

ObjectTracker::ObjectTracker(ros::NodeHandle& nh)
: next_id_(1), dist_thresh_(0.5), max_missed_(8)
{
  scan_sub_ = nh.subscribe("/scan", 10, &ObjectTracker::scanCallback, this);
  marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>(
    "/tracked_objects", 10);
}

void ObjectTracker::scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan) {
  // Convert scan to points
  std::vector<cv::Point2f> pts;
  for (size_t i = 0; i < scan->ranges.size(); ++i) {
    float r = scan->ranges[i];
    if (std::isfinite(r)) {
      float angle = scan->angle_min + i * scan->angle_increment;
      pts.emplace_back(r * cos(angle), r * sin(angle));
    }
  }
  // Cluster to get centroids
  std::vector<cv::Point2f> centroids;
  clusterPoints(pts, centroids);

  predictAll();
  associateAndUpdate(centroids);
  removeLostTracks();
  publishMarkers(scan->header.frame_id);
}

void ObjectTracker::clusterPoints(
  const std::vector<cv::Point2f>& pts,
  std::vector<cv::Point2f>& centroids)
{
  std::vector<bool> used(pts.size(), false);
  for (size_t i = 0; i < pts.size(); ++i) {
    if (used[i]) continue;
    std::vector<cv::Point2f> cluster { pts[i] };
    used[i] = true;
    for (size_t j = i+1; j < pts.size(); ++j) {
      if (!used[j] && cv::norm(pts[j] - pts[i]) < dist_thresh_) {
        cluster.push_back(pts[j]); used[j] = true;
      }
    }
    if (cluster.size() > 2) {
      cv::Point2f sum(0,0);
      for (auto& p : cluster) sum += p;
      centroids.push_back(sum * (1.0f/cluster.size()));
    }
  }
}

void ObjectTracker::predictAll() {
  for (auto& kv : tracks_) {
    TrackState& tr = kv.second;
    tr.kf.predict();
    tr.history.push_back(cv::Point2f(tr.kf.statePost.at<float>(0),
                                     tr.kf.statePost.at<float>(1)));
    if (tr.history.size() > 10) tr.history.erase(tr.history.begin());
    tr.missed_frames++;
  }
}

void ObjectTracker::associateAndUpdate(
  const std::vector<cv::Point2f>& centroids)
{
  std::set<int> updated;
  for (auto& c : centroids) {
    int best_id = -1; float best_dist = dist_thresh_;
    for (auto& kv : tracks_) {
      auto& tr = kv.second;
      float dx = tr.kf.statePost.at<float>(0) - c.x;
      float dy = tr.kf.statePost.at<float>(1) - c.y;
      float d = std::hypot(dx, dy);
      if (d < best_dist) { best_dist = d; best_id = tr.id; }
    }
    if (best_id > 0) {
      auto& tr = tracks_[best_id];
      cv::Mat meas = (cv::Mat_<float>(2,1) << c.x, c.y);
      tr.kf.correct(meas);
      tr.history.push_back(c);
      tr.missed_frames = 0;
      updated.insert(best_id);
    } else {
      TrackState tr;
      tr.id = next_id_++;
      tr.kf = cv::KalmanFilter(4,2,0);
      float dt = 0.1f;
      tr.kf.transitionMatrix = (cv::Mat_<float>(4,4) <<
        1,0,dt,0, 0,1,0,dt, 0,0,1,0, 0,0,0,1);
      tr.kf.measurementMatrix = cv::Mat::zeros(2,4,CV_32F);
      tr.kf.measurementMatrix.at<float>(0,0) = 1;
      tr.kf.measurementMatrix.at<float>(1,1) = 1;
      tr.kf.processNoiseCov = cv::Mat::eye(4,4,CV_32F)*0.01f;
      tr.kf.measurementNoiseCov = cv::Mat::eye(2,2,CV_32F)*0.15f;
      tr.kf.errorCovPost = cv::Mat::eye(4,4,CV_32F);
      tr.kf.statePost = (cv::Mat_<float>(4,1) << c.x, c.y, 0, 0);
      tr.history = {c};
      tr.missed_frames = 0;
      tr.color = std_msgs::ColorRGBA();
      tr.color.r = float(rand())/RAND_MAX;
      tr.color.g = float(rand())/RAND_MAX;
      tr.color.b = float(rand())/RAND_MAX;
      tr.color.a = 0.8;
      tracks_[tr.id] = tr;
      updated.insert(tr.id);
    }
  }
}

void ObjectTracker::removeLostTracks() {
  std::vector<int> to_remove;
  for (auto& kv : tracks_) {
    if (kv.second.missed_frames > max_missed_) to_remove.push_back(kv.first);
  }
  for (int id : to_remove) tracks_.erase(id);
}

void ObjectTracker::publishMarkers(const std::string& frame_id) {
  visualization_msgs::MarkerArray arr;
  for (auto& kv : tracks_) {
    auto& tr = kv.second;
    visualization_msgs::Marker m;
    m.header.frame_id = frame_id;
    m.header.stamp = ros::Time::now();
    m.ns = "tracked_objects";
    m.id = tr.id;
    m.type = visualization_msgs::Marker::CUBE;
    m.action = visualization_msgs::Marker::ADD;
    m.pose.position.x = tr.kf.statePost.at<float>(0);
    m.pose.position.y = tr.kf.statePost.at<float>(1);
    m.pose.position.z = 0.5;
    float yaw = 0;
    if (tr.history.size() >= 2) {
      auto& p1 = tr.history[tr.history.size()-2];
      auto& p2 = tr.history.back();
      yaw = std::atan2(p2.y-p1.y, p2.x-p1.x);
    }
    tf::Quaternion q;
    q.setRPY(0,0,yaw);
    m.pose.orientation.x = q.x();
    m.pose.orientation.y = q.y();
    m.pose.orientation.z = q.z();
    m.pose.orientation.w = q.w();
    m.scale.x = 0.3;
    m.scale.y = 0.3;
    m.scale.z = 0.5;
    m.color = tr.color;
    arr.markers.push_back(m);
    visualization_msgs::Marker t = m;
    t.id += 1000;
    t.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    t.scale.z = 0.2;
    t.pose.position.z = 1.2;
    t.text = std::to_string(tr.id);
    arr.markers.push_back(t);
  }
  marker_pub_.publish(arr);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "object_tracker_cpp");
  ros::NodeHandle nh;
  ObjectTracker tracker(nh);
  ros::spin();
  return 0;
}