#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion
import numpy as np
from tf.transformations import quaternion_from_euler
from filterpy.kalman import KalmanFilter

class TrackedObject:
    def __init__(self, obj_id, x, y, dt=0.1):
        self.id = obj_id
        self.history = []
        self.dt = dt
        self.color = np.random.rand(3)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # F: state transition
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        # H: measurement function
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        # Covariances (tune!)
        self.kf.P *= 1
        self.kf.R = np.eye(2) * 0.15
        self.kf.Q = np.eye(4) * 0.01
        # Initial state
        self.kf.x[:2] = np.array([[x], [y]])
        self.kf.x[2:] = np.array([[0], [0]])
        self.lost = 0  # Counter for lost frames

    def predict(self):
        self.kf.predict()
        self.history.append((self.x, self.y))
        if len(self.history) > 10:
            self.history.pop(0)

    def update(self, x, y):
        self.kf.update([x, y])
        self.history.append((self.x, self.y))
        if len(self.history) > 10:
            self.history.pop(0)
        self.lost = 0

    @property
    def x(self): return self.kf.x[0, 0]
    @property
    def y(self): return self.kf.x[1, 0]
    @property
    def vx(self): return self.kf.x[2, 0]
    @property
    def vy(self): return self.kf.x[3, 0]
    @property
    def yaw(self):
        if len(self.history) >= 2:
            dx = self.history[-1][0] - self.history[-2][0]
            dy = self.history[-1][1] - self.history[-2][1]
            if abs(dx) + abs(dy) > 1e-3:
                return np.arctan2(dy, dx)
        return 0.0

class ObjectTrackerNode:
    def __init__(self):
        rospy.init_node('object_tracker_kf')
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.marker_pub = rospy.Publisher('/tracked_objects', MarkerArray, queue_size=10)
        self.tracked_objects = {}
        self.next_id = 1
        self.dist_threshold = 0.5  # meters, for matching clusters to tracks
        self.lost_threshold = 8    # frames before removing lost object

    def scan_callback(self, scan_msg):
        # Convert ranges to 2D points in robot frame
        angles = scan_msg.angle_min + np.arange(len(scan_msg.ranges)) * scan_msg.angle_increment
        xs = np.array(scan_msg.ranges) * np.cos(angles)
        ys = np.array(scan_msg.ranges) * np.sin(angles)
        points = np.vstack((xs, ys)).T

        # Simple clustering
        clusters = []
        used = np.zeros(points.shape[0], dtype=bool)
        for i, (x, y) in enumerate(points):
            if not np.isfinite(x) or not np.isfinite(y) or used[i]:
                continue
            cluster = [(x, y)]
            used[i] = True
            for j in range(i + 1, len(points)):
                if used[j]: continue
                dist = np.hypot(points[j][0] - x, points[j][1] - y)
                if dist < self.dist_threshold:
                    cluster.append((points[j][0], points[j][1]))
                    used[j] = True
            clusters.append(cluster)
        # Get cluster centroids
        centroids = []
        for c in clusters:
            arr = np.array(c)
            if len(arr) > 2:
                centroid = np.mean(arr, axis=0)
                centroids.append(centroid)
        # Predict step for all tracked objects
        for obj in self.tracked_objects.values():
            obj.predict()
            obj.lost += 1
        # Data association: Nearest neighbor
        updated_ids = set()
        for cx, cy in centroids:
            min_dist = float('inf')
            min_id = None
            for obj_id, obj in self.tracked_objects.items():
                dist = np.hypot(obj.x - cx, obj.y - cy)
                if dist < min_dist and dist < self.dist_threshold:
                    min_dist = dist
                    min_id = obj_id
            if min_id is not None:
                self.tracked_objects[min_id].update(cx, cy)
                updated_ids.add(min_id)
            else:
                obj = TrackedObject(self.next_id, cx, cy)
                self.tracked_objects[self.next_id] = obj
                updated_ids.add(self.next_id)
                self.next_id += 1
        # Remove lost tracks
        to_delete = [obj_id for obj_id, obj in self.tracked_objects.items() if obj.lost > self.lost_threshold]
        for obj_id in to_delete:
            del self.tracked_objects[obj_id]
        self.publish_markers(scan_msg.header.frame_id)

    def publish_markers(self, frame_id):
        marker_array = MarkerArray()
        for obj_id, obj in self.tracked_objects.items():
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = rospy.Time.now()
            m.ns = 'tracked_objects'
            m.id = obj_id
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = obj.x
            m.pose.position.y = obj.y
            m.pose.position.z = 0.5
            q = quaternion_from_euler(0, 0, obj.yaw)
            m.pose.orientation = Quaternion(*q)
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.5
            m.color.r = obj.color[0]
            m.color.g = obj.color[1]
            m.color.b = obj.color[2]
            m.color.a = 0.8
            marker_array.markers.append(m)
            # Text ID marker
            t = Marker()
            t.header.frame_id = frame_id
            t.header.stamp = rospy.Time.now()
            t.ns = 'tracked_ids'
            t.id = 1000 + obj_id
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = obj.x
            t.pose.position.y = obj.y
            t.pose.position.z = 1.2
            t.scale.z = 0.2
            t.color.r = 1.0
            t.color.g = 1.0
            t.color.b = 1.0
            t.color.a = 1.0
            t.text = str(obj_id)
            marker_array.markers.append(t)
        self.marker_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        ObjectTrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
