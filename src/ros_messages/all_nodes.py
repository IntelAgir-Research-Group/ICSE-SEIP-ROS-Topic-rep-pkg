import subprocess

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32, Float64, Float64MultiArray, Int32
from sensor_msgs.msg import Image, PointCloud, PointCloud2, Imu, CameraInfo, Joy, NavSatFix, LaserScan, BatteryState, JointState
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped, Vector3, Vector3Stamped
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import argparse
from setproctitle import setproctitle

setproctitle("pubsub")


def create_publisher_node(msg_type, topic_name, msg_sample, interval):
    class PublisherNode(Node):
        def __init__(self):
            super().__init__('talker')
            self.publisher_ = self.create_publisher(msg_type, topic_name, 10)
            self.timer = self.create_timer(interval, self.publish_message)
            self.get_logger().info(f'Publishing {msg_type.__name__} messages on {topic_name}')

        def publish_message(self):
            self.publisher_.publish(msg_sample)

    return PublisherNode

def create_listener_node(msg_type, topic_name):
    class ListenerNode(Node):
        def __init__(self):
            super().__init__(f'listener')
            self.subscription = self.create_subscription(
                msg_type,
                topic_name,
                self.listener_callback,
                10)
            self.get_logger().info(f'Listening to {msg_type.__name__} messages on {topic_name}')

        def listener_callback(self, msg):
            self.get_logger().info(f'Received message on {topic_name}: {msg}')

    return ListenerNode

def get_message_type(msg_type_str):
    msg_types = {
        'Image': Image,
        'Odometry': Odometry,
        'Pose': Pose,
        'PointCloud': PointCloud,
        'PointCloud2': PointCloud2,
        'Imu': Imu,
        'JointState': JointState,
        'String': String,
        'PoseStamped': PoseStamped,
        'Marker': Marker,
        'LaserScan': LaserScan,
        'Bool': Bool,
        'Path': Path,
        'Float64': Float64,
        'MarkerArray': MarkerArray,
        'NavSatFix': NavSatFix,
        'Float32': Float32,
        'CameraInfo': CameraInfo,
        'Vector3': Vector3,
        'Float64MultiArray': Float64MultiArray,
        'Joy': Joy,
        'PoseWithCovariance': PoseWithCovariance,
        'PoseWithCovarianceStamped': PoseWithCovarianceStamped,
        'OccupancyGrid': OccupancyGrid,
        'Vector3Stamped': Vector3Stamped,
        'BatteryState': BatteryState,
        'Int32': Int32,
    }
    return msg_types.get(msg_type_str, None)

# Image
def create_image_sample():
    image = Image()
    image.height = 480
    image.width = 640
    image.encoding = 'rgb8'
    image.step = image.width * 3
    image.data = np.random.randint(0, 255, (image.height, image.width, 3), dtype=np.uint8).tobytes()
    return image

def main():
    parser = argparse.ArgumentParser(description='ROS 2 Message Publisher')
    parser.add_argument('--execution_time', type=int, default=60, help='Total execution time in seconds')
    parser.add_argument('--interval', type=float, default=1.0, help='Publishing interval in seconds')
    parser.add_argument('--message_type', type=str, required=True, help='Message type to publish and listen')
    args = parser.parse_args()

    msg_type = get_message_type(args.message_type)
    if not msg_type:
        print(f"Error: Unsupported message type '{args.message_type}'")
        return

    topic_name = f"{args.message_type.lower()}_topic"
    msg_sample = msg_type()

    if args.message_type == 'Image':
        msg_sample = create_image_sample()
    else:
        msg_sample = msg_type()

    rclpy.init()

    pub_node_class = create_publisher_node(msg_type, topic_name, msg_sample, args.interval)
    pub_node = pub_node_class()

    listener_node_class = create_listener_node(msg_type, topic_name)
    listener_node = listener_node_class()

    start_time = time.time()
    try:
        while rclpy.ok() and (time.time() - start_time) < args.execution_time:
            rclpy.spin_once(pub_node, timeout_sec=0.1)
            rclpy.spin_once(listener_node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pub_node.destroy_node()
        listener_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
