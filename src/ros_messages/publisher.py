
#!/usr/bin/env python3
import argparse
import importlib
import random
import sys
from time import time

import numpy as np
import rclpy
from rclpy.node import Node

def get_message_type(msg_type_str: str):
    msg_types = {
        'Image': 'sensor_msgs.msg',
        'Pose': 'geometry_msgs.msg',
        'PointCloud': 'sensor_msgs.msg',
        'PointCloud2': 'sensor_msgs.msg',
        'Imu': 'sensor_msgs.msg',
        'String': 'std_msgs.msg',
        'PoseStamped': 'geometry_msgs.msg',
        'LaserScan': 'sensor_msgs.msg',
        'Float64': 'std_msgs.msg',
        'Float32': 'std_msgs.msg',
        'Vector3': 'geometry_msgs.msg',
        'Float64MultiArray': 'std_msgs.msg',
        'PoseWithCovariance': 'geometry_msgs.msg',
        'PoseWithCovarianceStamped': 'geometry_msgs.msg',
        'Int32': 'std_msgs.msg',
        'Twist': 'geometry_msgs.msg',
        'JointState': 'sensor_msgs.msg',
    }
    msg_module_str = msg_types.get(msg_type_str)
    if not msg_module_str:
        return None
    msgs_module = importlib.import_module(msg_module_str)
    return getattr(msgs_module, msg_type_str, None)

# ---------------------------
# Message factories
# ---------------------------

def repeat(text, times):
    return text * times

def create_image_sample(size=1):
    # Try cv_bridge first
    try:
        Image = importlib.import_module('sensor_msgs.msg').Image
        CvBridge = importlib.import_module('cv_bridge').CvBridge
        has_cv_bridge = True
    except Exception:
        # Fallback to manual Image construction
        Image = importlib.import_module('sensor_msgs.msg').Image
        has_cv_bridge = False
        CvBridge = None  # noqa

    dim_map = {1: (86, 86), 2: (105, 105), 3: (148, 148)}
    dims = dim_map.get(size)
    if dims is None:
        raise ValueError("Size must be 1, 2, or 3")

    height, width = dims
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    if has_cv_bridge:
        bridge = CvBridge()
        image_msg = bridge.cv2_to_imgmsg(img, encoding='bgr8')
    else:
        # Manual construction if cv_bridge is not available
        image_msg = Image()
        image_msg.height = height
        image_msg.width = width
        image_msg.encoding = 'bgr8'
        image_msg.is_bigendian = False
        image_msg.step = width * 3
        image_msg.data = img.tobytes()

    image_msg.header.frame_id = "camera"
    return image_msg

def create_float64_multiarray(msg, size):
    max_elements = 8190
    size_map = {1: max_elements // 3, 2: max_elements // 2, 3: max_elements}
    num_elements = size_map.get(size)
    if num_elements is None:
        raise ValueError("Size must be '1', '2', or '3'")
    msg.data = np.random.rand(num_elements).astype(np.float64).tolist()
    # layout optional; skipped
    return msg

def create_imu(msg, msg_size):
    if msg_size >= 1:
        msg.angular_velocity.x = random.uniform(-1.0, 1.0)
        msg.linear_acceleration.x = random.uniform(-9.8, 9.8)
    if msg_size >= 2:
        msg.angular_velocity.y = random.uniform(-1.0, 1.0)
        msg.linear_acceleration.y = random.uniform(-9.8, 9.8)
    if msg_size >= 3:
        msg.angular_velocity.z = random.uniform(-1.0, 1.0)
        msg.linear_acceleration.z = random.uniform(-9.8, 9.8)
    msg.header.frame_id = "imu_link"
    return msg

def create_pointcloud(msg, msg_size):
    from sensor_msgs.msg import ChannelFloat32
    from geometry_msgs.msg import Point32

    bytes_per_point = 12
    max_bytes = 65536
    scale_map = {1: 1 / 3, 2: 1 / 2, 3: 1.0}
    scale = scale_map.get(msg_size, 1.0)
    num_points = int((max_bytes * scale) // bytes_per_point)

    msg.header.frame_id = "map"
    points_np = np.random.rand(num_points, 3).astype(np.float32)
    msg.points = [Point32(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points_np]

    channel = ChannelFloat32()
    channel.name = "intensity"
    channel.values = np.random.rand(num_points).astype(np.float32).tolist()
    msg.channels = [channel]
    return msg

def create_pointcloud2(msg, msg_size):
    from sensor_msgs.msg import PointField

    bytes_per_point = 12  # 3 * float32
    max_bytes = 65536
    scale_map = {1: 1 / 3, 2: 1 / 2, 3: 1.0}
    scale = scale_map.get(msg_size, 1.0)
    num_points = int((max_bytes * scale) // bytes_per_point)

    msg.header.frame_id = "map"
    msg.height = 1
    msg.width = num_points
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = bytes_per_point
    msg.row_step = msg.point_step * msg.width
    # Random xyz points
    msg.data = np.random.rand(num_points * 3).astype(np.float32).tobytes()
    msg.is_dense = True
    return msg

def create_laserscan(msg, msg_size):
    bytes_per_range = 8  # float64
    max_bytes = 65536
    scale_map = {1: 1 / 3, 2: 1 / 2, 3: 1.0}
    scale = scale_map.get(msg_size, 1.0)
    num_ranges = max(1, int((max_bytes * scale) // bytes_per_range))

    msg.header.frame_id = "laser"
    msg.angle_min = -1.57
    msg.angle_max = 1.57
    msg.angle_increment = (msg.angle_max - msg.angle_min) / num_ranges
    msg.time_increment = 0.0
    msg.scan_time = 0.1
    msg.range_min = 0.1
    msg.range_max = 10.0
    msg.ranges = np.random.uniform(0.1, 10.0, num_ranges).tolist()
    msg.intensities = np.random.uniform(0.0, 1.0, num_ranges).tolist()
    return msg

def create_pose(msg, msg_size):
    if msg_size >= 1:
        msg.position.x = 1.0
    if msg_size >= 2:
        msg.position.y = 2.0
    if msg_size >= 3:
        msg.position.z = 3.0
    return msg

def create_pose_stamped(msg, msg_size):
    if msg_size >= 1:
        msg.pose.position.x = 1.0
    if msg_size >= 2:
        msg.pose.position.y = 2.0
    if msg_size >= 3:
        msg.pose.position.z = 3.0
    msg.header.frame_id = "map"
    return msg

def create_vector3(msg, msg_size, data_size):
    if msg_size >= 1:
        msg.x = float(data_size)
    if msg_size >= 2:
        msg.y = float(data_size)
    if msg_size >= 3:
        msg.z = float(data_size)
    return msg

def create_pose_with_covariance(msg, msg_size):
    p = np.random.rand(3)
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = [float(v) for v in p]
    q = np.random.rand(4)
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = [float(v) for v in q]
    msg.covariance = np.random.rand(36).astype(np.float64).tolist()
    return msg

def create_pose_with_covariance_stamped(msg, msg_size):
    msg.header.frame_id = "map"
    # Fill the inner pose-with-covariance in place and return the stamped message
    create_pose_with_covariance(msg.pose, msg_size)
    return msg

def create_twist(msg, msg_size):
    if msg_size >= 1:
        msg.linear.x = 1.0
        msg.angular.x = 0.1
    if msg_size >= 2:
        msg.linear.y = 2.0
        msg.angular.y = 0.2
    if msg_size >= 3:
        msg.linear.z = 3.0
        msg.angular.z = 0.3
    return msg

def create_joint_state(msg, msg_size):
    num_joints_map = {1: 2, 2: 4, 3: 6}
    num_joints = num_joints_map.get(msg_size, 2)
    msg.name = [f'joint_{i}' for i in range(num_joints)]
    msg.position = np.random.uniform(-1.0, 1.0, num_joints).tolist()
    msg.velocity = np.random.uniform(-0.5, 0.5, num_joints).tolist()
    msg.effort = np.random.uniform(0.0, 1.0, num_joints).tolist()
    return msg

def create_vector3_stamped(msg, msg_size, data_size):
    if msg_size >= 1:
        msg.vector.x = np.random.uniform(0.0, float(data_size))
    if msg_size >= 2:
        msg.vector.y = np.random.uniform(0.0, float(data_size))
    if msg_size >= 3:
        msg.vector.z = np.random.uniform(0.0, float(data_size))
    msg.header.frame_id = "map"
    return msg

def create_twist_stamped(msg, msg_size):
    if msg_size >= 1:
        msg.twist.linear.x = np.random.uniform(0.0, 1.0)
        msg.twist.angular.x = np.random.uniform(0.0, 0.1)
    if msg_size >= 2:
        msg.twist.linear.y = np.random.uniform(0.0, 2.0)
        msg.twist.angular.y = np.random.uniform(0.0, 0.2)
    if msg_size >= 3:
        msg.twist.linear.z = np.random.uniform(0.0, 3.0)
        msg.twist.angular.z = np.random.uniform(0.0, 0.3)
    msg.header.frame_id = "map"
    return msg

def create_msg(msg_type: str, msg_size: int):
    # Simple validation to avoid zero division in scalar messages
    if msg_size not in (1, 2, 3):
        raise ValueError("message_size must be 1, 2, or 3")

    max_size = 1024
    data_size = {1: max_size // 3, 2: max_size // 2, 3: max_size}[msg_size]
    msg_class = get_message_type(msg_type)
    if msg_class is None:
        return None
    msg = msg_class()

    match msg_type:
        case 'String':
            msg.data = repeat('Hello, World! ', msg_size)
        case 'Float32':
            msg.data = 3.4e+38 / (5 - msg_size)  # denominators: 4,3,2
        case 'Float64':
            msg.data = sys.float_info.max / (4 - msg_size)  # denominators: 3,2,1
        case 'Int32':
            msg.data = (2**31 - 1) // (4 - msg_size)  # denominators: 3,2,1
        case 'Image':
            msg = create_image_sample(size=msg_size)
        case 'Float64MultiArray':
            msg = create_float64_multiarray(msg, msg_size)
        case 'Imu':
            msg = create_imu(msg, msg_size)
        case 'PointCloud':
            msg = create_pointcloud(msg, msg_size)
        case 'PointCloud2':
            msg = create_pointcloud2(msg, msg_size)
        case 'LaserScan':
            msg = create_laserscan(msg, msg_size)
        case 'Pose':
            msg = create_pose(msg, msg_size)
        case 'PoseStamped':
            msg = create_pose_stamped(msg, msg_size)
        case 'Vector3':
            msg = create_vector3(msg, msg_size, data_size)
        case 'PoseWithCovariance':
            msg = create_pose_with_covariance(msg, msg_size)
        case 'PoseWithCovarianceStamped':
            msg = create_pose_with_covariance_stamped(msg, msg_size)
        case 'Twist':
            msg = create_twist(msg, msg_size)
        case 'JointState':
            msg = create_joint_state(msg, msg_size)
        case 'Vector3Stamped':
            msg = create_vector3_stamped(msg, msg_size, data_size)
        case 'TwistStamped':
            msg = create_twist_stamped(msg, msg_size)
        case _:
            return None
    return msg

# ---------------------------
# Publisher node
# ---------------------------

class PublisherNode(Node):
    def __init__(self, msg_type, topic_name, msg_type_str, msg_size, interval):
        super().__init__('publisher_node')
        self.publisher_ = self.create_publisher(msg_type, topic_name, 10)
        self.timer = self.create_timer(interval, self.publish_message)
        self.msg_type_str = msg_type_str
        self.msg_size = msg_size
        self.get_logger().info(f'Publishing {msg_type.__name__} messages on {topic_name}')

    def publish_message(self):
        msg = create_msg(self.msg_type_str, self.msg_size)
        if msg is None:
            self.get_logger().warn(f"Failed to create message for type '{self.msg_type_str}'. Skipping publish.")
            return

        # Stamp with ROS time if the message has a header
        if hasattr(msg, "header"):
            msg.header.stamp = self.get_clock().now().to_msg()
        try:
            self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Publish failed: {e}")

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description='ROS 2 Message Publisher')
    parser.add_argument('--execution_time', type=int, default=60, help='Total execution time in seconds')
    parser.add_argument('--interval', type=float, default=1.0, help='Publishing interval in seconds')
    parser.add_argument('--message_type', type=str, required=True, help='Message type to publish')
    parser.add_argument('--message_size', type=int, required=True, choices=[1, 2, 3], help='Message size (1, 2, or 3)')
    parser.add_argument('--topic', type=str, default=None, help='Topic name (default: <message_type_lower>_topic)')
    args = parser.parse_args()

    msg_type = get_message_type(args.message_type)
    if not msg_type:
        print(f"Error: Unsupported message type '{args.message_type}'")
        return

    topic_name = args.topic if args.topic else f"{args.message_type.lower()}_topic"

    rclpy.init()
    node = PublisherNode(msg_type, topic_name, args.message_type, args.message_size, args.interval)

    start_time = time()
    try:
        while rclpy.ok() and (time() - start_time) < args.execution_time:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
