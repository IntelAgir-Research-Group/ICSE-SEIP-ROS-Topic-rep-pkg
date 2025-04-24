import rclpy
from rclpy.node import Node
import argparse
import numpy as np
from time import time
from setproctitle import setproctitle
import importlib
import struct
import sys
import cv2
import random

setproctitle("publisher")

def create_publisher_node(msg_type, topic_name, msg_sample, interval):
    class PublisherNode(Node):
        def __init__(self):
            super().__init__('talker')
            self.publisher_ = self.create_publisher(msg_type, topic_name, 10)
            self.timer = self.create_timer(interval, self.publish_message)
            self.get_logger().info(f'Publishing {msg_type.__name__} messages on {topic_name}')

        def publish_message(self):
            now = time()
            msg_sample.header.stamp.sec = int(now)
            msg_sample.header.stamp.nanosec = int((now % 1) * 1e9)
            self.publisher_.publish(msg_sample)

    return PublisherNode

def get_message_type(msg_type_str):
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
    }
    
    msg_module_str = msg_types.get(msg_type_str, None)
    if msg_module_str:
        msgs_module = importlib.import_module(msg_module_str)
        return getattr(msgs_module, msg_type_str)
    return None

def create_image_sample(size=1):
    import numpy as np
    import importlib

    # Import Image and CvBridge
    try:
        Image = importlib.import_module('sensor_msgs.msg').Image
    except (ImportError, AttributeError) as e:
        print(f"Error importing sensor_msgs.msg.Image: {e}")
        return None

    try:
        CvBridge = importlib.import_module('cv_bridge').CvBridge
    except (ImportError, AttributeError) as e:
        print(f"Error importing CvBridge: {e}")
        return None

    # Define compatible dimensions for ~22KB, ~33KB, ~66KB
    dim_map = {
        1: (86, 86),    # ~22 KB
        2: (105, 105),  # ~33 KB
        3: (148, 148)   # ~66 KB
    }

    dims = dim_map.get(size)
    if dims is None:
        raise ValueError("Size must be 1, 2, or 3")

    height, width = dims

    # Create an RGB image (uint8)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:height // 2, :] = [255, 0, 0]   # Red top
    img[height // 2:, :] = [0, 255, 0]   # Green bottom

    # Convert to ROS Image message
    bridge = CvBridge()
    image_msg = bridge.cv2_to_imgmsg(img, encoding='bgr8')
    image_msg.header.frame_id = "camera"

    return image_msg

def repeat(text, times):
    result = ''
    i = 1

    while i <= times:
        result = result + text
        i = i + 1

    return result

def create_float64_multiarray(msg, size):
    max_elements = 8190  # 8190 × 8 = ~64 KB

    size_map = {
        1: max_elements // 3,
        2: max_elements // 2,
        3: max_elements
    }

    num_elements = size_map.get(size)
    if num_elements is None:
        raise ValueError("Size must be '1', '2', or '3'")

    msg.data = np.random.rand(num_elements).astype(np.float64).tolist()

    return msg

def create_imu(msg, msg_size):
    # Fake angular velocity (rad/s)
    if msg_size >= 1:
        msg.angular_velocity.x = random.uniform(-1.0, 1.0)
    if msg_size >= 2:
        msg.angular_velocity.y = random.uniform(-1.0, 1.0)
    if msg_size >= 3:
        msg.angular_velocity.z = random.uniform(-1.0, 1.0)

    # Fake linear acceleration (m/s^2)
    if msg_size >= 1:
        msg.linear_acceleration.x = random.uniform(-9.8, 9.8)
    if msg_size >= 2:
        msg.linear_acceleration.y = random.uniform(-9.8, 9.8)
    if msg_size >= 3:
        msg.linear_acceleration.z = random.uniform(-9.8, 9.8)

    return msg

def create_pointcloud(msg, msg_size):
    from sensor_msgs.msg import ChannelFloat32
    from geometry_msgs.msg import Point32
    import numpy as np

    bytes_per_point = 12  # 3 x float32
    max_bytes = 65536

    scale_map = {
        1: 1/3,
        2: 1/2,
        3: 1.0
    }

    scale = scale_map.get(msg_size)
    if scale is None:
        raise ValueError("Size must be 1, 2, or 3")

    num_points = int((max_bytes * scale) // bytes_per_point)

    msg.header.frame_id = "map"

    points_np = np.random.rand(num_points, 3).astype(np.float32)
    msg.points = [Point32(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points_np]

    channel = ChannelFloat32()
    channel.name = "intensity"
    channel.values = [float(v) for v in np.random.rand(num_points).astype(np.float32)]
    msg.channels = [channel]

    return msg

def create_pointcloud2(msg, msg_size):
    from sensor_msgs.msg import PointField
    import numpy as np

    bytes_per_point = 12  # 3 x float32 (x, y, z)
    max_bytes = 65536

    scale_map = {
        1: 1/3,
        2: 1/2,
        3: 1.0
    }

    scale = scale_map.get(msg_size)
    if scale is None:
        raise ValueError("Size must be 1, 2, or 3")

    num_points = int((max_bytes * scale) // bytes_per_point)

    msg.header.frame_id = "map"
    msg.height = 1
    msg.width = num_points
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]
    msg.is_bigendian = False
    msg.point_step = bytes_per_point
    msg.row_step = msg.point_step * msg.width
    msg.data = np.random.rand(num_points * 3).astype(np.float32).tobytes()
    msg.is_dense = True

    return msg

def create_laserscan(msg, msg_size):
    import numpy as np

    bytes_per_range = 8  # 1 float32 for range, 1 for intensity
    max_bytes = 65536

    scale_map = {
        1: 1/3,
        2: 1/2,
        3: 1.0
    }

    scale = scale_map.get(msg_size)
    if scale is None:
        raise ValueError("Size must be 1, 2, or 3")

    num_ranges = int((max_bytes * scale) // bytes_per_range)

    msg.header.frame_id = "laser"
    msg.angle_min = -1.57
    msg.angle_max = 1.57
    msg.angle_increment = (msg.angle_max - msg.angle_min) / num_ranges
    msg.time_increment = 0.0
    msg.scan_time = 0.1
    msg.range_min = 0.1
    msg.range_max = 10.0

    msg.ranges = np.random.uniform(0.1, 10.0, num_ranges).astype(np.float32).tolist()
    msg.intensities = np.random.uniform(0.0, 1.0, num_ranges).astype(np.float32).tolist()

    return msg

def create_pose(msg, msg_size):
    if msg_size >= 1:
        msg.position.x = 1.0
    if msg_size >= 2:
        msg.position.x, msg.position.y = 1.0, 2.0
    if msg_size >= 3:
        msg.position.x, msg.position.y, msg.position.z = 1.0, 2.0, 3.0
    
    return msg

def create_pose_stamped(msg, msg_size):
    if msg_size >= 1:
        msg.pose.position.x = 1.0
    if msg_size >= 2:
        msg.pose.position.x, msg.pose.position.y = 1.0, 2.0
    if msg_size >= 3:
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = 1.0, 2.0, 3.0
    
    return msg

def create_vector3(msg, msg_size, data_size):
    if msg_size >= 1:
        msg.x = float(data_size)
    if msg_size >= 2:
        msg.x, msg.y = float(data_size), float(data_size)
    if msg_size >= 3:
        msg.x, msg.y, msg.z = float(data_size), float(data_size), float(data_size)
    
    return msg
    
def create_pose_with_covariance(msg, msg_size):
    msg_list = []

    max_count = 190  # 190 * 344 bytes ≈ 65 KB

    size_map = {
        1: max_count // 3,
        2: max_count // 2,
        3: max_count
    }

    num_msgs = size_map.get(msg_size)
    if num_msgs is None:
        raise ValueError("Size must be '1', '2', or '3'")

    for _ in range(num_msgs):
        # Random position
        p = np.random.rand(3)
        msg.pose.position.x = float(p[0])
        msg.pose.position.y = float(p[1])
        msg.pose.position.z = float(p[2])

        # Random orientation
        q = np.random.rand(4)
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])

        # Random covariance
        msg.covariance = np.random.rand(36).astype(np.float64).tolist()

        msg_list.append(msg)

    return msg

def create_pose_with_covariance_stamped(msg, msg_size):
    from std_msgs.msg import Header
    import time

    msg_list = []

    max_count = 178  # 178 * ~368 bytes ≈ 64 KB

    size_map = {
        1: max_count // 3,
        2: max_count // 2,
        3: max_count
    }

    num_msgs = size_map.get(msg_size)
    if num_msgs is None:
        raise ValueError("Size must be '1', '2', or '3'")

    for i in range(num_msgs):
        # Header
        msg.header.stamp.sec = int(time.time())
        msg.header.stamp.nanosec = int((time.time() % 1) * 1e9)
        msg.header.frame_id = "map"

        # Random position
        p = np.random.rand(3)
        msg.pose.pose.position.x = float(p[0])
        msg.pose.pose.position.y = float(p[1])
        msg.pose.pose.position.z = float(p[2])

        # Random orientation
        q = np.random.rand(4)
        msg.pose.pose.orientation.x = float(q[0])
        msg.pose.pose.orientation.y = float(q[1])
        msg.pose.pose.orientation.z = float(q[2])
        msg.pose.pose.orientation.w = float(q[3])

        # Random covariance
        msg.pose.covariance = np.random.rand(36).astype(np.float64).tolist()

        msg_list.append(msg)

    return msg

def create_msg(msg_type, msg_size):

    max_size = 1024  # Define a generic max size (adjust as needed per message type)
    data_size = {1: max_size // 3, 2: max_size // 2, 3: max_size}.get(msg_size, max_size)


    # Msg Type
    msg_class = get_message_type(msg_type)
    msg = msg_class()

    match msg_type:
        case 'String':
            msg.data = repeat('Hello, World! ', msg_size)
        case 'Float32':
            SAFE_FLOAT32_MAX = 3.4028234e+38  # ligeiramente abaixo do máximo permitido
            msg.data = float(SAFE_FLOAT32_MAX / (5 - msg_size))
        case 'Float64':
            max_float64 = sys.float_info.max
            msg.data = max_float64 / (4-msg_size)
        case 'Int32':
            max_int32 = (2**31) - 1
            msg.data = int(max_int32 / (4-msg_size))
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
        case _:
            return None
    return msg

def main():
    parser = argparse.ArgumentParser(description='ROS 2 Message Publisher')
    parser.add_argument('--execution_time', type=int, default=60, help='Total execution time in seconds')
    parser.add_argument('--interval', type=float, default=1.0, help='Publishing interval in seconds')
    parser.add_argument('--message_type', type=str, required=True, help='Message type to publish')
    parser.add_argument('--message_size', type=int, required=True, help='Message size')
    args = parser.parse_args()

    msg_type = get_message_type(args.message_type)
    if not msg_type:
        print(f"Error: Unsupported message type '{args.message_type}'")
        return

    topic_name = f"{args.message_type.lower()}_topic"

    msg_sample = create_msg(args.message_type, args.message_size)
    # msg_type_str = msg_type()
    # msg_type_str.data = msg_sample

    rclpy.init()

    pub_node_class = create_publisher_node(msg_type, topic_name, msg_sample, args.interval)
    pub_node = pub_node_class()

    start_time = time()
    try:
        while rclpy.ok() and (time() - start_time) < args.execution_time:
            rclpy.spin_once(pub_node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pub_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
