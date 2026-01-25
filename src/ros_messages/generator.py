# generator.py
import importlib
import time
import struct
import argparse
from multiprocessing import shared_memory
import numpy as np
from rclpy.serialization import serialize_message
from shared_config import SHM_NAME, MAX_MSG_SIZE
import random

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
        'Twist': 'geometry_msgs.msg',
        'JointState': 'sensor_msgs.msg'
    }

    msg_module_str = msg_types.get(msg_type_str, None)
    if msg_module_str:
        msgs_module = importlib.import_module(msg_module_str)
        return getattr(msgs_module, msg_type_str)
    return None


def create_image_sample(size=1):
    try:
        Image = importlib.import_module('sensor_msgs.msg').Image
        CvBridge = importlib.import_module('cv_bridge').CvBridge
    except Exception as e:
        print(f"Error importing ROS or cv_bridge modules: {e}")
        return None

    dim_map = {
        1: (86, 86),
        2: (105, 105),
        3: (148, 148)
    }

    dims = dim_map.get(size)
    if dims is None:
        raise ValueError("Size must be 1, 2, or 3")

    height, width = dims
    # Fill image with random RGB values
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    bridge = CvBridge()
    image_msg = bridge.cv2_to_imgmsg(img, encoding='bgr8')
    image_msg.header.frame_id = "camera"
    return image_msg


def repeat(text, times):
    return text * times


def create_float64_multiarray(msg, size):
    max_elements = 8190
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
    if msg_size >= 1:
        msg.angular_velocity.x = random.uniform(-1.0, 1.0)
    if msg_size >= 2:
        msg.angular_velocity.y = random.uniform(-1.0, 1.0)
    if msg_size >= 3:
        msg.angular_velocity.z = random.uniform(-1.0, 1.0)

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

    bytes_per_point = 12
    max_bytes = 65536
    scale_map = {1: 1 / 3, 2: 1 / 2, 3: 1.0}
    scale = scale_map.get(msg_size)
    num_points = int((max_bytes * scale) // bytes_per_point)

    msg.header.frame_id = "map"
    points_np = np.random.rand(num_points, 3).astype(np.float32)
    msg.points = [Point32(x=p[0], y=p[1], z=p[2]) for p in points_np]

    channel = ChannelFloat32()
    channel.name = "intensity"
    channel.values = np.random.rand(num_points).astype(np.float32).tolist()
    msg.channels = [channel]
    return msg


def create_pointcloud2(msg, msg_size):
    from sensor_msgs.msg import PointField

    bytes_per_point = 12
    max_bytes = 65536
    scale_map = {1: 1 / 3, 2: 1 / 2, 3: 1.0}
    scale = scale_map.get(msg_size)
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
    bytes_per_range = 8
    max_bytes = 65536
    scale_map = {1: 1 / 3, 2: 1 / 2, 3: 1.0}
    scale = scale_map.get(msg_size)
    num_ranges = int((max_bytes * scale) // bytes_per_range)

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
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = p
    q = np.random.rand(4)
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = q
    msg.covariance = np.random.rand(36).astype(np.float64).tolist()
    return msg

def create_pose_with_covariance_stamped(msg, msg_size):
    msg.header.stamp.sec = int(time())
    msg.header.stamp.nanosec = int((time() % 1) * 1e9)
    msg.header.frame_id = "map"
    return create_pose_with_covariance(msg.pose, msg_size)

def create_twist(msg, msg_size):
    if msg_size >= 1:
        msg.linear.x = 1.0
    if msg_size >= 2:
        msg.linear.y = 2.0
    if msg_size >= 3:
        msg.linear.z = 3.0

    if msg_size >= 1:
        msg.angular.x = 0.1
    if msg_size >= 2:
        msg.angular.y = 0.2
    if msg_size >= 3:
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

def create_msg(msg_type, msg_size):
    max_size = 1024
    data_size = {1: max_size // 3, 2: max_size // 2, 3: max_size}.get(msg_size, max_size)
    msg_class = get_message_type(msg_type)
    msg = msg_class()

    match msg_type:
        case 'String':
            msg.data = repeat('Hello, World! ', msg_size)
        case 'Float32':
            msg.data = 3.4e+38 / (5 - msg_size)
        case 'Float64':
            msg.data = sys.float_info.max / (4 - msg_size)
        case 'Int32':
            msg.data = (2**31 - 1) // (4 - msg_size)
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
        case _:
            return None
    return msg

def generator_loop(msg_type_str, msg_size, interval):
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=4 + MAX_MSG_SIZE)
    
    try:
        while True:
            msg = create_msg(msg_type_str, msg_size)
            # Serialize ROS message to bytes
            raw = serialize_message(msg)
            size = len(raw)
            if size > MAX_MSG_SIZE:
                raise ValueError(f"Message too large: {size} bytes")
            
            # Write size (4 bytes) + payload
            shm.buf[:4] = struct.pack("I", size)
            shm.buf[4:4+size] = raw

            # Wait for next iteration
            time.sleep(interval)
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message_type", type=str, required=True)
    parser.add_argument("--message_size", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()

    generator_loop(args.message_type, args.message_size, args.interval)
