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

def create_image_sample(size='1'):

    try:
        image_module = importlib.import_module('sensor_msgs.msg')
        Image = getattr(image_module, 'Image')
    except (ImportError, AttributeError) as e:
        print(f"Error importing Image: {e}")
        Image = None

    try:
        cv_bridge_module = importlib.import_module('cv_bridge')
        CvBridge = getattr(cv_bridge_module, 'CvBridge')
    except (ImportError, AttributeError) as e:
        print(f"Error importing CvBridge: {e}")
        CvBridge = None

    size_dict = {
        1: 256,
        2: 512,
        3: 1024
    }

    bridge = CvBridge()
   
    img = np.zeros((size_dict.get(size), size_dict.get(size), 3), dtype=np.uint8)
    img[:50, :] = [255, 0, 0]
    img[50:, :] = [0, 255, 0]

    image = bridge.cv2_to_imgmsg(img, encoding='bgr8')
    image.data = img.flatten().tolist()

    return image.data

def repeat(text, times):
    result = ''
    i = 1

    while i <= times:
        result = result + text
        i = i + 1

    return result

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
            msg.data = [float(max_float32 / (4 - msg_size))]
        case 'Float64':
            max_float64 = sys.float_info.max
            msg.data = max_float64 / (4-msg_size)
        case 'Int32':
            max_int32 = (2**31) - 1
            msg.data = int(max_int32 / (4-msg_size))
        case 'Image':
            msg.data = create_image_sample(size=msg_size)
        case 'Float64MultiArray':
            msg.data = np.random.rand(data_size).tolist()
        case 'Imu':
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
        case 'PointCloud':
            geometry_module = importlib.import_module('geometry_msgs.msg')
            Point32 = getattr(geometry_module, 'Point32')
            points_1 = [
                Point32(x=1.0, y=2.0, z=3.0),
                Point32(x=4.0, y=5.0, z=6.0)
            ]
            points_2 = [
                Point32(x=7.0, y=8.0, z=9.0),
                Point32(x=10.0, y=11.0, z=12.0),
                Point32(x=13.0, y=14.0, z=15.0)
            ]
            points_3 = [
                Point32(x=16.0, y=17.0, z=18.0)
            ]

            if msg_size == 1:
                all_points = points_3
            elif msg_size == 2:
                all_points = points_1 + points_3
            elif msg_size == 3:
                all_points = points_1 + points_2 + points_3

            std_module = importlib.import_module('std_msgs.msg')
            Header = getattr(std_module,'Header')

            # Define header
            header = Header()
            header.frame_id = 'map'

            msg.header = header
            msg.points = all_points
        case 'PointCloud2':
            PointField = getattr(importlib.import_module('sensor_msgs.msg'), 'PointField')

            msg.height = 1
            msg.width = data_size // 16  # Adjusted for point representation
            msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            msg.is_bigendian = False
            msg.point_step = 12  # 3 floats (x, y, z) * 4 bytes each
            msg.row_step = msg.point_step * msg.width
            msg.data = np.random.rand(msg.width * 3).astype(np.float32).tobytes()
            msg.is_dense = True
        case 'LaserScan':
            msg.angle_min = -1.57
            msg.angle_max = 1.57
            msg.angle_increment = 3.14 / data_size
            msg.range_min = 0.1
            msg.range_max = 10.0
            msg.ranges = np.random.uniform(0.1, 10.0, data_size).tolist()
            msg.intensities = np.random.uniform(0.0, 255.0, data_size).tolist()
        case 'Pose':
            msg.position.x, msg.position.y, msg.position.z = 1.0, 2.0, 3.0
        case 'PoseStamped':
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = 1.0, 2.0, 3.0
        case 'Vector3':
            msg.x, msg.y, msg.z = float(data_size), float(data_size), float(data_size)
        case 'PoseWithCovariance':
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = 1.0, 2.0, 3.0
        case 'PoseWithCovarianceStamped':
            msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = 1.0, 2.0, 3.0
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
