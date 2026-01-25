
import argparse
import importlib
import multiprocessing as mp
import signal
import sys
import time
from dataclasses import dataclass

import numpy as np

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

def repeat(text, times):
    return text * times

def create_image_sample(size=1):
    try:
        Image = importlib.import_module('sensor_msgs.msg').Image
        CvBridge = importlib.import_module('cv_bridge').CvBridge
        has_cv_bridge = True
    except Exception:
        Image = importlib.import_module('sensor_msgs.msg').Image
        has_cv_bridge = False

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
    return msg

def create_imu(msg, msg_size):
    import random
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
    bytes_per_point = 12
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

def create_msg(msg_type: str, msg_size: int):
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

@dataclass
class Payload:
    """Container for serialized data + optional meta."""
    data: bytes
    # You can add fields like generation_time, seq, etc.

def producer_loop(q: mp.Queue, stop_evt: mp.Event, msg_type_str: str, msg_size: int, gen_rate_hz: float):
    """Generates messages, serializes to bytes, pushes into a bounded Queue."""
    from rclpy.serialization import serialize_message

    msg_class = get_message_type(msg_type_str)
    if msg_class is None:
        print(f"[Producer] Unsupported message type: {msg_type_str}", flush=True)
        return

    period = 1.0 / gen_rate_hz if gen_rate_hz > 0 else 0.0
    dropped = 0
    produced = 0

    print(f"[Producer] Starting. Type={msg_type_str}, size={msg_size}, rate={gen_rate_hz} Hz", flush=True)

    next_t = time.perf_counter()
    while not stop_evt.is_set():
        # Generate and serialize
        msg = create_msg(msg_type_str, msg_size)
        if msg is None:
            # Shouldn't happen unless type unsupported
            continue
        try:
            data = serialize_message(msg)
        except Exception as e:
            print(f"[Producer] Serialization failed: {e}", flush=True)
            continue

        # Put into queue with drop-oldest policy if full (to keep latency bounded)
        placed = False
        while not placed and not stop_evt.is_set():
            try:
                q.put_nowait(Payload(data=data))
                placed = True
                produced += 1
            except mp.queues.Full:
                # Drop one oldest then retry
                try:
                    q.get_nowait()
                    dropped += 1
                except Exception:
                    pass

        # pacing
        if period > 0:
            next_t += period
            sleep_dt = next_t - time.perf_counter()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                # running behind; catch up next iteration
                next_t = time.perf_counter()

    print(f"[Producer] Stopping. produced={produced}, dropped={dropped}", flush=True)

def publisher_loop(q: mp.Queue, stop_evt: mp.Event, msg_type_str: str, topic: str, pub_timer_period: float, use_sensor_qos: bool):
    """Consumes serialized bytes, deserializes to ROS messages, stamps and publishes."""
    import rclpy
    from rclpy.node import Node
    from rclpy.serialization import deserialize_message
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data

    msg_class = get_message_type(msg_type_str)
    if msg_class is None:
        print(f"[Publisher] Unsupported message type: {msg_type_str}", flush=True)
        return

    class PublisherNode(Node):
        def __init__(self):
            super().__init__('mp_publisher_node')
            qos = qos_profile_sensor_data if use_sensor_qos else QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.pub = self.create_publisher(msg_class, topic, qos)
            self.published = 0
            self.dropped = 0
            self.timer = self.create_timer(pub_timer_period, self.on_timer)

        def on_timer(self):
            # Drain queue quickly to reduce latency
            drained = 0
            while True:
                try:
                    payload: Payload = q.get_nowait()
                except Exception:
                    break
                try:
                    msg = deserialize_message(payload.data, msg_class)
                    # Restamp with ROS time if header exists
                    if hasattr(msg, "header"):
                        msg.header.stamp = self.get_clock().now().to_msg()
                    self.pub.publish(msg)
                    self.published += 1
                    drained += 1
                    self.get_logger().debug(f"Published #{self.published}")
                except Exception as e:
                    self.get_logger().error(f"Deser/Publish failed: {e}")
            if drained == 0:
                # (Optional) log rarely if idle
                pass

    def shutdown_handler(signum, frame):
        stop_evt.set()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    rclpy.init()
    node = PublisherNode()
    node.get_logger().info(f"Publishing {msg_type_str} on '{topic}' (timer={pub_timer_period:.3f}s, sensor_qos={use_sensor_qos})")

    try:
        while rclpy.ok() and not stop_evt.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Publisher exiting. published={node.published}")
        node.destroy_node()
        rclpy.shutdown()

def main():
    parser = argparse.ArgumentParser(description="ROS 2 multiprocess generator -> publisher")
    parser.add_argument('--execution_time', type=float, default=20.0, help='Total run time (s)')
    parser.add_argument('--message_type', type=str, required=True, help='ROS 2 message type (e.g., Image, PointCloud2)')
    parser.add_argument('--message_size', type=int, required=True, choices=[1, 2, 3], help='Message size tier')
    parser.add_argument('--topic', type=str, default=None, help='Topic name (default: <message_type_lower>_topic)')
    parser.add_argument('--gen_rate', type=float, default=10.0, help='Generation rate (Hz)')
    parser.add_argument('--pub_timer', type=float, default=0.01, help='Publisher timer period (s)')
    parser.add_argument('--queue_size', type=int, default=64, help='Bounded queue size between processes')
    parser.add_argument('--sensor_qos', action='store_true', help='Use qos_profile_sensor_data for publisher')
    args = parser.parse_args()

    topic = args.topic if args.topic else f"{args.message_type.lower()}_topic"

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    q: mp.Queue = mp.Queue(maxsize=args.queue_size)
    stop_evt = mp.Event()

    prod = mp.Process(
        target=producer_loop,
        args=(q, stop_evt, args.message_type, args.message_size, args.gen_rate),
        name="producer",
        daemon=True,
    )
    pub = mp.Process(
        target=publisher_loop,
        args=(q, stop_evt, args.message_type, topic, args.pub_timer, args.sensor_qos),
        name="publisher",
        daemon=True,
    )

    prod.start()
    pub.start()

    pid = pub.pid
    with open("/tmp/publisher.pid", "w") as f:
        f.write(str(pid))

    try:
        time.sleep(args.execution_time)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        prod.join(timeout=5)
        pub.join(timeout=5)

if __name__ == '__main__':
    main()