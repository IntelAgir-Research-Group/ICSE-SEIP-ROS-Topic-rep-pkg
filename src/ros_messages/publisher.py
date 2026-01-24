# publisher.py
import rclpy
from rclpy.node import Node
import struct
from multiprocessing import shared_memory
from rclpy.serialization import deserialize_message
from shared_config import SHM_NAME, MAX_MSG_SIZE

def load_module(module_name):
    import importlib
    return importlib.import_module(module_name)

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
        msgs_module = load_module(msg_module_str)
        return getattr(msgs_module, msg_type_str)
    return None

class SharedMemoryPublisher(Node):

    def __init__(self, msg_type, topic):
        super().__init__("shm_publisher")
        self.publisher = self.create_publisher(msg_type, topic, 10)
        self.msg_type = msg_type
        self.shm = shared_memory.SharedMemory(name=SHM_NAME)
        self.timer = self.create_timer(0.01, self.publish_from_shm)
        self.last_size = 0

    def publish_from_shm(self):
        # Read message size
        size = struct.unpack("I", self.shm.buf[:4])[0]
        if size == 0:
            return
        raw = bytes(self.shm.buf[4:4+size])
        msg = deserialize_message(raw, self.msg_type)
        self.publisher.publish(msg)
        self.get_logger().info(f"Published message [{msg}] of size {size} bytes")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--message_type", type=str, required=True)
    args = parser.parse_args()

    msg_type = get_message_type(args.message_type)
    if not msg_type:
        print(f"Unsupported message type: {args.message_type}")
        return
    
    get_message_type(msg_type)

    rclpy.init()
    node = SharedMemoryPublisher(msg_type, f"{args.message_type.lower()}_topic")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        node.shm.close()


if __name__ == "__main__":
    main()
