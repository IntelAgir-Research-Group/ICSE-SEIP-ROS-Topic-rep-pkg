# publisher.py
import rclpy
from rclpy.node import Node
import struct
from multiprocessing import shared_memory
from rclpy.serialization import deserialize_message
from your_message_module import get_message_type  # import your get_message_type function
from shared_config import SHM_NAME, MAX_MSG_SIZE

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
        if size == 0 or size == self.last_size:
            return  # no new message

        self.last_size = size
        raw = bytes(self.shm.buf[4:4+size])
        msg = deserialize_message(raw, self.msg_type)
        self.publisher.publish(msg)
        self.get_logger().info(f"Published message of size {size} bytes")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--message_type", type=str, required=True)
    args = parser.parse_args()

    msg_type = get_message_type(args.message_type)
    if not msg_type:
        print(f"Unsupported message type: {args.message_type}")
        return

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
