import os
import rclpy
from rclpy.node import Node
import argparse
from time import time
from setproctitle import setproctitle
import importlib
import sys

setproctitle("subscriber")

TIMEOUT_SECONDS = 5  # Adjust timeout duration as needed

def create_listener_node(msg_type, topic_name):
    class ListenerNode(Node):
        def __init__(self):
            super().__init__('listener')

            self.pid = os.getpid()  # <-- get the PID
            self.get_logger().info(f'Listener PID: {self.pid}')
            
            with open("/tmp/listener.pid", "w") as f:
                f.write(str(self.pid))

            self.subscription = self.create_subscription(
                msg_type,
                topic_name,
                self.listener_callback,
                10)
            self.get_logger().info(f'Listening to {msg_type.__name__} messages on {topic_name}')
            
            self.last_msg_time = time()
            self.timer = self.create_timer(1.0, self.check_timeout)

        def listener_callback(self, msg):
            self.get_logger().info(f'Received message on {topic_name}: {msg}')
            self.last_msg_time = time()  # Reset timeout timer

        def check_timeout(self):
            if time() - self.last_msg_time > TIMEOUT_SECONDS:
                self.get_logger().info(f"No messages received for {TIMEOUT_SECONDS} seconds. Shutting down.")
                #rclpy.shutdown()
                sys.exit(1)

    return ListenerNode

def get_message_type(msg_type_str):
    msg_types = {
        'Image': 'sensor_msgs.msg',
        'Odometry': 'nav_msgs.msg',
        'Pose': 'geometry_msgs.msg',
        'PointCloud': 'sensor_msgs.msg',
        'PointCloud2': 'sensor_msgs.msg',
        'Imu': 'sensor_msgs.msg',
        'JointState': 'sensor_msgs.msg',
        'String': 'std_msgs.msg',
        'PoseStamped': 'geometry_msgs.msg',
        'Marker': 'visualization_msgs.msg',
        'LaserScan': 'sensor_msgs.msg',
        'Bool': 'std_msgs.msg',
        'Path': 'nav_msgs.msg',
        'Float64': 'std_msgs.msg',
        'MarkerArray': 'visualization_msgs.msg',
        'NavSatFix': 'sensor_msgs.msg',
        'Float32': 'std_msgs.msg',
        'CameraInfo': 'sensor_msgs.msg',
        'Vector3': 'geometry_msgs.msg',
        'Float64MultiArray': 'std_msgs.msg',
        'Joy': 'sensor_msgs.msg',
        'PoseWithCovariance': 'geometry_msgs.msg',
        'PoseWithCovarianceStamped': 'geometry_msgs.msg',
        'OccupancyGrid': 'nav_msgs.msg',
        'Vector3Stamped': 'geometry_msgs.msg',
        'BatteryState': 'sensor_msgs.msg',
        'Int32': 'std_msgs.msg',
        'Twist': 'geometry_msgs.msg',
        'JointState': 'sensor_msgs.msg',
    }
    msg_module_str = msg_types.get(msg_type_str, None)
    if msg_module_str:
        #print(f'Module: {msg_module_str}')
        msgs_module = importlib.import_module(msg_module_str)
        return getattr(msgs_module, msg_type_str)
    return None

def main():
    parser = argparse.ArgumentParser(description='ROS 2 Message Subscriber')
    parser.add_argument('--message_type', type=str, required=True, help='Message type to subscribe to')
    args = parser.parse_args()

    msg_type = get_message_type(args.message_type)
    if not msg_type:
        print(f"Error: Unsupported message type '{args.message_type}'")
        return

    topic_name = f"{args.message_type.lower()}_topic"

    rclpy.init()

    listener_node_class = create_listener_node(msg_type, topic_name)
    listener_node = listener_node_class()

    try:
        rclpy.spin(listener_node)
    except KeyboardInterrupt:
        pass
    finally:
        listener_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
