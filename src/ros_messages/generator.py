# generator.py
import time
import struct
import argparse
from multiprocessing import shared_memory
import numpy as np
from rclpy.serialization import serialize_message
from your_message_module import create_msg  # import your create_msg function
from shared_config import SHM_NAME, MAX_MSG_SIZE

def generator_loop(msg_type_str, msg_size, interval):
    # Create shared memory
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=4 + MAX_MSG_SIZE)
    
    try:
        while True:
            # Create ROS message
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
