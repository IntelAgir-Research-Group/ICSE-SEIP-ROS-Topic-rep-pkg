# shared_config.py
from multiprocessing import shared_memory

# Maximum message size in bytes (tune for Images / PointCloud2)
MAX_MSG_SIZE = 1024 * 1024  # 1 MB

# Name of the shared memory block
SHM_NAME = "ros_msg_shm"
