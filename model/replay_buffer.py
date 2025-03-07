
from collections import deque
import pickle

SL_REPLAY_FILE = "sl_replay.pkl"
RL_REPLAY_FILE = "rl_replay.pkl"
REPLAY_BUFFER_SIZE = 999999  # Maximum number of stored training samples
MIN_REPLAY_SIZE = 1024  # Minimum samples before sampling from buffer
BATCH_SIZE = 1024  # Adjust based on your GPU memory

# In model/replay_buffer.py (critical fix)
def load_sl_buffer():
    try:
        with open(SL_REPLAY_FILE, "rb") as f:
            # Load buffer content while maintaining FIFO structure
            buffer_content = pickle.load(f)
            return deque(buffer_content, maxlen=REPLAY_BUFFER_SIZE)
    except FileNotFoundError:
        return deque(maxlen=REPLAY_BUFFER_SIZE)

def save_sl_buffer(buffer):
    with open(SL_REPLAY_FILE, "wb") as f:
        # Save only the newest entries up to buffer capacity
        pickle.dump(list(buffer)[-REPLAY_BUFFER_SIZE:], f)


def load_rl_buffer():
    try:
        with open(RL_REPLAY_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return deque(maxlen=REPLAY_BUFFER_SIZE)

def save_rl_buffer(buffer):
    with open(RL_REPLAY_FILE, "wb") as f:
        pickle.dump(buffer, f)