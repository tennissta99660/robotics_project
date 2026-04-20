import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'z', 'done', 'action', 'next_state'))


class Memory:
    def __init__(self, buffer_size, seed):
        self.buffer_size = buffer_size
        # FIX: use deque with maxlen instead of list + pop(0).
        # list.pop(0) is O(n) — for a 1M buffer this is catastrophically slow.
        # deque with maxlen automatically evicts the oldest entry in O(1).
        self.buffer = deque(maxlen=buffer_size)
        self.seed = seed
        random.seed(self.seed)

    def add(self, *transition):
        self.buffer.append(Transition(*transition))
        # No manual pop needed — deque handles eviction automatically.

    def sample(self, size):
        return random.sample(self.buffer, size)

    def __len__(self):
        return len(self.buffer)

    @staticmethod
    def get_rng_state():
        return random.getstate()

    @staticmethod
    def set_rng_state(random_rng_state):
        random.setstate(random_rng_state)
