import random
from collections import deque

class ReplayBuffer:
    """Fixed-size buffer to store experiences."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory     = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        random.seed(seed)
    
    def add(self, experience):
        """Add a new experience to memory."""
        self.memory.append(experience)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        return experiences

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
