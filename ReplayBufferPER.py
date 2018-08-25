import numpy as np

from SumTree import SumTree

class ReplayBufferPER:
    """Prioritized Experience Replay (PER) buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, alpha=0.5, beta=0.4, beta_increment=0.001, epsilon=0.01):
        """Initialize a PER Buffer object.

        Params
        ======
            buffer_size     : maximum size of buffer
            batch_size      : size of each training batch
            alpha           : Hyperparameter to compute priorities for samling [0-1] : 0=> Uniform sampling 1=>Prioritized sampling
            beta            : Hyperparameter used in importance-sampling - Initial value increased to 1
            beta_increment  : Hyperparameter to increment beta at each sampling
            epsilon         : Hyperparameter used to avoid some experience to have 0 probability
        """
        self.tree = SumTree(buffer_size)

        self.batch_size = batch_size

        # PER Hyperparameters
        self.alpha          = alpha
        self.beta           = beta
        self.beta_increment = beta_increment
        self.epsilon        = epsilon

        np.random.seed(seed)
    
    def _get_priority(self, error):
        """Get the priority for this error"""
        return (abs(error) + self.epsilon) ** self.alpha

    def add(self, error, experience):
        """Add a new experience to memory.

        Params
        ======
            error       : Current error of this experience 
            experience  : The experience to add (most often a tuple <s, r, a, s'>)
        """
        self.tree.add(self._get_priority(error), experience)

    def update_priorities(self, indexes, errors):
        """Update the priorities of experiences at indexes

        Params
        ======
            indexes (numpy): The indexes to update
            errors  (numpy): The errors of the experiences at indexes
        """
        for i,error in zip(indexes, errors):
            self.tree.update_tree(i, self._get_priority(error))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""


        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority() / self.batch_size       # priority segment        


        indexes     = [0  for _ in range(self.batch_size)]
        experiences = [() for _ in range(self.batch_size)]
        IS_weights  = [0  for _ in range(self.batch_size)]

        # To Compute the importance sampling we need the max of it, to do so we need the min priority
        prob_min = self.tree.min_priority() / self.tree.total_priority()
        max_weight_IS = (prob_min * self.batch_size) ** (-self.beta)

        # Sample uniformely from each segment
        for i in range(self.batch_size):

            # Segment bounds
            low  = priority_segment * i
            high = low + priority_segment

            value = np.random.uniform(low, high)

            # Retrieve experience from the tree
            tree_index, priority, experience = self.tree.get_information(value)

            # P(j) : Sampling proba = pj/Sum(pi)
            sampling_proba = priority / self.tree.total_priority()
            # Importance sampling weight
            weight_IS = (sampling_proba * self.batch_size) ** (-self.beta)
            weight_IS /= max_weight_IS

            indexes[i]     = tree_index
            experiences[i] = experience
            IS_weights[i]  = weight_IS


        self.beta = min(self.beta + self.beta_increment , 1.)

        return experiences, indexes, IS_weights

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.tree)