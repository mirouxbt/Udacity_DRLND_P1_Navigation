import numpy as np

class SumTree:
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    def __init__(self, capacity):
        """ Initialize the tree with all nodes = 0"""

        self.capacity = capacity # Number of leaf nodes (final nodes) that contains data
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation of number of nodes
        # Remember we have a binary tree, so number of leaf nodes = 2^n ( n = depth of tree)
        # So all nodes before the lead nodes = sum(2^k) for k in [0, n-1] which is equal to (2^n)-1
        # So the total number of nodes is 2^n + 2^n -1 = 2*(2^n)-1
        # Here our capacity = 2^n
        self.tree = np.zeros(2 * capacity - 1)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

        # Start at index o and fill from left to right
        self.data_index = 0

        # Length of the data
        self.length = 0

    def _tree_index(self, data_index):
        """Get the tree_index based on the data index
           The leaf nodes will be at the end of the tree array
           So the index start at (capacity-1)
        """
        return data_index + self.capacity - 1

    def _data_index(self, tree_index):
        """Get the data_index based on the tree index
           The leaf nodes will be at the end of the tree array
           So the index start at (capacity-1)
        """
        return tree_index - self.capacity + 1

    def add(self, priority, data):
        """Add our priority score in the sumtree leaf
           We fill the leaves from left to right like the data
           and overwrite whatever was there
        """
        # Update data frame
        self.data[self.data_index] = data
        
        # Update the leaf
        self.update_tree(self._tree_index(self.data_index), priority)
        
        # Add 1 to data_pointer and start over if needed
        self.data_index += 1
        self.data_index %= self.capacity

        # Length increased
        self.length = min(self.length + 1, self.capacity)
        
    
    def update_tree(self, tree_index, priority):
        """Update the leaf priority score and propagate the change through tree

            To find the parent node index just divide by 2 the (node index -1) and discard decimal

            The tree structure in array (node index) is [0,1,2,3,4,5,6] for example 
            Child of node 1 is 3 and 4, so child are at (parent index * 2) + 1 
        """
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]

        # Update the node
        self.tree[tree_index] = priority
        
        # then propagate the change through tree until the root
        while tree_index != 0:
            # Parent index
            tree_index = (tree_index - 1) // 2
            # Apply the change
            self.tree[tree_index] += change
    
    
    def get_information(self, value):
        """ Get information for the value required
            
            Params
            ======
                Value : Value to search in the priority tree using the algorithm at
                        https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/

                        Each priority take a range in the cumulative sum of priority, so bigger priorites have more
                        odds to be selected

            Return the leaf_index, priority value of that leaf and experience associated with that index
        """

        # Start at the root
        parent_index = 0
        
        # Going deeper in the tree 
        while True:
            left_child_index  = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reached the leaf, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: # downward search, always search for a higher priority node
                # Go left if our value <= left nodes
                # If not go right using a new value = value - left

                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                    
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        return leaf_index, self.tree[leaf_index], self.data[self._data_index(leaf_index)]
    

    def total_priority(self):
        return self.tree[0] # Returns the root node

    def min_priority(self):
        if self.length < self.capacity:
            return np.min(self.tree[-self.capacity:-self.capacity+self.length])
        else:
            return np.min(self.tree[-self.capacity:])

    def __len__(self):
        """Return the current size of the tree."""
        return self.length