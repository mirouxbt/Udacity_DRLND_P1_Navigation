import numpy as np
import random

from model           import QNetwork
from ReplayBuffer    import ReplayBuffer
from ReplayBufferPER import ReplayBufferPER

import torch
import torch.nn.functional as F
import torch.optim as optim

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=0, gamma=0.99, learning_rate = 5e-4,
                       use_RB=True, RB_size=int(1e5), RB_batch_size=64,
                       use_TM=True, TM_update_every=4, use_DDQN=True,
                       use_PER=False, PER_epsilon=0.01, PER_alpha=0.5, PER_beta=0.4, PER_beta_increment=0.001,
                       use_DUELING=True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size                  (int)   : dimension of each state
            action_size                 (int)   : dimension of each action
            seed                        (int)   : random seed
            gamma                       (float) : discount factor
            learning_rate               (float) : learning rate of the model

            use_RB                      (bool)  : Use a replay buffer
            RB_size                     (int)   : replay buffer size
            RB_batch_size               (int)   : minibatch size of the learning

            use_TM                      (bool)  : Use a target model
            TM_update_every             (int)   : update target model every t steps

            use_DDQN                    (bool)  : Use Double DQN, only valid if use target model
            
            use_PER                     (bool)  : Use a prioritized replay buffer
            PER_epsilon                 (float) : Small value added to priorities to avoid zero probabilities
            PER_alpha                   (float) : Power used to compute the sampling probabilities
                                                  [0-1] : 0=> Uniform sampling 1=>Fully prioritized
            PER_beta                    (float) : Used in importance-sampling - Initial value increased to 1
            PER_beta_increment          (float) : To increment beta at each sampling

            use_DUELING                 (bool)  : Use DUELING network
        """
        # Control some parameters
        assert not use_PER  or (use_PER and use_RB) , "Use replay buffer if use PER" # To make sure we remember to update RB params
        assert not use_DDQN or (use_DDQN and use_TM), "Use target model if use DDQN"


        self.state_size  = state_size
        self.action_size = action_size

        self.gamma = gamma

        # Q-Network
        self.qnetwork_policy = QNetwork(state_size, action_size, seed,use_DUELING=use_DUELING).to(device)
        self.optimizer = optim.Adam(self.qnetwork_policy.parameters(), lr=learning_rate)
        
        self.use_DDQN = use_DDQN
        self.use_TM   = use_TM
        if use_TM:
            self.qnetwork_target = QNetwork(state_size, action_size, seed,use_DUELING=use_DUELING).to(device)
            self.TM_update_every = TM_update_every

        # Initialize time step
        self.t_step = 0


        # Replay memory
        self.use_RB        = use_RB
        self.RB_batch_size = RB_batch_size
        self.use_PER       = use_PER
        if use_PER:
            self.memory = ReplayBufferPER(RB_size, RB_batch_size, seed, 
                                          epsilon=PER_epsilon, alpha=PER_alpha, beta=PER_beta, beta_increment=PER_beta_increment)
        elif use_RB:
            self.memory = ReplayBuffer(RB_size, RB_batch_size, seed)

        # Init the seed
        random.seed(seed)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_policy.eval()
            with torch.no_grad():
                action_values = self.qnetwork_policy(state)
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))



    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory if any
        if self.use_PER:
            # Need to compute the error of this experience
            Q_target, Q_expected = self._QValues([(state, action, reward, next_state, done)])
            error = (Q_target - Q_expected).cpu().squeeze().data.item()
            
            self.memory.add(error , (state, action, reward, next_state, done) )
        elif self.use_RB:
            self.memory.add( (state, action, reward, next_state, done) )
        else:
            self.experiences = [(state, action, reward, next_state, done)]

        # One more step.
        self.t_step += 1

        # If no replay buffer or not enough samples available in memory, learn
        if not self.use_RB or len(self.memory) > self.RB_batch_size:
            self._learn()

    def _QValues(self, batch):
        """Execute a forward path for the QNetworks to get the QValues (expected and target)
           So the TD error can be computed or used to learn

           Params
           ======

           batch : Array of tuple <state, action, reward, next_state, done>
        """

        # Get the types by line
        mini_batch = np.array(batch).transpose()

        states      = torch.Tensor(np.vstack(mini_batch[0])).float().to(device)
        actions     = torch.Tensor(np.vstack(mini_batch[1])).long().to(device)
        rewards     = torch.Tensor(np.vstack(mini_batch[2])).float().to(device)
        next_states = torch.Tensor(np.vstack(mini_batch[3])).float().to(device)
        dones       = torch.Tensor(np.vstack(mini_batch[4]).astype(int)).float().to(device)

        # Get max predicted Q values (for next states) from target model
        if not self.use_TM or (self.use_TM and self.use_DDQN):
            self.qnetwork_policy.eval()
            with torch.no_grad():
                action_values_policy = self.qnetwork_policy(next_states)

        if self.use_TM:
            self.qnetwork_target.eval()
            with torch.no_grad():
                action_values_target = self.qnetwork_target(next_states)

        if self.use_TM:
            if self.use_DDQN:
                Q_targets_next = action_values_target.gather(dim=1, index=action_values_policy.max(dim=1, keepdim=True)[1])
            else:
                Q_targets_next = action_values_target.max(dim=1, keepdim=True)[0]
        else:
            Q_targets_next = action_values_policy.max(dim=1, keepdim=True)[0]

        # Need to be at zero if we were done
        Q_targets_next *= torch.ones_like(dones) - dones

        # Compute the Q targets for current states
        Q_targets = rewards + self.gamma * Q_targets_next

        
        # Get the Q values from policy model
        self.qnetwork_policy.train()
        Q_expected = self.qnetwork_policy(states).gather(dim=1, index=actions)

        return Q_targets, Q_expected

    def _learn(self):
        """Update value parameters using given a batch of experience tuples."""

        if self.use_PER:
            experiences, indexes, IS_weights = self.memory.sample()
            IS_weights = torch.Tensor(np.vstack(IS_weights)).float().to(device)
        elif self.use_RB:
            experiences = self.memory.sample()
        else:
            experiences = self.experiences

        # Get the Qvalues for those experiences
        Q_targets, Q_expected = self._QValues(experiences)


        if self.use_PER:
            # Update priorities of the replay buffer
            errors = (Q_targets - Q_expected).cpu().squeeze().data.numpy()
            self.memory.update_priorities(indexes, errors)

            # Update Qs with the importance-sampling weight correction
            Q_expected *= IS_weights**0.5
            Q_targets  *= IS_weights**0.5

        # Loss computation
        loss = F.mse_loss(Q_expected, Q_targets)
        #loss = F.smooth_l1_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.use_TM:
            self.t_step %= self.TM_update_every
            if self.t_step == 0:
                self.qnetwork_target.load_state_dict(self.qnetwork_policy.state_dict())                    

    def save_weights(self, file='checkpoint.pth'):
        """Save the agent network weights in a checkpoint file"""
        torch.save(self.qnetwork_policy.state_dict(), file)

    def load_weights(self, file='checkpoint.pth'):
        """Load the agent network weights from a checkpoint file"""
        self.qnetwork_policy.load_state_dict(torch.load(file))