import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, use_DUELING=True, fc1_unit = 128, fc2_unit = 64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        torch.manual_seed(seed)

        self.use_DUELING = use_DUELING
        if use_DUELING:
            self.fc1 = nn.Linear(state_size, fc1_unit)

            self.fc1_value  = nn.Linear(fc1_unit  , fc2_unit)
            self.fc2_value  = nn.Linear(fc2_unit  , fc2_unit)
            self.out_value = nn.Linear(fc2_unit  , 1)

            self.fc1_action  = nn.Linear(fc1_unit  , fc2_unit)
            self.fc2_action  = nn.Linear(fc2_unit  , fc2_unit)
            self.out_action = nn.Linear(fc2_unit  , action_size)
        else:
            self.fc1 = nn.Linear(state_size, fc1_unit)
            self.fc2 = nn.Linear(fc1_unit  , fc2_unit)
            self.fc3 = nn.Linear(fc2_unit  , fc2_unit)
            self.out = nn.Linear(fc2_unit  , action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        if self.use_DUELING:
            x = F.relu(self.fc1(state))

            x_value = F.relu(self.fc1_value(x))
            x_value = F.relu(self.fc2_value(x_value))
            x_value = self.out_value(x_value)

            x_action = F.relu(self.fc1_action(x))
            x_action = F.relu(self.fc2_action(x_action))
            x_action = self.out_action(x_action)

            # Now combine using the equation x_value + x_action - mean(x_action)
            return x_value + x_action - x_action.mean()
        else:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.out(x)


