import logging
import torch
import numpy as np
from numpy.linalg import norm
import itertools
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import tensor_to_joint_state
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_nav.policy.state_predictor import StatePredictor, LinearStatePredictor_batch
from crowd_nav.policy.graph_model import RGL,GAT_RL
from crowd_nav.policy.value_estimator import DQNNetwork, Noisy_DQNNetwork
from crowd_nav.policy.reward_estimate import estimate_reward_on_predictor
from crowd_nav.policy.actor import Actor
from crowd_nav.policy.critic import Critic


class TD3RL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'TD3RL'
        self.trainable = True
        self.multiagent_training = True
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.action_space = None
        self.rotation_constraint = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.robot_state_dim = 9
        self.human_state_dim = 5
        self.v_pref = 1
        self.share_graph_model = None
        self.value_estimator = None
        self.linear_state_predictor = None
        self.state_predictor = None
        self.planning_depth = None
        self.planning_width = None
        self.do_action_clip = None
        self.sparse_search = None
        self.sparse_speed_samples = 2
        self.sparse_rotation_samples = 8
        self.action_group_index = []
        self.traj = None
        self.use_noisy_net = False
        self.count=0
        self.action_dim = 2
        # max_action must be a tensor
        self.max_action = None

    def configure(self, config, device):
        self.set_common_parameters(config)
        self.planning_depth = config.model_predictive_rl.planning_depth
        self.do_action_clip = config.model_predictive_rl.do_action_clip
        if hasattr(config.model_predictive_rl, 'sparse_search'):
            self.sparse_search = config.model_predictive_rl.sparse_search
        self.planning_width = config.model_predictive_rl.planning_width
        self.share_graph_model = config.model_predictive_rl.share_graph_model
        self.linear_state_predictor = config.model_predictive_rl.linear_state_predictor
        # self.set_device(device)
        self.device = device


        if self.linear_state_predictor:
            self.state_predictor = LinearStatePredictor_batch(config, self.time_step)
            graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
            self.value_estimator = DQNNetwork(config, graph_model)
            self.model = [graph_model, self.value_estimator.value_network]
        else:
            if self.share_graph_model:
                graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
                self.value_estimator = DQNNetwork(config, graph_model)
                self.state_predictor = StatePredictor(config, graph_model, self.time_step)
                self.model = [graph_model, self.value_estimator.value_network, self.state_predictor.human_motion_predictor]
            else:
                graph_model1 = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                self.actor = Actor(config,graph_model1,self.action_dim, self.max_action)
                graph_model2 = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                graph_model3 = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                self.critic = Critic(config, graph_model2, graph_model3, self.action_dim)
                graph_model4 = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                self.state_predictor = StatePredictor(config, graph_model4, self.time_step)
                self.model = [graph_model1, graph_model2, graph_model3, self.actor.action_network,
                              self.critic.score_network1, self.critic.score_network2,
                              self.state_predictor.human_motion_predictor]

        logging.info('TD3 action_dim is : {}'.format(self.action_dim))

    def set_common_parameters(self, config):
        self.gamma = config.rl.gamma
        self.kinematics = config.action_space.kinematics
        self.sampling = config.action_space.sampling
        self.speed_samples = config.action_space.speed_samples
        self.rotation_samples = config.action_space.rotation_samples
        self.rotation_constraint = config.action_space.rotation_constraint

    def set_device(self, device):
        self.device = device
        for model in self.model:
            model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_noisy_net(self, use_noisy_net):
        self.use_noisy_net = use_noisy_net

    def set_time_step(self, time_step):
        self.time_step = time_step
        self.state_predictor.time_step = time_step

    def get_normalized_gamma(self):
        return pow(self.gamma, self.time_step * self.v_pref)

    def get_model(self):
        return self.value_estimator

    def get_state_dict(self):
        return {
                'graph_model1': self.actor.graph_model.state_dict(),
                'graph_model2': self.critic.graph_model1.state_dict(),
                'graph_model3': self.critic.graph_model2.state_dict(),
                'graph_model4': self.state_predictor.graph_model.state_dict(),
                'action_network': self.actor.action_network.state_dict(),
                'score_network1': self.critic.score_network1.state_dict(),
                'score_network2': self.critic.score_network2.state_dict(),
                'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
            }


    def get_traj(self):
        return self.traj

    def load_state_dict(self, state_dict):
        self.actor.graph_model.load_state_dict(state_dict['graph_model1'])
        self.critic.graph_model1.load_state_dict(state_dict['graph_model2'])
        self.critic.graph_model2.load_state_dict(state_dict['graph_model3'])
        self.state_predictor.graph_model.load_state_dict(state_dict['graph_model4'])
        self.actor.action_network.load_state_dict(['action_network'])
        self.critic.score_network1.load_state_dict(['score_network1'])
        self.critic.score_network2.load_state_dict(['score_network2'])
        self.state_predictor.human_motion_predictor.load_state_dict(state_dict['motion_predictor'])

    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon and self.use_noisy_net is False:
            random_action = torch.Tensor(np.random.random(self.action_dim)) * self.max_action
            return ActionXY(random_action[0],random_action[1]) if self.kinematics == 'holonomic' \
                else ActionRot(random_action[0],random_action[1])
        else:
            action = self.actor(state)
            return ActionXY(action[0], action[1]) if self.kinematics == 'holonomic' else ActionRot(action[0], action[1])

    def get_attention_weights(self):
        return self.actor.graph_model.attention_weights