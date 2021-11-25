import torch.nn as nn
import torch
import torch.nn.functional as F
from crowd_nav.policy.helpers import mlp
import numpy as np

class Actor(nn.Module):
    def __init__(self, config, graph_model, action_dim, max_action, min_action):
        super(Actor, self).__init__()
        self.graph_model = graph_model
        self.action_network = mlp(config.gcn.X_dim, [256, action_dim])
        self.max_action = None
        self.min_action = None
        self.action_dim = action_dim
        self.action_amplitude = max_action
        self.action_middle = min_action

    def set_action(self, action_dim, max_action, min_action):
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        self.action_amplitude = (self.max_action - self.min_action) / 2.0
        self.action_amplitude = torch.from_numpy(self.action_amplitude)
        self.action_middle = torch.from_numpy(self.min_action + self.max_action) / 2.0

    def forward(self, state):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3

        # only use the feature of robot node as state representation
        state_embedding = self.graph_model(self.rotate(state))[:, 0, :]
        a = self.action_network(state_embedding)
        action = self.action_middle + self.action_amplitude * torch.tanh(a)
        return action
        # return self.max_action * torch.tanh(a)

    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input tuple include robot state tensor and human state tensor.
        robot state tensor is of size (batch_size, number, state_length)(for example 100*1*9)
        human state tensor is of size (batch_size, number, state_length)(for example 100*5*5)
        """
        # for robot
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
        #  0     1      2     3      4        5     6      7         8
        # for human
        #  'px', 'py', 'vx', 'vy', 'radius'
        #  0     1      2     3      4
        assert len(state[0].shape) == 3
        if len(state[1].shape) == 3:
            batch = state[0].shape[0]
            robot_state = state[0]
            human_state = state[1]
            human_num = state[1].shape[1]
            dx = robot_state[:, :, 5] - robot_state[:, :, 0]
            dy = robot_state[:, :, 6] - robot_state[:, :, 1]
            dx = dx.unsqueeze(1)
            dy = dy.unsqueeze(1)
            dg = torch.norm(torch.cat([dx, dy], dim=2), 2, dim=2, keepdim=True)
            rot = torch.atan2(dy, dx)
            cos_rot = torch.cos(rot)
            sin_rot = torch.sin(rot)
            transform_matrix = torch.cat((cos_rot, -sin_rot, sin_rot, cos_rot), dim=1).reshape(batch, 2, 2)
            robot_velocities = torch.bmm(robot_state[:, :, 2:4], transform_matrix)
            radius_r = robot_state[:, :, 4].unsqueeze(1)
            v_pref = robot_state[:, :, 7].unsqueeze(1)
            target_heading = torch.zeros_like(radius_r)
            pos_r = torch.zeros_like(robot_velocities)
            cur_heading = (robot_state[:, :, 8].unsqueeze(1) - rot + np.pi) % (2 * np.pi) - np.pi
            new_robot_state = torch.cat((pos_r, robot_velocities, radius_r, dg, target_heading, v_pref, cur_heading), dim=2)
            human_positions = human_state[:, :, 0:2] - robot_state[:, :, 0:2]
            human_positions = torch.bmm(human_positions, transform_matrix)
            human_velocities = human_state[:, :, 2:4]
            human_velocities = torch.bmm(human_velocities, transform_matrix)
            human_radius = human_state[:, :, 4].unsqueeze(2) + 0.3
            new_human_state = torch.cat((human_positions, human_velocities, human_radius), dim=2)
            new_state = (new_robot_state, new_human_state)
            return new_state
        else:
            batch = state[0].shape[0]
            robot_state = state[0]
            dx = robot_state[:, :, 5] - robot_state[:, :, 0]
            dy = robot_state[:, :, 6] - robot_state[:, :, 1]
            dx = dx.unsqueeze(1)
            dy = dy.unsqueeze(1)
            radius_r = robot_state[:, :, 4].unsqueeze(1)
            dg = torch.norm(torch.cat([dx, dy], dim=2), 2, dim=2, keepdim=True)
            rot = torch.atan2(dy, dx)
            cos_rot = torch.cos(rot)
            sin_rot = torch.sin(rot)
            vx = (robot_state[:, :, 2].unsqueeze(1) * cos_rot +
                  robot_state[:, :, 3].unsqueeze(1) * sin_rot).reshape((batch, 1, -1))
            vy = (robot_state[:, :, 3].unsqueeze(1) * cos_rot -
                  robot_state[:, :, 2].unsqueeze(1) * sin_rot).reshape((batch, 1, -1))
            v_pref = robot_state[:, :, 7].unsqueeze(1)
            theta = robot_state[:, :, 8].unsqueeze(1)
            px_r = torch.zeros_like(v_pref)
            py_r = torch.zeros_like(v_pref)
            new_robot_state = torch.cat((px_r, py_r, vx, vy, radius_r, dg, rot, v_pref, theta), dim=2)
            new_state = (new_robot_state, None)
            return new_state


    def trans_no_rotation(self, state):
        """
        Transform the coordinate to agent-centric.
        Input tuple include robot state tensor and human state tensor.
        robot state tensor is of size (batch_size, number, state_length)(for example 100*1*9)
        human state tensor is of size (batch_size, number, state_length)(for example 100*5*5)
        """
        # for robot
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
        #  0     1      2     3      4        5     6      7         8
        # for human
        #  'px', 'py', 'vx', 'vy', 'radius'
        #  0     1      2     3      4
        assert len(state[0].shape) == 3
        if state[1] is None:
            robot_state = state[0]
            dx = robot_state[:, :, 5] - robot_state[:, :, 0]
            dy = robot_state[:, :, 6] - robot_state[:, :, 1]
            dx = dx.unsqueeze(1)
            dy = dy.unsqueeze(1)
            radius_r = robot_state[:, :, 4].unsqueeze(1)
            dg = torch.norm(torch.cat([dx, dy], dim=2), 2, dim=2, keepdim=True)
            rot = torch.atan2(dy, dx)
            vx = robot_state[:, :, 2].unsqueeze(1)
            vy = robot_state[:, :, 3].unsqueeze(1)
            v_pref = robot_state[:, :, 7].unsqueeze(1)
            px_r = torch.zeros_like(v_pref)
            py_r = torch.zeros_like(v_pref)
            theta = robot_state[:, :, 8].unsqueeze(1)
            new_robot_state = torch.cat((px_r, py_r, vx, vy, radius_r, dg, rot, v_pref, theta), dim=2)
            new_state = (new_robot_state, None)
            return new_state
        else:
            batch = state[0].shape[0]
            robot_state = state[0]
            human_state = state[1]
            human_num = state[1].shape[1]
            dx = robot_state[:, :, 5] - robot_state[:, :, 0]
            dy = robot_state[:, :, 6] - robot_state[:, :, 1]
            dx = dx.unsqueeze(1)
            dy = dy.unsqueeze(1)
            radius_r = robot_state[:, :, 4].unsqueeze(1)
            dg = torch.norm(torch.cat([dx, dy], dim=2), 2, dim=2, keepdim=True)
            rot = torch.atan2(dy, dx)
            vx = robot_state[:, :, 2].unsqueeze(1)
            vy = robot_state[:, :, 3].unsqueeze(1)
            v_pref = robot_state[:, :, 7].unsqueeze(1)
            px_r = torch.zeros_like(v_pref)
            py_r = torch.zeros_like(v_pref)
            theta = robot_state[:, :, 8].unsqueeze(1)
            new_robot_state = torch.cat((px_r, py_r, vx, vy, radius_r, dg, rot, v_pref, theta), dim=2)
            new_human_state = None
            for i in range(human_num):
                dx1 = human_state[:, i, 0].unsqueeze(1) - robot_state[:, :, 0]
                dy1 = human_state[:, i, 1].unsqueeze(1) - robot_state[:, :, 1]
                dx1 = dx1.unsqueeze(1).reshape((batch, 1, -1))
                dy1 = dy1.unsqueeze(1).reshape((batch, 1, -1))
                vx1 = (human_state[:, i, 2].unsqueeze(1).unsqueeze(2)).reshape((batch, 1, -1))
                vy1 = (human_state[:, i, 3].unsqueeze(1).unsqueeze(2)).reshape((batch, 1, -1))
                radius_h = human_state[:, i, 4].unsqueeze(1).unsqueeze(2)
                cur_human_state = torch.cat((dx1, dy1, vx1, vy1, radius_h), dim=2)
                if new_human_state is None:
                    new_human_state = cur_human_state
                else:
                    new_human_state = torch.cat((new_human_state, cur_human_state), dim=1)
            new_state = (new_robot_state, new_human_state)
            return new_state