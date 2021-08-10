import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_nav.policy.reward_estimate import Reward_Estimator
import rospy
from crowd_sim.envs.utils.state import ObservableState, FullState, JointState
from sgdqn_common.msg import ObserveInfo, VelInfo, AgentVel
from crowd_sim.envs.utils.action import ActionXY, ActionRot

class sgdqn_planner:
    def init(self):
        self.robot_policy = None
        self.peds_policy = None
        rospy.init_node('sgdqn_planner_node', anonymous=True)
        self.human_vel_pub = rospy.Publisher('human_vel_cmd', VelInfo, queue_size=10)
        self.robot_vel_pub = rospy.Publisher('robot_vel_cmd', AgentVel, queue_size=10)
        self.robot_policy = policy_factory['tree_search_rl']()  # ??? 目前这样写是不行的
        self.peds_policy = policy_factory['centralized_orca']()
        rospy.Subscriber("observe_info", ObserveInfo, self.state_callback)
        rospy.spin()

    def configure(self, config):
        self.robot_policy = policy_factory['tree_search_rl']()  # ??? 目前这样写是不行的
        self.peds_policy = policy_factory['centralized_orca']()

    def state_callback(self, observe_info):
        print("state callback")
        robot_state = observe_info.robot_state
        robot_full_state = FullState(robot_state.pos_x, robot_state.pos_y, robot_state.vel_x, robot_state.vel_y,
                                     robot_state.radius, robot_state.goal_x, robot_state.goal_y, robot_state.vmax,
                                     robot_state.theta)
        peds_full_state = [FullState(ped_state.pos_x, ped_state.pos_y, ped_state.vel_x, ped_state.vel_y,
                                     ped_state.radius, ped_state.goal_x, ped_state.goal_y, ped_state.vmax,
                                     ped_state.theta) for ped_state in observe_info.ped_states]
        observable_states = self.compute_observation(peds_full_state)
        state = JointState(robot_full_state, observable_states)
        # robot_action, robot_action_index = self.robot_policy.predict(state)
        human_actions = self.peds_policy.predict(peds_full_state)
        test_action = ActionXY(0.0, 0.0)
        robot_vel = AgentVel()
        robot_vel.vel_x = test_action.vx
        robot_vel.vel_y = test_action.vy
        vel_infos = VelInfo()
        vel_infos.vel_info.append(robot_vel)


        # human policy
        for human_action in human_actions:
            human_vel = AgentVel()
            human_vel.vel_x = human_action.vx
            human_vel.vel_y = human_action.vy
            vel_infos.vel_info.append(human_vel)
        self.human_vel_pub.publish(vel_infos)

    def compute_observation(self, full_states):
        observation_states = [full_state.get_observable_state() for full_state in full_states]
        return observation_states


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='model_predictive_rl')
    parser.add_argument('-m', '--model_dir', type=str, default='data/output1')#None
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=None)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')
    sys_args = parser.parse_args()
    try:
        planner = sgdqn_planner()
        planner.init()
        # planner.configure()
    except rospy.ROSException:
        pass