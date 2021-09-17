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
from sgdqn_common.msg import ObserveInfo, ActionCmd
from crowd_sim.envs.utils.action import ActionXY, ActionRot, ActionDiff

class sgdqn_planner:
    def init(self):
        self.robot_policy = None
        self.peds_policy = None
        self.cur_state = None
        rospy.init_node('sgdqn_planner_node', anonymous=True)

    def start(self):
        rospy.Subscriber("observeInfo", ObserveInfo, self.state_callback)
        # self.human_vel_pub = rospy.Publisher('human_vel_cmd', VelInfo, queue_size=10)
        self.robot_action_pub = rospy.Publisher('robot_action_cmd', ActionCmd, queue_size=10)
        rospy.spin()

    def configure(self):
        self.robot_policy = policy_factory['tree_search_rl']()
        self.peds_policy = policy_factory['centralized_orca']()

    def load_policy_model(self, args):
        level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S")
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
        logging.info('Using device: %s', device)

        if args.model_dir is not None:
            if args.config is not None:
                config_file = args.config
            else:
                config_file = os.path.join(args.model_dir, 'config.py')
            if args.il:
                model_weights = os.path.join(args.model_dir, 'il_model.pth')
                logging.info('Loaded IL weights')
            elif args.rl:
                if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                    model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
                else:
                    print(os.listdir(args.model_dir))
                    model_weights = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
                logging.info('Loaded RL weights')
            else:
                model_weights = os.path.join(args.model_dir, 'best_val.pth')
                logging.info('Loaded RL weights with best VAL')

        else:
            config_file = args.config

        spec = importlib.util.spec_from_file_location('config', config_file)
        if spec is None:
            parser.error('Config file not found.')
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        # configure policy
        policy_config = config.PolicyConfig(args.debug)
        policy = policy_factory[policy_config.name]()
        reward_estimator = Reward_Estimator()
        env_config = config.EnvConfig(args.debug)
        reward_estimator.configure(env_config)
        policy.reward_estimator = reward_estimator
        if args.planning_depth is not None:
            policy_config.model_predictive_rl.do_action_clip = True
            policy_config.model_predictive_rl.planning_depth = args.planning_depth
        if args.planning_width is not None:
            policy_config.model_predictive_rl.do_action_clip = True
            policy_config.model_predictive_rl.planning_width = args.planning_width
        if args.sparse_search:
            policy_config.model_predictive_rl.sparse_search = True

        policy.configure(policy_config, device)
        if policy.trainable:
            if args.model_dir is None:
                parser.error('Trainable policy must be specified with a model weights directory')
            policy.load_model(model_weights)

        # configure environment
        env_config = config.EnvConfig(args.debug)

        if args.human_num is not None:
            env_config.sim.human_num = args.human_num
        env = gym.make('CrowdSim-v0')
        env.configure(env_config)

        if args.square:
            env.test_scenario = 'square_crossing'
        if args.circle:
            env.test_scenario = 'circle_crossing'
        if args.test_scenario is not None:
            env.test_scenario = args.test_scenario

        # for continous action
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high
        min_action = env.action_space.low
        if policy.name == 'TD3RL':
            policy.set_action(action_dim, max_action, min_action)
        self.robot_policy.set_time_step(env.time_step)
        self.robot_policy = policy

        train_config = config.TrainConfig(args.debug)
        epsilon_end = train_config.train.epsilon_end
        if not isinstance(self.robot_policy, ORCA):
            self.robot_policy.set_epsilon(epsilon_end)

        policy.set_phase(args.phase)
        policy.set_device(device)

        # set safety space for ORCA in non-cooperative simulation
        if isinstance(self.robot_policy, ORCA):
            self.robot_policy.safety_space = args.safety_space
            logging.info('ORCA agent buffer: %f', self.robot_policy.safety_space)

    def state_callback(self, observe_info):
        print("state callback")
        robot_state = observe_info.robot_state
        robot_full_state = FullState(robot_state.pos_x, robot_state.pos_y, robot_state.vel_x, robot_state.vel_y,
                                     robot_state.radius, robot_state.goal_x, robot_state.goal_y, robot_state.vmax,
                                     robot_state.theta)
        peds_full_state = [ObservableState(ped_state.pos_x, ped_state.pos_y, ped_state.vel_x, ped_state.vel_y,
                                     ped_state.radius) for ped_state in observe_info.ped_states]
        observable_states = peds_full_state
        self.cur_state = JointState(robot_full_state, observable_states)
        action_cmd = ActionCmd()

        dis = np.sqrt((robot_full_state.px - robot_full_state.gx)**2 + (robot_full_state.py - robot_full_state.gy)**2)
        if dis < 0.3:
            action_cmd.stop = True
            action_cmd.vel_x = - np.sign(robot_full_state.vx) * robot_full_state.vx * 2.0
            action_cmd.vel_y = - np.sign(robot_full_state.vy) * robot_full_state.vy * 2.0
        else:
            action_cmd.stop = False
            robot_action, robot_action_index = self.robot_policy.predict(self.cur_state)
            print('robot_action', robot_action.al, robot_action.ar)
            action_cmd.vel_x = robot_action.al
            action_cmd.vel_y = robot_action.ar
        self.robot_action_pub.publish(action_cmd)
        # human_actions = self.peds_policy.predict(peds_full_state)
        #
        # test_action = ActionXY(0.0, 0.0)
        # robot_vel = AgentVel()
        # robot_vel.vel_x = test_action.vx
        # robot_vel.vel_y = test_action.vy
        # vel_infos = VelInfo()
        # vel_infos.vel_info.append(robot_vel)
        # # human policy
        # for human_action in human_actions:
        #     human_vel = AgentVel()
        #     human_vel.vel_x = human_action.vx
        #     human_vel.vel_y = human_action.vy
        #     vel_infos.vel_info.append(human_vel)
        # self.human_vel_pub.publish(vel_infos)

    def compute_observation(self, full_states):
        observation_states = [full_state.get_observable_state() for full_state in full_states]
        return observation_states


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='tree_search_rl')
    parser.add_argument('-m', '--model_dir', type=str, default='data/0827/tsrl/2')#None
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
        planner.configure()
        planner.load_policy_model(sys_args)
        planner.start()
    except rospy.ROSException:
        pass
