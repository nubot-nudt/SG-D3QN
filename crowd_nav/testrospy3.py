#!/usr/bin/env python
import rospy
import numpy as np
#from test_py_ros.msg import test
from std_msgs.msg import String
from sgdqn_common.msg import ObserveInfo, RobotState, PedState
def talker():
    pub = rospy.Publisher('observe_info', ObserveInfo, queue_size=10)
    rospy.init_node('test_node', anonymous=True)
    rate = rospy.Rate(4)
    while not rospy.is_shutdown():
        robot_state = RobotState()
        robot_state.pos_x = 0.0
        robot_state.pos_y = -4.0
        robot_state.vel_x = 0.0
        robot_state.vel_y = 0.0
        robot_state.radius = 0.3
        robot_state.vmax = 1.0
        robot_state.theta = 0.0
        robot_state.goal_x = 0.0
        robot_state.goal_y = 4.0

        human1_state = PedState()
        human1_state.pos_x = 4.0
        human1_state.pos_y = 0.0
        human1_state.vel_x = 0.0
        human1_state.vel_y = 0.0
        human1_state.radius = 0.3
        # human1_state.goal_x = -4.0
        # human1_state.goal_y = 0.0
        # human1_state.vmax = 1.0
        # human1_state.theta = 0.0

        theta = np.pi / 4.0
        human2_state = PedState()
        human2_state.pos_x = 4.0 * np.cos(theta)
        human2_state.pos_y = 4.0 * np.sin(theta)
        human2_state.vel_x = 0.0
        human2_state.vel_y = 0.0
        human2_state.radius = 0.3
        # human2_state.goal_x = - 4.0 * np.cos(theta)
        # human2_state.goal_y = - 4.0 * np.sin(theta)
        # human2_state.vmax = 1.0
        # human2_state.theta = 0.0

        theta = np.pi / 3.0
        human3_state = PedState()
        human3_state.pos_x = 4.0 * np.cos(theta)
        human3_state.pos_y = 4.0 * np.sin(theta)
        human3_state.vel_x = 0.0
        human3_state.vel_y = 0.0
        human3_state.radius = 0.3
        # human3_state.goal_x = - 4.0 * np.cos(theta)
        # human3_state.goal_y = - 4.0 * np.sin(theta)
        # human3_state.vmax = 1.0
        # human3_state.theta = 0.0

        theta = - np.pi / 3.0
        human4_state = PedState()
        human4_state.pos_x = 4.0 * np.cos(theta)
        human4_state.pos_y = 4.0 * np.sin(theta)
        human4_state.vel_x = 0.0
        human4_state.vel_y = 0.0
        human4_state.radius = 0.3
        # human4_state.goal_x = - 4.0 * np.cos(theta)
        # human4_state.goal_y = - 4.0 * np.sin(theta)
        # human4_state.vmax = 1.0
        # human4_state.theta = 0.0

        observe_info = ObserveInfo()
        observe_info.robot_state = robot_state
        observe_info.ped_states.append(human1_state)
        observe_info.ped_states.append(human2_state)
        observe_info.ped_states.append(human3_state)
        observe_info.ped_states.append(human4_state)
        # hello_str="hello world %s"%rospy.get_time()
        # rospy.loginfo(hello_str)
        pub.publish(observe_info)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSException:
        pass