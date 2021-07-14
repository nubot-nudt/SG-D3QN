import numpy as np
from numpy.linalg import norm
from crowd_sim.envs.utils.state import tensor_to_joint_state
from crowd_sim.envs.utils.utils import point_to_segment_dist
def estimate_reward_on_predictor(state, next_state):
    """ If the time step is small enough, it's okay to model agent as linear movement during this period
    """
    # collision detection
    if isinstance(state, list) or isinstance(state, tuple):
        state = tensor_to_joint_state(state)
    human_states = state.human_states
    robot_state = state.robot_state
    weight_goal = 0.1
    weight_safe = 1.25
    weight_terminal = 1.0
    re_collision = -0.25
    re_arrival = 1.0
    next_robot_state = next_state.robot_state
    next_human_states = next_state.human_states
    cur_position = np.array((robot_state.px, robot_state.py))
    end_position = np.array((next_robot_state.px, next_robot_state.py))
    goal_position = np.array((robot_state.gx, robot_state.gy))
    reward_goal = (norm(cur_position - goal_position) - norm(end_position - goal_position))
    # check if reaching the goal
    reaching_goal = norm(end_position - np.array([robot_state.gx, robot_state.gy])) < robot_state.radius
    dmin = float('inf')
    collision = False
    safety_penalty = 0
    for i, human in enumerate(human_states):
        next_human = next_human_states[i]
        px = human.px - robot_state.px
        py = human.py - robot_state.py
        ex = next_human.px - next_robot_state.px
        ey = next_human.py - next_robot_state.py
        # closest distance between boundaries of two agents
        closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - robot_state.radius
        if closest_dist < 0:
            collision = True
        if closest_dist < dmin:
            dmin = closest_dist
        if closest_dist < 0.2:
            safety_penalty = safety_penalty + (closest_dist - 0.2)
        # dis_begin = np.sqrt(px ** 2 + py ** 2) - human.radius - robot_state.radius
        # dis_end = np.sqrt(ex ** 2 + ey ** 2) - human.radius - robot_state.radius
        # penalty_begin = 0
        # penalty_end = 0
        # discomfort_dist = 0.5
        # if dis_begin < discomfort_dist:
        #     penalty_begin = dis_begin - discomfort_dist
        # if dis_end < discomfort_dist:
        #     penalty_end = dis_end - discomfort_dist
        # safety_penalty = safety_penalty + (penalty_end - penalty_begin)
    reward_col = 0
    reward_arrival = 0
    if collision:
        reward_col = re_collision
    elif reaching_goal:
        reward_arrival = re_arrival
    reward_terminal = reward_col + reward_arrival
    reward = weight_terminal * reward_terminal + weight_goal * reward_goal + weight_safe * safety_penalty
    # if collision:
        # reward = reward - 100
    reward = reward
    return reward