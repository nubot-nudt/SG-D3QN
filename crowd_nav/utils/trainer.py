import logging
import abc
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from crowd_sim.envs.utils.action import ActionXY


class TSRLTrainer(object):
    def __init__(self, value_estimator, state_predictor, memory, device, policy, writer, batch_size, optimizer_str, human_num,
                 reduce_sp_update_frequency, freeze_state_predictor, detach_state_predictor, share_graph_model):
        """
        Train the trainable model of a policy
        """
        self.value_estimator = value_estimator
        self.state_predictor = state_predictor
        self.device = device
        self.writer = writer
        self.target_policy = policy
        self.target_model = None
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str
        self.reduce_sp_update_frequency = reduce_sp_update_frequency
        self.state_predictor_update_interval = human_num
        self.freeze_state_predictor = freeze_state_predictor
        self.detach_state_predictor = detach_state_predictor
        self.share_graph_model = share_graph_model
        self.v_optimizer = None
        self.s_optimizer = None

        # for value update
        self.gamma = 0.9
        self.time_step = 0.25
        self.v_pref = 1

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def set_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.v_optimizer = optim.Adam(self.value_estimator.parameters(), lr=learning_rate)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.Adam(self.state_predictor.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.v_optimizer = optim.SGD(self.value_estimator.parameters(), lr=learning_rate, momentum=0.9)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.SGD(self.state_predictor.parameters(), lr=learning_rate,momentum=0.9)
        else:
            raise NotImplementedError

        if self.state_predictor.trainable:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters()) +
                 list(self.state_predictor.named_parameters())]), self.optimizer_str))
        else:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters())]), self.optimizer_str))

    def set_rl_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.v_optimizer = optim.Adam(self.value_estimator.parameters(), lr=learning_rate)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.Adam(self.state_predictor.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.v_optimizer = optim.SGD(self.value_estimator.parameters(), lr=learning_rate, momentum=0.9)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.SGD(self.state_predictor.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError

        if self.state_predictor.trainable:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters()) +
                 list(self.state_predictor.named_parameters())]), self.optimizer_str))
        else:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters())]), self.optimizer_str))
    # just build for imitation learning
    def optimize_epoch(self, num_epochs):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        for epoch in range(num_epochs):
            epoch_v_loss = 0
            epoch_s_loss = 0
            logging.debug('{}-th epoch starts'.format(epoch))

            update_counter = 0
            for data in self.data_loader:
                robot_states, human_states, actions, values, dones, rewards, next_robot_state, next_human_states = data

                # optimize value estimator
                self.v_optimizer.zero_grad()
                actions = actions.to(self.device)
                outputs = self.value_estimator((robot_states, human_states)).gather(1, actions.unsqueeze(1))
                values = values.to(self.device)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.v_optimizer.step()
                epoch_v_loss += loss.data.item()

                # optimize state predictor
                if self.state_predictor.trainable:
                    update_state_predictor = True
                    if update_counter % self.state_predictor_update_interval != 0:
                        update_state_predictor = False

                    if update_state_predictor:
                        self.s_optimizer.zero_grad()
                        _, next_human_states_est = self.state_predictor((robot_states, human_states), None)
                        loss = self.criterion(next_human_states_est, next_human_states)
                        loss.backward()
                        self.s_optimizer.step()
                        epoch_s_loss += loss.data.item()
                    update_counter += 1
                else:
                    _, next_human_states_est = self.state_predictor((robot_states, human_states), ActionXY(0, 0))
                    loss = self.criterion(next_human_states_est, next_human_states)
                    epoch_s_loss += loss.data.item()

            logging.debug('{}-th epoch ends'.format(epoch))
            self.writer.add_scalar('IL/epoch_v_loss', epoch_v_loss / len(self.memory), epoch)
            self.writer.add_scalar('IL/epoch_s_loss', epoch_s_loss / len(self.memory), epoch)
            logging.info('Average loss in epoch %d: %.2E, %.2E', epoch, epoch_v_loss / len(self.memory),
                         epoch_s_loss / len(self.memory))
        return

    def optimize_batch(self, num_batches, episode):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        v_losses = 0
        s_losses = 0
        batch_count = 0
        self.target_model.value_network.eval()
        self.value_estimator.value_network.eval()
        for data in self.data_loader:
            batch_num = int(self.data_loader.sampler.num_samples // self.batch_size)
            robot_states, human_states, actions, _, done, rewards, next_robot_states, next_human_states = data

            # optimize value estimator
            self.v_optimizer.zero_grad()
            actions = actions.to(self.device)
            # outputs = self.value_estimator((robot_states, human_states))
            #利用value estimator计算当前动作下的q_value
            outputs = self.value_estimator((robot_states, human_states)).gather(1, actions.unsqueeze(1))
            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            #利用value_estimaor选取s_(t+1)下的最优动作
            max_next_Q_index = torch.max(self.value_estimator((next_robot_states, next_human_states)), dim=1)[1]
            #利用target model估计最优动作对应的q_value
            #这就是一个double DQN版本，而不是dqn版本
            next_Q_value = self.target_model((next_robot_states, next_human_states)).gather(1, max_next_Q_index.unsqueeze(1))
            # 这个是DQN版本
            # next_Q_values = self.target_model((next_robot_states, next_human_states))
            # next_Q_value, _ = torch.max(next_Q_values, dim=1)
            # for DQN
            done_infos = (1-done)
            target_values = rewards + torch.mul(done_infos, next_Q_value * gamma_bar)
            clip_base = outputs - target_values
            value_loss = self.criterion(outputs, target_values)
            value_loss.backward()
            self.v_optimizer.step()
            v_losses += value_loss.data.item()

            # optimize state predictor
            if self.state_predictor.trainable:
                update_state_predictor = True
                if self.freeze_state_predictor:
                    update_state_predictor = False
                elif self.reduce_sp_update_frequency and batch_count % self.state_predictor_update_interval == 0:
                    update_state_predictor = False

                if update_state_predictor:
                    self.s_optimizer.zero_grad()
                    _, next_human_states_est = self.state_predictor((robot_states, human_states), None,
                                                                    detach=self.detach_state_predictor)
                    loss = self.criterion(next_human_states_est, next_human_states)
                    loss.backward()
                    self.s_optimizer.step()
                    s_losses += loss.data.item()
            else:
                _, next_human_states_est = self.state_predictor((robot_states, human_states), None,
                                                                detach=self.detach_state_predictor)
                loss = self.criterion(next_human_states_est, next_human_states)
                s_losses += loss.data.item()


            batch_count += 1
            if batch_count > num_batches or batch_count == batch_num:
                break

        average_v_loss = v_losses / num_batches
        average_s_loss = s_losses / num_batches
        logging.info('Average loss : %.2E, %.2E', average_v_loss, average_s_loss)
        self.writer.add_scalar('RL/average_v_loss', average_v_loss, episode)
        self.writer.add_scalar('RL/average_s_loss', average_s_loss, episode)
        self.value_estimator.value_network.train()
        return average_v_loss, average_s_loss

class MPRLTrainer(object):
    def __init__(self, value_estimator, state_predictor, memory, device, policy, writer, batch_size, optimizer_str, human_num,
                 reduce_sp_update_frequency, freeze_state_predictor, detach_state_predictor, share_graph_model):
        """
        Train the trainable model of a policy
        """
        self.value_estimator = value_estimator
        self.state_predictor = state_predictor
        self.device = device
        self.writer = writer
        self.target_policy = policy
        self.target_model = None
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str
        self.reduce_sp_update_frequency = reduce_sp_update_frequency
        self.state_predictor_update_interval = human_num
        self.freeze_state_predictor = freeze_state_predictor
        self.detach_state_predictor = detach_state_predictor
        self.share_graph_model = share_graph_model
        self.v_optimizer = None
        self.s_optimizer = None

        # for value update
        self.gamma = 0.9
        self.time_step = 0.25
        self.v_pref = 1

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def set_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            # self.v_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.value_estimator.parameters()), lr=learning_rate)
            self.v_optimizer = optim.Adam(self.value_estimator.parameters(), lr=learning_rate)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.Adam(self.state_predictor.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.v_optimizer = optim.SGD(self.value_estimator.parameters(), lr=learning_rate, momentum=0.9)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.SGD(self.state_predictor.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise NotImplementedError

        if self.state_predictor.trainable:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters()) +
                 list(self.state_predictor.named_parameters())]), self.optimizer_str))
        else:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters())]), self.optimizer_str))

    def set_rl_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.v_optimizer = optim.Adam(self.value_estimator.parameters(), lr=learning_rate)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.Adam(self.state_predictor.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.v_optimizer = optim.SGD(self.value_estimator.parameters(), lr=learning_rate, momentum=0.9)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.SGD(self.state_predictor.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError

        if self.state_predictor.trainable:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters()) +
                 list(self.state_predictor.named_parameters())]), self.optimizer_str))
        else:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters())]), self.optimizer_str))

    def optimize_epoch(self, num_epochs):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        for epoch in range(num_epochs):
            epoch_v_loss = 0
            epoch_s_loss = 0
            logging.debug('{}-th epoch starts'.format(epoch))

            update_counter = 0
            for data in self.data_loader:
                robot_states, human_states, actions, values, dones, rewards, next_robot_state, next_human_states = data

                # optimize value estimator
                self.v_optimizer.zero_grad()
                actions = actions.to(self.device)
                outputs = self.value_estimator((robot_states, human_states))
                values = values.to(self.device)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.v_optimizer.step()
                epoch_v_loss += loss.data.item()

                # optimize state predictor
                if self.state_predictor.trainable:
                    update_state_predictor = True
                    if update_counter % self.state_predictor_update_interval != 0:
                        update_state_predictor = False

                    if update_state_predictor:
                        self.s_optimizer.zero_grad()
                        _, next_human_states_est = self.state_predictor((robot_states, human_states), None)
                        loss = self.criterion(next_human_states_est, next_human_states)
                        loss.backward()
                        self.s_optimizer.step()
                        epoch_s_loss += loss.data.item()
                    update_counter += 1
                else:
                    _, next_human_states_est = self.state_predictor((robot_states, human_states), ActionXY(0, 0))
                    loss = self.criterion(next_human_states_est, next_human_states)
                    epoch_s_loss += loss.data.item()

            logging.debug('{}-th epoch ends'.format(epoch))
            self.writer.add_scalar('IL/epoch_v_loss', epoch_v_loss / len(self.memory), epoch)
            self.writer.add_scalar('IL/epoch_s_loss', epoch_s_loss / len(self.memory), epoch)
            logging.info('Average loss in epoch %d: %.2E, %.2E', epoch, epoch_v_loss / len(self.memory),
                         epoch_s_loss / len(self.memory))
        return

    def optimize_batch(self, num_batches, episode):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        v_losses = 0
        s_losses = 0
        batch_count = 0
        self.target_model.value_network.eval()
        self.value_estimator.value_network.eval()
        for data in self.data_loader:
            # 从这一步来看，他的学习过程中，将所有的样本都是放进了学习过程的，而不是随机采样的样本
            batch_num = int(self.data_loader.sampler.num_samples // self.batch_size)
            robot_states, human_states, actions, _, done, rewards, next_robot_states, next_human_states = data

            # optimize value estimator
            self.v_optimizer.zero_grad()
            actions = actions.to(self.device)
            outputs = self.value_estimator((robot_states, human_states))
            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            next_value = self.target_model((next_robot_states, next_human_states))
            done_infos = (1 - done)
            target_values = rewards + torch.mul(done_infos, next_value * gamma_bar)

            # values = values.to(self.device)
            loss = self.criterion(outputs, target_values)
            loss.backward()
            self.v_optimizer.step()
            v_losses += loss.data.item()

            # optimize state predictor
            if self.state_predictor.trainable:
                update_state_predictor = True
                if self.freeze_state_predictor:
                    update_state_predictor = False
                elif self.reduce_sp_update_frequency and batch_count % self.state_predictor_update_interval == 0:
                    update_state_predictor = False

                if update_state_predictor:
                    self.s_optimizer.zero_grad()
                    _, next_human_states_est = self.state_predictor((robot_states, human_states), None,
                                                                    detach=self.detach_state_predictor)
                    loss = self.criterion(next_human_states_est, next_human_states)
                    loss.backward()
                    self.s_optimizer.step()
                    s_losses += loss.data.item()
            else:
                _, next_human_states_est = self.state_predictor((robot_states, human_states), None,
                                                                detach=self.detach_state_predictor)
                loss = self.criterion(next_human_states_est, next_human_states)
                s_losses += loss.data.item()
            batch_count += 1
            if batch_count > num_batches or batch_count == batch_num:
                break

        average_v_loss = v_losses / num_batches
        average_s_loss = s_losses / num_batches
        logging.info('Average loss : %.2E, %.2E', average_v_loss, average_s_loss)
        self.writer.add_scalar('RL/average_v_loss', average_v_loss, episode)
        self.writer.add_scalar('RL/average_s_loss', average_s_loss, episode)
        self.value_estimator.value_network.train()
        return average_v_loss, average_s_loss

class VNRLTrainer(object):
    def __init__(self, model, memory, device, policy, batch_size, optimizer_str, writer):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.policy = policy
        self.target_model = None
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str
        self.optimizer = None
        self.writer = writer

        # for value update
        self.gamma = 0.9
        self.time_step = 0.25
        self.v_pref = 1

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def set_rl_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise NotImplementedError
        logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
            [name for name, param in self.model.named_parameters()]), self.optimizer_str))

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, collate_fn=pad_batch)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            logging.debug('{}-th epoch starts'.format(epoch))
            for data in self.data_loader:
                inputs, values, _, _ = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                values = values.to(self.device)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()
            logging.debug('{}-th epoch ends'.format(epoch))
            average_epoch_loss = epoch_loss / len(self.memory)
            self.writer.add_scalar('IL/average_epoch_loss', average_epoch_loss, epoch)
            logging.info('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches, episode=None):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, collate_fn=pad_batch)
        losses = 0
        batch_count = 0
        for data in self.data_loader:
            inputs, _, rewards, next_states = data
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            target_values = rewards + gamma_bar * self.target_model(next_states)

            loss = self.criterion(outputs, target_values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()
            batch_count += 1
            if batch_count > num_batches:
                break

        average_loss = losses / num_batches
        logging.info('Average loss : %.2E', average_loss)

        return average_loss

class TD3RLTrainer(object):
    def __init__(self, actor_network, critic_network, state_predictor, memory, device, policy, writer, batch_size, optimizer_str, human_num,
                 reduce_sp_update_frequency, freeze_state_predictor, detach_state_predictor, share_graph_model):
        """
        Train the trainable model of a policy
        """
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.target_actor_network = copy.deepcopy(self.actor_network)
        self.target_critic_network = copy.deepcopy(self.critic_network)
        self.state_predictor = state_predictor
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.state_optimizer = None

        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str

        self.device = device
        self.writer = writer
        # self.target_model = None
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.reduce_sp_update_frequency = reduce_sp_update_frequency
        self.state_predictor_update_interval = human_num
        self.freeze_state_predictor = freeze_state_predictor
        self.detach_state_predictor = detach_state_predictor

        policy_noise = 0.2
        noise_clip = 0.5
        policy_freq = 2
        # parameter for TD3
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.action_dim = policy.action_dim
        self.max_action = policy.max_action
        self.tau = 0.005
        # for value update
        self.gamma = 0.9
        self.time_step = 0.25
        self.v_pref = 1
        self.discount = pow(self.gamma,self.time_step*self.v_pref)
        self.total_iteration = 0

    # 没有必要通过外面的model进行参数传递吧，总感不咋聪明
    def update_target_model(self, target_model):
        print('test for update target model')
        # self.target_actor_network = copy.deepcopy(self.actor_network)
        # self.target_critic_network = copy.deepcopy(self.critic_network)

    def set_rl_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=learning_rate)
            self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=learning_rate)
            if self.state_predictor.trainable:
                self.state_optimizer = optim.Adam(self.state_predictor.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.actor_optimizer = optim.SGD(self.actor_network.parameters(), lr=learning_rate, momentum=0.9)
            self.critic_optimizer = optim.SGD(self.critic_network.parameters(), lr=learning_rate, momentum=0.9)
            if self.state_predictor.trainable:
                self.state_optimizer = optim.SGD(self.state_predictor.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise NotImplementedError

        if self.state_predictor.trainable:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in
                 list(self.actor_network.named_parameters()) + list(self.critic_network.named_parameters())
                 + list(self.state_predictor.named_parameters())]), self.optimizer_str))
        else:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.actor_network.named_parameters()) +
                 list(self.critic_network.named_parameters())]), self.optimizer_str))
    # wait for revise
    # def optimize_epoch(self, num_epochs):
    #     if self.v_optimizer is None:
    #         raise ValueError('Learning rate is not set!')
    #     if self.data_loader is None:
    #         self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
    #     for epoch in range(num_epochs):
    #         epoch_v_loss = 0
    #         epoch_s_loss = 0
    #         logging.debug('{}-th epoch starts'.format(epoch))
    #
    #         update_counter = 0
    #         for data in self.data_loader:
    #             robot_states, human_states, actions, values, dones, rewards, next_robot_state, next_human_states = data
    #
    #             # optimize value estimator
    #             self.v_optimizer.zero_grad()
    #             actions = actions.to(self.device)
    #             outputs = self.value_estimator((robot_states, human_states))
    #             values = values.to(self.device)
    #             loss = self.criterion(outputs, values)
    #             loss.backward()
    #             self.v_optimizer.step()
    #             epoch_v_loss += loss.data.item()
    #
    #             # optimize state predictor
    #             if self.state_predictor.trainable:
    #                 update_state_predictor = True
    #                 if update_counter % self.state_predictor_update_interval != 0:
    #                     update_state_predictor = False
    #
    #                 if update_state_predictor:
    #                     self.s_optimizer.zero_grad()
    #                     _, next_human_states_est = self.state_predictor((robot_states, human_states), None)
    #                     loss = self.criterion(next_human_states_est, next_human_states)
    #                     loss.backward()
    #                     self.s_optimizer.step()
    #                     epoch_s_loss += loss.data.item()
    #                 update_counter += 1
    #             else:
    #                 _, next_human_states_est = self.state_predictor((robot_states, human_states), ActionXY(0, 0))
    #                 loss = self.criterion(next_human_states_est, next_human_states)
    #                 epoch_s_loss += loss.data.item()
    #
    #         logging.debug('{}-th epoch ends'.format(epoch))
    #         self.writer.add_scalar('IL/epoch_v_loss', epoch_v_loss / len(self.memory), epoch)
    #         self.writer.add_scalar('IL/epoch_s_loss', epoch_s_loss / len(self.memory), epoch)
    #         logging.info('Average loss in epoch %d: %.2E, %.2E', epoch, epoch_v_loss / len(self.memory),
    #                      epoch_s_loss / len(self.memory))
    #     return

    def optimize_batch(self, num_batches, episode):
        self.total_iteration += 1
        if self.actor_optimizer is None or self.critic_network is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)

        v_losses = 0
        s_losses = 0
        batch_count = 0
        # 前向预测过程中，需要利用eval关闭算法中所有的dropout层和batch normalization
        # 与之对应的是，在train过程中，需要利用train开启算法中所有的dropout层和normalization

        self.actor_network.train()
        self.target_actor_network.train()
        self.critic_network.train()
        self.target_critic_network.train()
        for data in self.data_loader:
            batch_num = int(self.data_loader.sampler.num_samples // self.batch_size)
            robot_states, human_states, actions, _, done, rewards, next_robot_states, next_human_states = data
            with torch.no_grad():
                next_states = (next_robot_states, next_human_states)
                cur_states = (robot_states, human_states)
                # Select action according to policy and add clipped noise
                # 在策略优化过程中，往往会添加噪声，使得训练的结果更加地平滑
                noise = (
                        torch.randn_like(actions) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                noise = noise.float()
                #next action is a tensor
                next_action = (
                        self.target_actor_network(next_states) + noise
                )
                next_action_array = next_action.numpy()
                next_action_array = next_action_array.clip(-self.max_action, self.max_action)
                # float 32
                next_action = torch.tensor(next_action_array, dtype=torch.float32)
                # Compute the target Q value
                target_Q1, target_Q2 = self.target_critic_network(next_states, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                done_infos = (1 - done)
                target_Q = rewards + done_infos * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic_network(cur_states, actions)
            # 在反向传播过程中，是可以区分到两个critic network的
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            v_losses += critic_loss.data.item()
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_iteration % self.policy_freq == 0:
                # Compute actor loss

                actor_loss = -self.critic_network.Q1(cur_states, self.actor_network(cur_states)).mean()
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            # # optimize state predictor
            # if self.state_predictor.trainable:
            #     update_state_predictor = True
            #     if self.freeze_state_predictor:
            #         update_state_predictor = False
            #     elif self.reduce_sp_update_frequency and batch_count % self.state_predictor_update_interval == 0:
            #         update_state_predictor = False
            #
            #     if update_state_predictor:
            #         self.state_optimizer.zero_grad()
            #         _, next_human_states_est = self.state_predictor((robot_states, human_states), None,
            #                                                         detach=self.detach_state_predictor)
            #         loss = self.criterion(next_human_states_est, next_human_states)
            #         loss.backward()
            #         self.state_optimizer.step()
            #         s_losses += loss.data.item()
            # else:
            #     _, next_human_states_est = self.state_predictor((robot_states, human_states), None,
            #                                                     detach=self.detach_state_predictor)
            #     loss = self.criterion(next_human_states_est, next_human_states)
            #     s_losses += loss.data.item()

            batch_count += 1
            if batch_count > num_batches or batch_count == batch_num:
                break

        if self.total_iteration % self.policy_freq == 0:
            # Update the frozen target models
            for param, target_param in zip(self.critic_network.parameters(), self.target_critic_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_network.parameters(), self.target_actor_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        average_v_loss = v_losses / num_batches
        average_s_loss = s_losses / num_batches
        logging.info('Average loss : %.2E, %.2E', average_v_loss, average_s_loss)
        self.writer.add_scalar('RL/average_v_loss', average_v_loss, episode)
        self.writer.add_scalar('RL/average_s_loss', average_s_loss, episode)
        self.actor_network.eval()
        self.critic_network.eval()
        self.target_critic_network.eval()
        self.target_actor_network.eval()
        return average_v_loss, average_s_loss

def pad_batch(batch):
    """
    args:
        batch - list of (tensor, label)
    return:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
    """
    def sort_states(position):
        # sort the sequences in the decreasing order of length
        sequences = sorted([x[position] for x in batch], reverse=True, key=lambda t: t.size()[0])
        packed_sequences = torch.nn.utils.rnn.pack_sequence(sequences)
        return torch.nn.utils.rnn.pad_packed_sequence(packed_sequences, batch_first=True)

    states = sort_states(0)
    values = torch.cat([x[1] for x in batch]).unsqueeze(1)
    rewards = torch.cat([x[2] for x in batch]).unsqueeze(1)
    next_states = sort_states(3)

    return states, values, rewards, next_states
