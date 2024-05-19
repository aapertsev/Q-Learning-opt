import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from collections import Counter
import time
import os
import numpy as np
from environment1 import ABCOptimizationEnv
import random
import config


state_size = 11  # Размер вектора состояния
action_size = 3  # Размер пространства действий (refactor, rewrite, balance)
learning_rate = 0.001
gamma = 0.9
epsilon = 0.7
epsilon_decay = 0.95
epsilon_min = 0.3
episodes = 100
batch_size = 10
max_steps_per_episode = 10


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQN(state_size, action_size).eval()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)

    def act(self, state, history):
        if len(history) == 9:
            counts = Counter(history)
            return counts.most_common()[-1][0]

        if np.random.rand() <= self.epsilon:
            print("Выбрано рандомное действие")
            action = random.randrange(3)
            if len(history)!=0:
                while(action == history[-1]):
                    action = random.randrange(3)
            return action
        else:
            print("\033[92m"+"Действие предсказала dqn"+"\033[0m")
            state = torch.from_numpy(state).float().unsqueeze(0)
            q_values = self.model(state)
            print("\033[92m" + 'q values: ' + str(q_values) + "\033[0m")
            action = torch.argmax(q_values).item()
            if len(history)!=0:
                if action == history[-1]:
                    q_values = q_values[q_values!=q_values[0,action]]
                    print(f"Измененные q-values: {q_values}")
                    action = torch.argmax(q_values).item()
            return action
    def predict(self, state, history):
        state = torch.from_numpy(state).float().unsqueeze(0)
        q_values = self.model(state)
        action = torch.argmax(q_values).item()
        if len(history) != 0:
            if action == history[-1]:
                q_values = q_values[q_values != q_values[0, action]]
                print(f"Измененные q-values: {q_values}")
                action = torch.argmax(q_values).item()
        if len(history) == 9:
            counts = Counter(history)
            action  = counts.most_common()[-1][0]
        return action
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size):
        print(self.memory)
        start_time = time.time()

        def replay(self, batch_size):
            if len(self.memory) < batch_size:
                return

            batch = random.sample(self.memory, batch_size)
            states, actions, rewards, next_states = zip(*batch)

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)

            # Вычисление текущих Q-значений
            current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Вычисление предсказанных Q-значений
            next_q_values = self.target_model(next_states).max(1)[0].detach()
            expected_q = rewards + (self.gamma * next_q_values)

            # Вычисление потерь и обновление модели
            loss = nn.SmoothL1Loss()(current_q, expected_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Обновление epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        print(f"Пакетное обучение выполнено за {time.time() - start_time:.2f} секунд")
def train_dqn(agent, env, episodes, batch_size, max_steps_per_episode):
    #state = env.reset()
    total_reward = 0

    for step in range(max_steps_per_episode):
        step_start_time = time.time()
        action = agent.act(env.state, env.history)
        env.history.append(action)
        last_state, reward = env.step(action)
        env.state[8+action]+=1
        total_reward += reward
        agent.remember(last_state, action, reward, env.state)

        print(f"Шаг {step + 1}/{max_steps_per_episode} выполнен за {time.time() - step_start_time:.2f} секунд\n")
    agent.replay(batch_size)


def predict_and_optimize(agent, predict_path):
    adp_stats = {}
    print('\n\n\nНачалось предсказание модели\n\n')
    for root, dirs, files in os.walk(predict_path):
        for bench_dir in dirs:
            bench_dir_path = os.path.join(root, bench_dir)
            for file in os.listdir(bench_dir_path):
                if file.endswith("_orig.bench"):
                    bench_path = os.path.join(bench_dir_path, file)
                    env = ABCOptimizationEnv(config.abc_path, bench_path, config.lib_path)
                    env.create_new_file()
                    metrics, state = env.parse_stats()
                    initial_adp = metrics[0]*metrics[1]
                    for _ in range(10):  # Последовательное предсказание и выполнение 10 действий
                        metrics, state = env.parse_stats()  # Текущее состояние среды
                        print(metrics)
                        action = agent.predict(state, env.history)# Предсказание действия моделью
                        env.history.append(action)
                        print("Действие: ", action)
                        env.step(action)  # Применяем действие
                    res_adp = env.metrics[0]*env.metrics[1]
                    stat = res_adp/initial_adp*100
                    adp_stats[f"{file}"] = str(stat)+'%'
    return adp_stats