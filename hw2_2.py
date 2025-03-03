import os
import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as o
from matplotlib import pyplot as plt
from homework2 import Hw2Env
import mlflow


GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_DECAY_ITER = 100
MIN_EPSILON = 0.1
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
UPDATE_FREQ = 4
TARGET_NETWORK_UPDATE_FREQ = 1000
BUFFER_LENGTH = 100000
EPISODE_COUNT = 10001
TAU = 0
N_ACTIONS = 8

hyper_parameters = {
    "gamma": GAMMA,
    "epsilon": EPSILON,
    "epsilon_decay": EPSILON_DECAY,
    "epsilon_decay_iter": EPSILON_DECAY_ITER,
    "min_epsilon": MIN_EPSILON,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "update_freq": UPDATE_FREQ,
    'target_network_update_freq': TARGET_NETWORK_UPDATE_FREQ,
    "buffer_length": BUFFER_LENGTH,
    'tau': TAU,
    'episode_count': EPISODE_COUNT
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN2(nn.Module):
    def __init__(self):
        super(DQN2, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, 4, 2, 1)
        self.conv1 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.pool = nn.AvgPool2d(2, 3)

        self.linear = nn.Linear(512, N_ACTIONS)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.linear(x)
        return x


def train(name):

    os.makedirs('models', exist_ok=True)
    os.makedirs('visuals', exist_ok=True)

    mlflow.start_run(run_name=f'{name}')
    mlflow.log_params(hyper_parameters)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    env = Hw2Env(output="normal", render_mode='offscreen')
    policy_net = DQN2().to(device)
    mlflow.log_param("model", str(policy_net))
    target_net = DQN2().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    loss_function = nn.MSELoss()
    optimizer = o.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = Memory(BUFFER_LENGTH)
    epsilon = EPSILON
    step_counter = 0

    x_values = []
    total_reward_values = []
    rps_values = []
    epsilon_values = []
    loss_values = []

    for episode in range(EPISODE_COUNT):
        env.reset()
        state = env.state()
        done = False
        total_reward = 0
        current_step_counter = 0
        loss_value = 0
        loss_count = 0

        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(N_ACTIONS)
            else:
                with torch.no_grad():
                    state_tensor = state.to(device)
                    with torch.no_grad():
                        policy_net.eval()
                        q_values = policy_net(state_tensor.unsqueeze(0))
                    action = torch.argmax(q_values).to('cpu').item()

            next_state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            total_reward += reward

            memory.push(state, action, reward, next_state, done)
            state = next_state

            step_counter += 1
            current_step_counter += 1
            if step_counter % UPDATE_FREQ == 0 and len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))

                states = torch.stack([arr for arr in batch.state]).float().to(device)
                actions = torch.asarray(batch.action).long().to(device)
                rewards = torch.asarray(batch.reward).float().to(device)
                next_states = torch.stack([arr for arr in batch.next_state]).float().to(device)
                dones = torch.asarray(batch.done).bool().to(device)

                policy_net.train()
                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                target_q = rewards + GAMMA * next_q * (~dones)

                loss = loss_function(current_q, target_q)
                loss_value += loss.item()
                loss_count += 1
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

            if step_counter % TARGET_NETWORK_UPDATE_FREQ == 0:

                for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)

            if step_counter % EPSILON_DECAY_ITER == 0:
                epsilon = max(epsilon * EPSILON_DECAY, MIN_EPSILON)

        rps = total_reward/current_step_counter
        x_values.append(episode)
        total_reward_values.append(total_reward)
        rps_values.append(rps)
        epsilon_values.append(epsilon)
        loss_value = loss_value / loss_count if loss_count > 0 else 0
        loss_values.append(loss_value)

        mlflow.log_metrics({
            "total reward": total_reward,
            "rps": rps,
            "epsilon": epsilon,
            "loss": loss_value,
        }, step=episode)

        print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, RPS: {rps}, Epsilon: {epsilon:.3f}")
        if episode % 500 == 0:
            torch.save(policy_net.state_dict(), f"models/{name}_{episode:06}.pth")

    plt.plot(x_values, total_reward_values)
    plt.title("Total Reward")
    plt.savefig(f"visuals/{name}_total_reward.png")
    plt.close()
    plt.plot(x_values, rps_values)
    plt.title("RPS")
    plt.savefig(f"visuals/{name}_rps.png")
    plt.close()
    plt.plot(x_values, epsilon_values)
    plt.title("Epsilon")
    plt.savefig(f"visuals/{name}_epsilon.png")
    plt.close()


def evaluate(model_path="hw2_2.pth", experiment_count=1, prediction_epsilon=0.05):

    os.makedirs('evaluation', exist_ok=True)

    env = Hw2Env(output='normal', render_mode='gui')
    policy_net = DQN2()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    policy_net.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
    policy_net = policy_net.to(device)
    policy_net.eval()

    rewards = []
    rps = []
    experiment_ids = []

    for e in range(experiment_count):
        env.reset()
        state = env.state()
        done = False
        total_reward = 0
        current_step_counter = 0

        while not done:
            state_tensor = state.to(device)
            if np.random.random() < prediction_epsilon:
                action = np.random.randint(N_ACTIONS)
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor.unsqueeze(0))
                    action = torch.argmax(q_values).to('cpu').item()

            next_state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            total_reward += reward
            state = next_state

            current_step_counter += 1

        rewards.append(total_reward)
        rps.append(total_reward/current_step_counter)
        experiment_ids.append(e)

    plt.bar(experiment_ids, rewards)
    plt.title("Accumulated Rewards for HW2_2")
    plt.xlabel("Experiment ids")
    plt.ylabel("Total Reward")
    plt.savefig('evaluation/hw2_2_reward.jpg')
    plt.close()

    plt.bar(experiment_ids, rps)
    plt.title("RPS for HW2_2")
    plt.xlabel("Experiment ids")
    plt.ylabel("RPS")
    plt.savefig('evaluation/hw2_2_rps.jpg')
    plt.close()


if __name__ == '__main__':
    # train('2_0001')
    evaluate(experiment_count=5)
