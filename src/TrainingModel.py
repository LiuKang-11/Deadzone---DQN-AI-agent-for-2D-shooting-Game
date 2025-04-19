import math
import random

class EnvironmentSimulator:
    def __init__(self):
        # grid representation of the map 0 for open/ 1 for obstacles
        self.map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        
        self.player_gun = random.randint(0, 1) # 1 for rifle, 0 for shotgun
        self.player_pos = self.initialize_position() # row, column
        self.player_orientation = random.randint(-180, 180) # facing right
        self.player_health = 10

        self.opponent_gun = random.randint(0, 1) # 1 for rifle, 0 for shotgun
        self.opponent_pos = self.initialize_position() # row, column
        self.opponent_orientation = random.randint(-180, 180) # facing left
        self.opponent_health = 10

    # intialize player and opponent position randomly and make sure they do not collide with the obstacles
    def initialize_position(self):
        pos = random.randint(1, 13), random.randint(1, 18)

        while (self.map[pos[0]][pos[1]] == 1):
            pos = random.randint(1, 13), random.randint(1, 18)

        return list(pos)

    # move player or opponent
    def move(self, offset: tuple, movePlayer = True):
        if movePlayer:
            new_row = self.player_pos[0] + offset[0]
            new_col = self.player_pos[1] + offset[1]
            if self.map[new_row][new_col] != 1:
                self.player_pos = [new_row, new_col]
        else:
            new_row = self.opponent_pos[0] + offset[0]
            new_col = self.opponent_pos[1] + offset[1]
            if self.map[new_row][new_col] != 1:
                self.opponent_pos = [new_row, new_col]

    # rotate the player/opponent sprites
    def turn(self, turnLeft = True, turnPlayer = True):
        if turnPlayer:
            self.player_orientation += 5 if turnLeft else -5
            self.player_orientation = self.mapToRange(self.player_orientation)

        else:
            self.opponent_orientation += 5 if turnLeft else -5
            self.opponent_orientation = self.mapToRange(self.opponent_orientation)

    def mapToRange(self, degrees):
        degrees = degrees % 360
        if degrees > 180:
            degrees -= 360
        return degrees

    def ray_distance(self, angle_deg, max_dist=None):
        """
        往 angle_deg 方向投射一條 Bresenham-like 線，找到第一個障礙物
        回傳距離（格數），如果超過 max_dist 則回傳 max_dist。
        """
        if max_dist is None:
            max_dist = max(len(self.map), len(self.map[0]))
        rad = math.radians(angle_deg)
        dx, dy = math.cos(rad), math.sin(rad)
        px, py = self.player_pos[1], self.player_pos[0]  # x=col, y=row

        for d in range(1, max_dist+1):
            x = int(round(px + dx * d))
            y = int(round(py + dy * d))
            # 越界就當作看不到障礙，回傳當前 d
            if not (0 <= y < len(self.map) and 0 <= x < len(self.map[0])):
                return d
            if self.map[y][x] == 1:
                return d
        return max_dist
    
    # switch guns rifle/shotgun
    def switchWeapon(self, switchPlayerWeapon = True):
        if switchPlayerWeapon:
            self.player_gun = not self.player_gun
        else:
            self.opponent_gun = not self.opponent_gun

    # player shooting the gun
    def shoot(self, playerShooting = True):
        if playerShooting:
            angleRadians = math.degrees(math.atan2(self.opponent_pos[0] - self.player_pos[0], self.opponent_pos[1] - self.player_pos[1]))
            angleDiff = angleRadians - self.player_orientation
            distance = math.sqrt(math.pow(self.opponent_pos[0] - self.player_pos[0], 2) + math.pow(self.opponent_pos[1] - self.player_pos[1], 2))

            # if the player is facing in a direction within 10 degrees of the opponent, shooting will deplete the opponent's health
            if self.player_gun == 1:
                if abs(angleDiff) < 10 and not self.has_obstacle_between() and distance*32 <= 300:
                    self.opponent_health -= 1
                    return True # if shoot was successful
            else:
                if abs(angleDiff) < 10 and not self.has_obstacle_between() and distance*32 <= 100:
                    self.opponent_health -= 3
                    return True
            
        else:
            angleRadians = math.degrees(math.atan2(self.player_pos[0] - self.opponent_pos[0], self.player_pos[1] - self.opponent_pos[1]))
            angleDiff = angleRadians - self.opponent_orientation
            distance = math.sqrt(math.pow(self.player_pos[0] - self.opponent_pos[0], 2) + math.pow(self.player_pos[1] - self.opponent_pos[1], 2))


            # if the player is facing in a direction within 10 degrees of the opponent, shooting will deplete the opponent's health
            if self.opponent_gun == 1:
                if abs(angleDiff) < 10 and not self.has_obstacle_between() and distance*32 <= 300:
                    self.player_health -= 1
                    return True # if shoot was successful
            else:
                if abs(angleDiff) < 10 and not self.has_obstacle_between() and distance*32 <= 100:
                    self.player_health -= 3
                    return True

    # reset everything (randomly initialize all values)
    def reset(self):
        self.player_gun = random.randint(0, 1) # 1 for rifle, 0 for shotgun
        self.player_pos = self.initialize_position() # row, column
        self.player_orientation = random.randint(-180, 180) # facing right
        self.player_health = 10

        self.opponent_gun = random.randint(0, 1) # 1 for rifle, 0 for shotgun
        self.opponent_pos = self.initialize_position() # row, column
        self.opponent_orientation = random.randint(-180, 180) # facing left
        self.opponent_health = 10

    # getting the coordinates of the points between two points
    def bresenham_line(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return points
    
    def perform(self, action):
        if action == 0:
            self.shoot()
        elif action == 1:
            self.move((-1, 0))
        elif action == 2:
            self.move((1, 0))
        elif action == 3:
            self.move((0, -1))
        elif action == 4:
            self.move((0, 1))
        elif action == 5:
            self.move((-1, -1))
        elif action == 6:
            self.move((-1, 1))
        elif action == 7:
            self.move((1, -1))
        elif action == 8:
            self.move((1, 1))
        elif action == 9:
            self.turn(True)
        elif action == 10:
            self.turn(False)
        else:
            self.switchWeapon()

    # checking if there's an obstacle between the player and the opponent
    def has_obstacle_between(self):
        line = self.bresenham_line(self.player_pos[1], self.player_pos[0], self.opponent_pos[1], self.opponent_pos[0])
        for (x, y) in line[1:-1]:  # exclude start and end points
            if self.map[y][x] == 1:
                return True
        return False
    
    def is_terminal(self):
        return self.player_health == 0

import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update_freq = 100, tau = None, opponent=False):
        
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 64
        self.gamma = gamma
        self.environment_simulator = EnvironmentSimulator()
        self.player = not opponent

        # Epsilon-greedy params
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # target network update
        self.target_update_freq = target_update_freq
        self.tau = tau
        self._train_steps = 0

        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.update_target()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state)
                return q_values.argmax().item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Loss and optimization
        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

        # **update target network** **
        if self.tau is None:
            # per target_update_freq step
            self._train_steps += 1
            if self._train_steps % self.target_update_freq == 0:
                self.update_target()
        else:
            # (Polyak averaging)
            for t_param, param in zip(self.target_network.parameters(),
                                      self.q_network.parameters()):
                t_param.data.copy_(
                    self.tau * param.data +
                    (1.0 - self.tau) * t_param.data
                )

    def step(self):
        state = self.getState()
        action = self.select_action(state)

        # Simulate environment
        self.environment_simulator.perform(action)
        next_state = self.getState()
        reward = self.calculate_reward(state, action, next_state)
        done = self.environment_simulator.is_terminal()

        # Add to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Train
        self.train_step()

        return next_state, reward, done

    def calculate_reward(self, state, action, next_state):
        # actions
        # 0 - shoot, 1 - move up, 2 - move down, 3 - move left, 4 - move right
        # 5 - top left, 6 top right, 7 bottom left, 8 bottom right, 9 turn left, 10 turn right, 11 swith weapon

        reward = 0
        distance = self.magnitude((state[2], state[3])) * 32
        isHidden = self.environment_simulator.has_obstacle_between()
        player_orientation = self.mapToRange(self.environment_simulator.player_orientation)
        angle_difference = self.angleDiff(self.angle_between_player_and_opponent((state[2], state[3])), player_orientation)

        # if action is shoot but the opponent health does not deplete, i.e. misfired
        if (action == 0 and next_state[4] == state[4]):
            reward += -2

        # If health is low and running away from the opponent
        if (state[5] < 4 and action in range(1, 9) and self.magnitude((next_state[2], next_state[3])) > distance):
            reward += 1

        # if health is low and hidden behind obstacle
        if (state[5] < 4 and isHidden):
            reward += 5

        # if health is low and not hidden behind obstacle
        if (state[5] < 4 and not isHidden):
            reward -= 2

        # if the player dies
        if (next_state[5] <= 0):
            reward -= 20
        # the player gets hit
        elif (next_state[5] < state[5]):
            reward -= 5

        # if the oppoent dies
        if next_state[4] <= 0 and action == 0:
            reward += 20
        # if action is shoot and the oppoent health depletes
        if (action == 0 and next_state[4] < state[4]):
            reward += 10


        if action == 11:
            if distance < 100 and state[6] == 1 and next_state[6] == 0:
                reward += 2

            if distance < 100 and state[6] == 0 and next_state[6] == 1:
                reward -= 2

            if distance > 100 and state[6] == 1 and next_state[6] == 0:
                reward -= 2

            if distance > 100 and state[6] == 0 and next_state[6] == 1:
                reward += 2

        # if the player is close and the gun type is shotgun
        if distance <= 100 and next_state[6] == 0 and action == 11:
            reward += 1

        # if the player is far and gun type is rifle
        if distance > 100 and next_state[6] == 1:
            reward += 1

        # if the player is close and gun type is not shotgun
        if distance <= 100 and next_state[6] == 1:
            reward -= 3

        # if the player is far and gun type is not rifle
        if distance > 100 and next_state[6] == 0:
            reward -= 3

        # if health not low and still hiding behind obstacles
        if state[5] > 4 and isHidden:
            reward -= 1

        # if the opponent is far and the player has high health
        if distance > 250 and state[5] > 4:
            reward -= 1

        if abs(angle_difference) < 10:
            reward += 5
        else:
            reward -= 2

        if action == 9:
            if angle_difference > 0:
                reward += -1
            else:
                reward += 2
        
        elif action == 10:
            if angle_difference < 0:
                reward += -1
            else:
                reward += 2

        # if the action is a movement action but the position of the player does not change i.e. tried to collide
        if (state[0] == next_state[0] or state[1] == next_state[1]) and action in range(1, 9):
            reward -= 2

        return reward

    # get magnitude of the vector
    def magnitude(self, relative_opponent_pos):
        return math.sqrt(math.pow(relative_opponent_pos[0], 2) + math.pow(relative_opponent_pos[1], 2))
    
    def mapToRange(self, degrees):
        degrees = degrees % 360  # brings it to [0, 360)
        if degrees > 180:
            degrees -= 360        # shift to (-180, 180]
        return degrees
    
    def angle_between_player_and_opponent(self, relative_opponent_pos):
        # get the angle in radians, convert it to degrees and map it to -180 to 180
        return self.mapToRange(math.degrees(math.atan2(relative_opponent_pos[0], relative_opponent_pos[1])))
    
    def angleDiff(self, angle1, angle2):
        # both the angles are in the range [-180, 180]

        diff = angle1 - angle2
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        return diff  # can be negative or positive

    def getState(self):
        # state
        # 0, 1 - player position, 2, 3 - relative position opponent, 4 - opponent health, 5 - player health, 6 - player gun type, 7 - player_orientation, relative position of each obstacle tiles
        env = self.environment_simulator

        player_pos = self.environment_simulator.player_pos[0], self.environment_simulator.player_pos[1] # row, column
        opponent_pos = self.environment_simulator.opponent_pos[0], self.environment_simulator.opponent_pos[1] # row, column
        relative_opponent_pos = opponent_pos[0] - player_pos[0], opponent_pos[1] - player_pos[1]

        ori = env.mapToRange(env.player_orientation)

        # three line
        max_dist = max(len(env.map), len(env.map[0]))
        front   = env.ray_distance(ori,        max_dist) / max_dist
        left30  = env.ray_distance(env.mapToRange(ori + 30), max_dist) / max_dist
        right30 = env.ray_distance(env.mapToRange(ori - 30), max_dist) / max_dist

        # 正規化到 [0,1]
        front   /= max_dist
        left30  /= max_dist
        right30 /= max_dist

        player_heatlh = self.environment_simulator.player_health
        opponent_health = self.environment_simulator.opponent_health
        player_gun_type = self.environment_simulator.player_gun
        player_orientatioon = self.mapToRange(self.environment_simulator.player_orientation)
        
        return [player_pos[0], player_pos[1], relative_opponent_pos[0], relative_opponent_pos[1], player_heatlh, opponent_health, player_gun_type, ori, front, left30, right30]

    def save_model(self, filename="dqn_model.pth"):
        os.makedirs("models", exist_ok=True)
        torch.save(self.q_network.state_dict(), os.path.join("models", filename))

    def load_model(self, filename="dqn_model.pth"):
        self.q_network.load_state_dict(torch.load(os.path.join("models", filename)))
        self.q_network.eval()  # Good practice if you're evaluating



NUMBER_OF_EPISODES = 500

NUMBER_OF_STEPS = 1000

player_dqn_model = DQNAgent(11, 12) # state size and action size
opponent_dqn_model = DQNAgent(11, 12)

def reset():
    player_dqn_model.environment_simulator.reset()

    opponent_dqn_model.environment_simulator.player_pos = player_dqn_model.environment_simulator.opponent_pos
    opponent_dqn_model.environment_simulator.player_health = player_dqn_model.environment_simulator.opponent_health
    opponent_dqn_model.environment_simulator.player_orientation = player_dqn_model.environment_simulator.opponent_orientation
    opponent_dqn_model.environment_simulator.player_gun = player_dqn_model.environment_simulator.opponent_gun

    opponent_dqn_model.environment_simulator.opponent_pos = player_dqn_model.environment_simulator.player_pos
    opponent_dqn_model.environment_simulator.opponent_health = player_dqn_model.environment_simulator.player_health
    opponent_dqn_model.environment_simulator.opponent_orientation = player_dqn_model.environment_simulator.player_orientation
    opponent_dqn_model.environment_simulator.opponent_gun = player_dqn_model.environment_simulator.player_gun

for episode in range(NUMBER_OF_EPISODES):
    winner = None
    player_total_reward = 0
    opponent_total_reward = 0
    for steps in range(NUMBER_OF_STEPS):
        player_next_state, player_reward, done = player_dqn_model.step() # do some action
        opponent_next_state, opponent_reward, opponent_done = opponent_dqn_model.step() # do some action

        player_total_reward += player_reward
        opponent_total_reward += opponent_reward

        if (done or opponent_done):
            print("done")
            if opponent_done:
                print(f"Episode {episode}: Player wins!  save player_wins_ep{episode}.pth")
                player_dqn_model.save_model(f"model_weight.pth")
                winner = player_dqn_model
            elif done:
                print(f"Episode {episode}: Opponent wins!  save opponent_wins_ep{episode}.pth")
                opponent_dqn_model.save_model(f"model_weight.pth")
                winner = opponent_dqn_model
            # 立即複製勝者權重到雙方
            state = winner.q_network.state_dict()
            player_dqn_model.q_network.load_state_dict(state)
            opponent_dqn_model.q_network.load_state_dict(state)
            player_dqn_model.update_target()
            opponent_dqn_model.update_target()
            break

        # syncing the environements of the both models
        player_dqn_model.environment_simulator.opponent_pos = opponent_dqn_model.environment_simulator.player_pos
        player_dqn_model.environment_simulator.player_health = opponent_dqn_model.environment_simulator.opponent_health
        player_dqn_model.environment_simulator.opponent_orientation = opponent_dqn_model.environment_simulator.player_orientation
        player_dqn_model.environment_simulator.opponent_gun = opponent_dqn_model.environment_simulator.player_gun

        opponent_dqn_model.environment_simulator.opponent_pos = player_dqn_model.environment_simulator.player_pos
        opponent_dqn_model.environment_simulator.player_health = player_dqn_model.environment_simulator.opponent_health
        opponent_dqn_model.environment_simulator.opponent_orientation = player_dqn_model.environment_simulator.player_orientation
        opponent_dqn_model.environment_simulator.opponent_gun = player_dqn_model.environment_simulator.player_gun

    if winner is None:
        if player_total_reward >= opponent_total_reward:
            print(f"Episode {episode}: No early done — Player reward higher ({player_total_reward} ≥ {opponent_total_reward})")
            winner = player_dqn_model
        else:
            print(f"Episode {episode}: No early done — Opponent reward higher ({opponent_total_reward} > {player_total_reward})")
            winner = opponent_dqn_model

        # 存檔並同步 winner 權重
        winner.save_model(f"model_winner.pth")
        state = winner.q_network.state_dict()
        player_dqn_model.q_network.load_state_dict(state)
        opponent_dqn_model.q_network.load_state_dict(state)
        player_dqn_model.update_target()
        opponent_dqn_model.update_target()
   

    print(f"EPISODE {episode}: Player Total Reward : {player_total_reward}")
    print(f"EPISODE {episode}: Opponent Total Reward : {opponent_total_reward}")
    print("------------------------------------")

    reset()