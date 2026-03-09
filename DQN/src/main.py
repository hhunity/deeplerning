import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import lib.mongodb as mdb
import lib.dataframeLib as dl
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ==================== Qネットワーク（DQN） ====================
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.net(x)

# ==================== 経験を保存するバッファ ====================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), actions, rewards, np.array(next_states), dones)

    def __len__(self):
        return len(self.buffer)

# ==================== FX取引環境 ====================
class FXEnv:
    def __init__(self, data: np.ndarray, window_size=10, spread=0.002):
        self.data = data
        self.window_size = window_size
        self.current_step = window_size
        self.done = False
        self.max_step = len(self.data)
        self.prev_price = self.data[self.current_step - 1][3]  # 初期のclose価格
        self.spread = spread
        self.position = None  # 'long' か None
        self.entry_price = 0.0

    def reset(self):
        self.current_step = self.window_size
        self.done = False
        self.prev_price = self.data[self.current_step - 1][3]
        self.position = None
        self.entry_price = 0.0
        return self._get_state()

    def _get_state(self):
        window = self.data[self.current_step - self.window_size:self.current_step]
        return window.flatten().astype(np.float32)

    def step(self, action):
        self.current_step += 1
        if self.current_step >= self.max_step:
            self.done = True

        next_state = self._get_state()
        current_price = self.data[self.current_step - 1][3]  # 今のclose価格

        reward = 0.0

        if 0:
            if action == 0:  # sell
                reward = self.prev_price - current_price
            elif action == 2:  # buy
                reward = current_price - self.prev_price
        else:
            if action == 2:  # BUY（エントリー）
                if self.position is None:
                    self.position = 'long'
                    self.entry_price = current_price
                    # スプレッドは今は考慮しない（SELLで引く）
                elif self.position == 'long':
                    reward = current_price - self.prev_price

            elif action == 0:  # SELL（決済）
                if self.position == 'long':
                    profit = current_price - self.entry_price - self.spread
                    reward = profit
                    self.position = None
                    self.entry_price = 0.0

            elif action == 1:  # HOLD（保有中の含み益差分）
                if self.position == 'long':
                    reward = current_price - self.prev_price
        
        # print(f"reward: {reward:.5f}, action: {action}, position: {self.position}")

        self.prev_price = current_price
        return next_state, reward, self.done

# ==================== データ読み込み（正規化付き） ====================
def load_tick_data():
    db = mdb.mongoDBW('USDJPY3')
    df = db.find("tick2", dt.datetime(2025, 1, 10, 0, 0, 0), dt.datetime(2025, 1, 15, 23, 59, 59))
    df = df.dropna()
    df = dl.make_ohlc_from_ticks(df, interval='3T')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled.astype(np.float32)

# ==================== 学習損失の可視化 ====================
def plot_loss(loss_list):
    plt.figure(figsize=(12, 5))
    plt.plot([l if l is not None else 0 for l in loss_list], label='Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Episodes')
    plt.grid(True)
    plt.legend()
    plt.show()

# ==================== 学習済みモデルによるPNL評価 ====================
def evaluate_pnl(env, policy_net):
    state = env.reset()
    position = None  # ポジションの有無（'long' か None）
    entry_price = 0.0
    total_pnl = 0.0
    pnl_list = []

    while True:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = torch.argmax(q_values).item()  # 最もQ値の高い行動を選択

        current_price = state[-1]  # 最後の要素をclose価格と仮定

        if action == 2 and position is None:
            position = 'long'
            entry_price = current_price

        elif action == 0 and position == 'long':
            profit = current_price - entry_price - env.spread
            total_pnl += profit
            position = None

        elif action == 1 and position == 'long':
            pass
            # total_pnl += current_price - entry_price

        pnl_list.append(total_pnl)

        next_state, _, done = env.step(action)
        state = next_state
        if done:
            break

    return pnl_list

# ==================== 累積PNLグラフの描画 ====================
def plot_pnl(pnl_list):
    plt.figure(figsize=(12, 5))
    plt.plot(pnl_list, label='Cumulative PNL')
    plt.xlabel('Time Step')
    plt.ylabel('PNL')
    plt.title('Agent Profit and Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

# ==================== ハイパーパラメータ設定 ====================
data = load_tick_data()
spred = 0.00
window_size = 10            # 状態として使用するローソク足の本数
feature_dim = 4             # 各ローソク足の特徴量（open, high, low, close）
state_dim = window_size * feature_dim  # flattenした状態の次元数
hidden_dim = 64             # DQNの中間層のノード数
action_size = 3             # 行動数（0: sell, 1: hold, 2: buy）

env = FXEnv(data, window_size, spread=spred)  # スプレッドありの環境
policy_net = DQN(state_dim, hidden_dim, action_size)
target_net = DQN(state_dim, hidden_dim, action_size)
target_net.load_state_dict(policy_net.state_dict())  # ターゲットネット初期化

gamma = 0.99                # 割引率（将来の報酬の価値）
epsilon = 0.1               # ε-greedy法の探索率
lr = 0.001                  # 学習率
batch_size = 32            # ミニバッチのサイズ
buffer = ReplayBuffer(capacity=10000)  # 経験バッファの容量
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

loss_list = []

# ==================== 学習ループ ====================
for episode in range(1000):
    state = env.reset()
    episode_losses = []

    while True:
        # ε-greedy で行動選択
        if random.random() < epsilon:
            action = random.randint(0, action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()

        # 環境ステップ実行 & 経験保存
        next_state, reward, done = env.step(action)
        buffer.push((state, action, reward, next_state, done))
        state = next_state

        # バッチサイズ以上なら学習実行
        if len(buffer) > batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Q値の予測とターゲット計算
            q_values = policy_net(states).gather(1, actions.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target_net(next_states).max(1)[0]
            targets = rewards + gamma * next_q_values * (1 - dones)

            # 損失計算と最適化
            loss = loss_fn(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_losses.append(loss.item())

        if done:
            break

    # エピソードごとの平均損失記録
    if episode_losses:
        avg_loss = sum(episode_losses) / len(episode_losses)
        loss_list.append(avg_loss)
    else:
        loss_list.append(None)

    # ターゲットネットを定期更新
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(loss_list[-1])

# 損失とPNLのグラフ表示
plot_loss(loss_list)

spred = 0.00
pnl_list = evaluate_pnl(env, policy_net)
plot_pnl(pnl_list)