import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque


class TDNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TDNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)


class TDAgent:
    def __init__(self, alpha=0.01, gamma=0.95, epsilon=0.1, batch_size=32):
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Probabilidad de exploración
        self.batch_size = batch_size

        self.model = TDNetwork(input_size=42, hidden_size=64, output_size=7)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=1000)

    def store_experience(self, board, action, reward, next_board, done):
        # Guarda la experiencia en la memoria
        self.memory.append((board, action, reward, next_board, done))

    def state_to_tensor(self, board):
        # Convierte el estado del tablero en un tensor de PyTorch
        return torch.tensor(board.flatten(), dtype=torch.float32)

    def select_action(self, board, valid_actions):
        # Elige una acción con una política epsilon-greedy
        if not valid_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(valid_actions)  # Exploración
        else:
            with torch.no_grad():
                q_values = self.model(self.state_to_tensor(board))
                q_values = q_values.numpy()
                best_action = max(valid_actions, key=lambda c: q_values[c])
                return best_action  # Explotación

    def update_q_values(self, board, action, reward, next_board, done):
        # Actualiza los valores Q utilizando la diferencia temporal
        state_tensor = self.state_to_tensor(board)
        next_state_tensor = self.state_to_tensor(next_board)

        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)

        target = q_values.clone()
        if done:
            target[action] = reward  # Si el juego terminó, no hay valor futuro
        else:
            target[action] = reward + self.gamma * torch.max(next_q_values)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train_td_agent(game, agent, episodes=1000, print_every=100):
    win_rates = []
    wins = 0

    for episode in range(episodes):
        game.reset()
        done = False
        turn = 1

        while not done:
            player = 1 if turn % 2 != 0 else 2
            valid_moves = game.get_valid_columns()

            if not valid_moves:
                done = True
                break

            action = agent.select_action(game.board, valid_moves)

            if action is None:
                done = True
                break

            game.drop_piece(action, player)

            reward = 0
            if game.check_winner(player):
                reward = 1 if player == 1 else -1
                wins += 1 if player == 1 else 0
                done = True

            next_board = np.copy(game.board)
            agent.update_q_values(game.board, action, reward, next_board, done)

            if len(game.get_valid_columns()) == 0:
                done = True

            turn += 1

        if (episode + 1) % print_every == 0:
            win_rate = wins / print_every
            win_rates.append(win_rate)
            wins = 0

    # Gráfica de aprendizaje
    plt.plot(range(print_every, episodes + 1, print_every), win_rates)
    plt.xlabel("Episodios")
    plt.ylabel("Tasa de victorias")
    plt.title("Evolución del aprendizaje del agente TD")
    plt.show()
