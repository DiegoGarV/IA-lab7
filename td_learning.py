import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


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
    def __init__(
        self,
        alpha=0.0001,
        gamma=0.99,
        epsilon=1.0,
        min_epsilon=0.01,
        decay_rate=0.995,
        input_size=42,
        hidden_size=128,
        output_size=7,
    ):
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Probabilidad de exploración
        self.model = TDNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

    def state_to_tensor(self, board):
        # Convierte el estado del tablero en un tensor de PyTorch
        return torch.tensor(board.flatten(), dtype=torch.float32)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def select_action(self, board, valid_actions):
        # Elige una acción con una política epsilon-greedy
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


def train_td_agent(agent, game, episodes=5000):
    for episode in range(episodes):
        game.reset()
        done = False
        turn = 1

        while not done:
            player = 1 if turn % 2 != 0 else 2
            valid_actions = game.get_valid_columns()
            action = agent.select_action(game.board, valid_actions)

            # Realiza la jugada
            game.drop_piece(action, player)

            # Calcula la recompensa (win=1, loss=-1, draw=0)
            if game.check_winner(player):
                reward = 1
                done = True
            elif game.check_winner(3 - player):
                reward = -1
                done = True
            elif len(game.get_valid_columns()) == 0:
                reward = 0
                done = True
            else:
                reward = 0

            # Actualiza los Q-values
            next_board = game.board.copy()
            agent.update_q_values(game.board, action, reward, next_board, done)

            turn += 1

        agent.decay_epsilon()
