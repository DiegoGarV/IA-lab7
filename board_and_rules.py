"""
Clase creada por chatgpt con el prompt:
Crea una clase para simular un juego de conecta 4. Esta debe de tener las mismas reglas de
victoria que en el juego original, colocar las fichas de abajo hacia arriba, revisar si las
filas y columnas son validas o están llenas y debe de devolver el tablero.
"""

import numpy as np
import random


class Connect4:
    def __init__(self, rows=6, columns=7):
        self.rows = rows
        self.columns = columns
        self.board = np.zeros((rows, columns), dtype=int)

    def drop_piece(self, column, player):
        for row in range(self.rows):
            if self.board[row, column] == 0:
                self.board[row, column] = player
                return True
        return False

    def is_valid_location(self, column):
        return self.board[5, column] == 0

    def get_valid_columns(self):
        return [c for c in range(self.columns) if self.is_valid_location(c)]

    def print_board(self):
        flipped_board = np.flip(self.board, 0)
        symbols = {0: "⚫", 1: "🔴", 2: "🟡"}

        for row in flipped_board:
            print(" ".join(symbols[cell] for cell in row))

    def check_winner(self, player):
        # Horizontal
        for r in range(self.rows):
            for c in range(self.columns - 3):
                if all(self.board[r, c + i] == player for i in range(4)):
                    return True

        # Vertical
        for c in range(self.columns):
            for r in range(self.rows - 3):
                if all(self.board[r + i, c] == player for i in range(4)):
                    return True

        # Diagonal positiva
        for r in range(self.rows - 3):
            for c in range(self.columns - 3):
                if all(self.board[r + i, c + i] == player for i in range(4)):
                    return True

        # Diagonal negativa
        for r in range(3, self.rows):
            for c in range(self.columns - 3):
                if all(self.board[r - i, c + i] == player for i in range(4)):
                    return True

        return False

    """HASTA AQUÍ LLEGÓ LO QUE ME DIÓ CHAT CON EL PROMPT"""

    def reset(self):
        # Reinicia el tablero a su estado inicial.
        self.board = np.zeros((self.rows, self.columns), dtype=int)
    
    def evaluate_line(self, line, player):
        # Evalua una linea de 4 casillas para ver cuantas piezas le pertenece a cada jugador
        # Las 4 casillas pueden estár horizontal, vertical o diagonal

        score = 0
        opponent = 2 if player == 1 else 1

        player_count = np.count_nonzero(line == player)
        empty_count = np.count_nonzero(line == 0)
        opponent_count = np.count_nonzero(line == opponent)

        if player_count == 4:
            score += 100
        elif player_count == 3 and empty_count == 1:
            score += 10
        elif player_count == 2 and empty_count == 2:
            score += 1

        if opponent_count == 4:
            score -= 100
        elif opponent_count == 3 and empty_count == 1:
            score -= 10
        elif opponent_count == 2 and empty_count == 2:
            score -= 1

        return score

    def evaluate_board(self, player):
        # Busca las mejores lineas (como se definió en la función de arriba) en todo el tablero

        score = 0

        for r in range(self.rows):
            for c in range(self.columns - 3):
                line = self.board[r, c : c + 4]
                score += self.evaluate_line(line, player)

        for c in range(self.columns):
            for r in range(self.rows - 3):
                line = self.board[r : r + 4, c]
                score += self.evaluate_line(line, player)

        for r in range(self.rows - 3):
            for c in range(self.columns - 3):
                line = [self.board[r + i, c + i] for i in range(4)]
                score += self.evaluate_line(line, player)

        for r in range(3, self.rows):
            for c in range(self.columns - 3):
                line = [self.board[r - i, c + i] for i in range(4)]
                score += self.evaluate_line(line, player)

        return score


def minimax(board, depth, alpha, beta, maximizing_player, player, use_alpha_beta=True):
    valid_columns = board.get_valid_columns()
    is_terminal = (
        board.check_winner(1) or board.check_winner(2) or len(valid_columns) == 0
    )

    if depth == 0 or is_terminal:
        if board.check_winner(player):
            return (None, 100000000)
        elif board.check_winner(3 - player):
            return (None, -100000000)
        else:
            return (None, board.evaluate_board(player))

    if maximizing_player:
        max_eval = -float("inf")
        best_column = random.choice(valid_columns)
        for column in valid_columns:
            board_copy = Connect4()
            board_copy.board = np.copy(board.board)
            board_copy.drop_piece(column, player)
            eval = minimax(
                board_copy, depth - 1, alpha, beta, False, player, use_alpha_beta
            )[1]
            if eval > max_eval:
                max_eval = eval
                best_column = column
            if use_alpha_beta:
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        return best_column, max_eval
    else:
        min_eval = float("inf")
        best_column = random.choice(valid_columns)
        for column in valid_columns:
            board_copy = Connect4()
            board_copy.board = np.copy(board.board)
            board_copy.drop_piece(column, 3 - player)
            eval = minimax(
                board_copy, depth - 1, alpha, beta, True, player, use_alpha_beta
            )[1]
            if eval < min_eval:
                min_eval = eval
                best_column = column
            if use_alpha_beta:
                beta = min(beta, eval)
                if beta <= alpha:
                    break
        return best_column, min_eval
