import pandas as pd
import matplotlib.pyplot as plt
from board_and_rules import Connect4, minimax
from td_agent import TDAgent, train_td_agent


RESULTADOS_FILE = "resultados.csv"


def guardar_resultado(modo, ganador):
    # Guarda el resultado de cada partida en un archivo CSV.
    try:
        df = pd.read_csv(RESULTADOS_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["modo", "ganador"])

    df = pd.concat([df, pd.DataFrame({"modo": [modo], "ganador": [ganador]})], ignore_index=True)
    df.to_csv(RESULTADOS_FILE, index=False)


def graficar_resultados():
    # Genera y actualiza una gráfica con los resultados de las partidas.
    try:
        df = pd.read_csv(RESULTADOS_FILE)
    except FileNotFoundError:
        print("No hay datos para graficar aún.")
        return

    conteo = df.groupby(["modo", "ganador"]).size().unstack(fill_value=0)

    # Modos de juego
    modos = ["1 - Jugador vs TDL", "2 - TDL vs Minimax", "3 - TDL vs Minimax+Poda"]
    ganadores = ["Jugador", "Minimax", "Empate", "TDL"]

    # Asegurar que todas las combinaciones tengan valores (evita errores si no hay datos)
    for ganador in ganadores:
        if ganador not in conteo.columns:
            conteo[ganador] = 0

    conteo = conteo[ganadores]

    ax = conteo.plot(kind="bar", figsize=(9, 6), width=0.7, colormap="viridis")

    plt.title("Resultados de las partidas")
    plt.xlabel("Modo de juego")
    plt.ylabel("Cantidad de victorias")
    plt.xticks(ticks=range(len(modos)), labels=modos, rotation=0)
    plt.legend(title="Ganador", labels=ganadores)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("grafica_resultados.png")
    plt.show()


def play_game():
    td_agent = TDAgent()
    game = Connect4()

    print("Agente de TD learning entrenando")
    train_td_agent(agent=td_agent, game=game, episodes=5000)
    print("El agente a terminado de entrenar")

    while True:
        game.reset()

        print("\nElige el modo de juego:")
        print("1 - Jugador vs IA (TD Learning)")
        print("2 - IA (TDL) vs IA (Minimax sin poda)")
        print("3 - IA (TDL) vs IA (Minimax con poda)")
        print("4 - Ver gráficas")

        mode = input("\nIngresa el número de la opción: ")

        if mode == "4":
            graficar_resultados()
            continue

        turn = 1
        while True:
            game.print_board()
            player = 1 if turn % 2 != 0 else 2

            if mode == "1" and player == 1:
                # Modo 1: Jugador vs IA (TD Learning)
                try:
                    column = int(input(f"Jugador {player}, elige una columna (1-7): ")) - 1
                except ValueError:
                    print("Entrada inválida. Intenta nuevamente.")
                    continue

                if column not in range(7):
                    print("Columna fuera de rango. Intenta nuevamente.")
                    continue

                if not game.is_valid_location(column):
                    print("Columna llena. Intenta nuevamente.")
                    continue

                game.drop_piece(column, player)
            else:
                if (mode == "1" and player == 2) or mode in ["2", "3"]:
                    # IA (TD Learning) juega
                    valid_actions = game.get_valid_columns()
                    column = td_agent.select_action(game.board, valid_actions)
                else:
                    # IA (Minimax) juega
                    use_alpha_beta = mode == "3"
                    column, _ = minimax(game, 4, -float("inf"), float("inf"), True, player, use_alpha_beta)

                print(f"Jugador {player} elige la columna: {column+1}")
                game.drop_piece(column, player)

            # Verificar ganador
            if game.check_winner(player):
                game.print_board()
                print(f"¡Jugador {player} ha ganado!")
                guardar_resultado(mode, "TDL" if player == 1 else "Minimax")
                break

            if len(game.get_valid_columns()) == 0:
                game.print_board()
                print("¡Empate!")
                guardar_resultado(mode, "Empate")
                break

            turn += 1

        replay = input("¿Quieres jugar de nuevo? (s/n): ").lower()
        if replay != "s":
            break


if __name__ == "__main__":
    play_game()
