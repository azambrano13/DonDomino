from domino import*
import definitions
EPISODES=2000

juego = Juego(6,4)

for episode in range(EPISODES):
    print(f'Partida {episode:d}')
    juego.jugar()

    juego.reset()
