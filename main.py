from domino import*
import definitions
EPISODES=1000

juego = Juego(6,4)

for episode in range(EPISODES):
    juego.jugar()
    juego.reset()
