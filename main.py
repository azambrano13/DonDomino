from domino import *
import definitions
import matplotlib.pyplot as plt

EPISODES = 10

juego = Juego(6,4)

E = []
R = []
for episode in range(EPISODES):
    print(f'Partida {episode:d}...')
    rewards = juego.jugar()

    juego.reset()

    R.extend( rewards )
    E.extend( [episode]*len(rewards) )

plt.figure()
plt.scatter( E, R )
plt.xlabel("Episodes")
plt.ylabel("Rewards")


plt.figure()
plt.hist(R )
plt.xlabel("Rewards")
plt.show()