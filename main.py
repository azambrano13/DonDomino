from domino import *
import definitions
import matplotlib.pyplot as plt

EPISODES = 1000

juego = Juego(6,4)

E = []
R = []
loss = []
for episode in range(EPISODES):
    print(f'Partida {episode+1:d}/{EPISODES:d}...')
    juego.jugar()

    juego.reset()


'''
plt.figure()
plt.scatter( E, R )
plt.xlabel("Episodes")
plt.ylabel("Rewards")


plt.figure()
plt.hist( R )
plt.xlabel("Rewards")'''

plt.figure()
plt.plot( juego.policy.loss_history )


plt.show()