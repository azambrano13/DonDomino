from domino import *
from definitions import *
import matplotlib.pyplot as plt

EPISODES = 100000
testEp=1000
testNum=200
pMax=6
jug=4
juego = Juego(pMax,jug)

E = []
R = []
loss = []
vict=[]
for episode in range(EPISODES+1):
    print(f'Partida {episode+1:d}/{EPISODES:d}...')
    ganador=juego.jugar(episode,0)

    juego.reset()

    if episode%testEp==0:
        vict.append(juego.test(testNum))
        print(vict)

#juego.agent.saveModel( 'test' )

'''
plt.figure()
plt.scatter( E, R )
plt.xlabel("Episodes")
plt.ylabel("Rewards")


plt.figure()
plt.hist( R )
plt.xlabel("Rewards")'''

plt.figure()
plt.plot(vict)
plt.show()