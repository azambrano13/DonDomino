from itertools import combinations 
import random as rnd
from copy import deepcopy as dpc
from definitions import *
import numpy as np

DEBUG = False

class Juego :

    def __init__(self, nMax:int, nJug:int) :
        self.nMax = nMax
        assert self.cantFichas() % nJug == 0, "No se pueden repartir las fichas!!!"
        self.nJug = nJug

        nFichas = int(self.cantFichas()/nJug)
        types = [1,0,0,0]
        self.jugadores = [ Jugador( i, nFichas, nMax, types[i] ) for i in range(nJug) ]

        self.tablero = []
        self.fichas = []

        cantFichas = self.cantFichas()
        # Tamaño del vector de estados y acciones
        self.actionSize = cantFichas * 2 + 1
        self.stateSize = cantFichas * nJug + nMax + 1
        # Crea al agente
        self.agent = DQNAgent(self.stateSize, self.actionSize)

    def cantFichas(self) -> int:
        n = self.nMax
        return int( 0.5*( n + 1 )*( n + 2 ) )

    def repartir(self):
        numeros = range(self.nMax+1)

        fichas = list( combinations( numeros, 2) )
        for i in numeros: fichas.append( (i,i) )

        jugIdx = [ i % self.nJug for i in range (len( fichas ) ) ]
        rnd.shuffle( jugIdx )
        for ficha, jug in zip( fichas, jugIdx ): 
            f = Ficha(ficha[0],ficha[1])
            self.fichas.append( dpc( f ) )
            self.jugadores[jug].agregarFicha( f )
    
    def reset(self):
         self.tablero = []
         self.fichas = []
         for jugador in self.jugadores : jugador.fichas=[]

    def printJugadores(self):
        for j in self.jugadores: print(j)

    def printTablero(self):
        s = f'Tablero:\n\t'
        for ficha in self.tablero: s += str(ficha) + "  "
        print(s+"\n")

    def jugar(self,ep,prueba):
        self.repartir()
        if DEBUG : self.printJugadores()            

        idx = -1
        for i,j in enumerate(self.jugadores):
            if Ficha(self.nMax,self.nMax) in j.fichas:
                idx = i
                break

        k = 0
        nPas = 0
        acabar = False

        self.encFichas = Encoder( self.fichas )
        self.encNum = Encoder( range(self.nMax+1) )

        while not acabar:
            if self.jugadores[idx].typeAgent==0:
                self.tablero, ficha, acabar, pasar = self.jugadores[idx].jugarRandom( self.tablero, self )
            elif self.jugadores[idx].typeAgent==1:
                self.tablero, ficha, acabar, pasar = self.jugadores[idx].jugarRL(self.tablero, self, self.agent, prueba)

            if DEBUG : print(f'Turno {k:d}, el Jugador {idx:d} juega la Ficha {ficha}')

            if pasar: nPas += 1
            else: nPas = 0
            ganador = idx
            if nPas == self.nJug:
                acabar = True
                suma=100
                for j in range(len(self.jugadores)):
                    for f in self.jugadores[j].fichas:
                        suma1=f.n1+f.n2
                    if suma1<suma:
                        suma=suma1
                        ganador=j
            if DEBUG : self.printJugadores()
            if DEBUG : self.printTablero()

            #Determina ganador de la partida
            idx += 1
            idx %= self.nJug
            k += 1

        idx = (idx-1)%self.nJug

        # self.rewards = dpc( self.policy.reward_episode )

        # if len(self.rewards) > 0 : update_policy(self.policy, self.optim)

        states, actions, nextStates, rewards, done = [], [], [], [], []
        for jugador in self.jugadores : 
            states.extend( jugador.states )
            actions.extend( jugador.actions )
            nextStates.extend( jugador.nextStates )
            nextStates.extend(np.zeros((1,self.stateSize)).astype(int))
            done.extend((np.zeros(len(jugador.nextStates))))
            done.extend([1])
            rewards.extend(jugador.rewards)

        self.agent.remember(states, actions, rewards, nextStates, done)
        #Se esperan ciertos juegos hasta empezar a entrenar al agente

        if ep % 2500==0 and len(self.agent.memory)>500:
            self.agent.replay(256)
        self.reset()

        return ganador

        #train=np.concatenate(states,actions,axis=1)

        '''if len(states) > 0 : 
            self.policy.update_policy_supervised( np.array(states,dtype=np.float32), np.array(actions) )

        if DEBUG and nPas < self.nJug: print(f'Se acabó el Juego, ganó {idx:d}!!!')
        if DEBUG and nPas == self.nJug: print('Se cerró el Juego :(')

        return self.policy.loss_history'''

    #Verifica cuántas ganó RL
    def test(self,Nume):
        vict=0
        ganador=[]
        for i in range(Nume):
            ganador.append(self.jugar(1,0))
            if ganador[i]==0:
                vict=vict+1
        return vict


        
class Jugador :

    def __init__(self, id:int, nFichas:int, nMax:int, typeAgent):
        self.id = id
        self.fichas = []
        self.nMax = nMax
        
        self.typeAgent = typeAgent

        self.states = []
        self.actions = []
        self.nextStates = []
        self.rewards = []
        self.done=[]

    def __str__(self):
        s = f'Jugador {self.id:d}:\n\t'
        for ficha in self.fichas: s += str(ficha) + "  "
        return s

    def agregarFicha( self, ficha ):
        self.fichas.append( ficha )

    def jugarRandom( self, tablero, juego ):
        # Al inicio siempre se juega el [6|6]
        if not tablero:
            ficha = Ficha(self.nMax, self.nMax)
            self.fichas.remove(ficha)
            tablero.append(ficha)

            return tablero, ficha, len(self.fichas) == 0, ficha is None

        # Fichas de todos los jugadores
        fichas = [juego.jugadores[(self.id + i) % juego.nJug].fichas for i in range(juego.nJug)]

        # Encoders
        encoder_to_fichas = juego.encFichas
        encoder_number = juego.encNum

        # Posibles Jugadas
        nJug1, nJug2 = tablero[0].n1, tablero[-1].n2

        state = []
        #Concatena estado siguiente después de que el estado original sea mayor a 1
        if len(self.states) > 0:
            for f in fichas: state.extend(encoder_to_fichas.encode(f))
            state.extend(encoder_number.encode([nJug1, nJug2]))
            state = np.array(state)
            # state = torch.tensor( state, dtype=torch.float )
            # state = state.reshape( [-1,1] )
            self.nextStates.append(state)
            self.states.append(state)
        else:
            for f in fichas: state.extend(encoder_to_fichas.encode(f))
            state.extend(encoder_number.encode([nJug1, nJug2]))
            state = np.array(state)
            # state = torch.tensor( state, dtype=torch.float )
            # state = state.reshape( [-1,1] )
            self.states.append(state)

        # Lado para poner ficha
        idx1, idx2 = False, False

        # Ficha a poner
        ficha = None

        for f in self.fichas:
            if nJug1 in f: ficha, idx1 = f, True
            if nJug2 in f: ficha, idx2 = f, True
            if idx1 or idx2: break

        if ficha is not None:
            self.fichas.remove(ficha)
            if idx1:
                if ficha.n2 == nJug1:
                    tablero = [ficha] + tablero
                else:
                    tablero = [ficha.inv()] + tablero
            else:
                if ficha.n1 == nJug2:
                    tablero = tablero + [ficha]
                else:
                    tablero = tablero + [ficha.inv()]

        action = -1
        if ficha is None:
            action = [2 * juego.cantFichas()]
            reward=-1
        else:
            action = [np.argmax(encoder_to_fichas.encode([ficha]))]
            reward=1
        self.actions.append(action)
        self.rewards.append(reward)

        return tablero, ficha, len(self.fichas) == 0, ficha is None

    def jugarRL( self, tablero, juego, agent, prueba ) :
        # Al inicio siempre se juega el [6|6]
        if not tablero:
            ficha = Ficha(self.nMax, self.nMax)
            self.fichas.remove(ficha)
            tablero.append(ficha)

            return tablero, ficha, len(self.fichas) == 0, ficha is None

        # Fichas de todos los jugadores
        fichas = [juego.jugadores[(self.id + i) % juego.nJug].fichas for i in range(juego.nJug)]

        # Encoders
        encoder_to_fichas = juego.encFichas
        encoder_number = juego.encNum

        # Posibles Jugadas
        nJug1, nJug2 = tablero[0].n1, tablero[-1].n2

        # Lado para poner ficha
        idx1, idx2 = False, False

        state = []
        # Concatena estado siguiente después de que el estado original sea mayor a 1
        if len(self.states) > 0:
            for f in fichas: state.extend(encoder_to_fichas.encode(f))
            state.extend(encoder_number.encode([nJug1, nJug2]))
            state = np.array(state)
            # state = torch.tensor( state, dtype=torch.float )
            # state = state.reshape( [-1,1] )
            self.nextStates.append(state)
            self.states.append(state)
        else:
            for f in fichas: state.extend(encoder_to_fichas.encode(f))
            state.extend(encoder_number.encode([nJug1, nJug2]))
            state = np.array(state)
            # state = torch.tensor( state, dtype=torch.float )
            # state = state.reshape( [-1,1] )
            self.states.append(state)

        tengo = False
        ficha_jugada=None
        #Aplica o no exploración dependiendo de si debe aprender o probar
        if prueba==0: action = agent.act(state)
        else: action = agent.test(state)
        if action==56:
            reward=-1
        else:
            lado = action // 28
            if lado==0: idx1=True
            if lado==1: idx2= True
            fic = np.zeros(28)
            fic[action % 28] = 1
            ficha_jugada = encoder_to_fichas.decode(fic)[0]

            # Verifico que tenga la ficha al asignar recompensa
            for ficha in self.fichas:
                tengo = ficha == ficha_jugada
                if tengo: break

            reward = 1

            if not tengo: reward -= 30

        self.actions.append(action)
        self.rewards.append(reward)

        if tengo:
            self.fichas.remove(ficha)
            if idx1:
                if ficha.n2 == nJug1:
                    tablero = [ficha] + tablero
                else:
                    tablero = [ficha.inv()] + tablero
            else:
                if ficha.n1 == nJug2:
                    tablero = tablero + [ficha]
                else:
                    tablero = tablero + [ficha.inv()]

            return tablero, ficha_jugada, len(self.fichas) == 0, ficha_jugada is None

        else:
            return tablero, ficha_jugada, len(self.fichas) == 0 , ficha_jugada is None

        print(f'Tablero: {tablero} ')
        print(f'Mis Fichas: {self.fichas}')
        print(f'\t Tratar de jugar ficha {ficha_jugada} - Reward: {reward:d}')


class Ficha :
    def __init__(self, n1:int, n2:int):
        self.n1 = n1
        self.n2 = n2

    def __str__(self):
        return f'[{self.n1:d}|{self.n2:d}]'

    __repr__ = __str__

    def __eq__(self, value):
        if not isinstance(value, Ficha): return False
        return (self.n1 == value.n1 and self.n2 == value.n2) or (self.n1 == value.n2 and self.n2 == value.n1)

    def __contains__(self, key):
        return self.n1 == key or self.n2 == key

    def inv(self):
        self.n1, self.n2 = self.n2, self.n1
        return self

class Encoder : 
    def __init__(self, elements):
        self.elements = elements

    def encode( self, newElements ) :
        e = len(self.elements)*[0]
        for ne in newElements : e[ self.elements.index( ne ) ] = 1
        return e

    def decode( self, codeElements ) :
        e = []
        for i,ce in enumerate(codeElements) : 
            if ce == 1 : e.append( self.elements[i] )
        return e
