from itertools import combinations 
import random as rnd
from copy import deepcopy as dpc
import definitions as defs
import numpy as np
from definitions import *
from torch.distributions import Categorical
DEBUG = True

class Juego:
    def __init__(self, nMax:int, nJug:int) :
        self.nMax = nMax
        assert self.cantFichas() % nJug == 0, "No se pueden repartir las fichas!!!"
        self.nJug = nJug

        nFichas = int(self.cantFichas()/nJug)
        types = ['no random','random','random','random']
        self.jugadores = [ Jugador( i, nFichas, nMax, types[i] ) for i in range(nJug) ]

        self.tablero = []
        self.fichas = []
        self.optim=optim.Adam(self.jugadores[0].policy.parameters())

        # self.jugar()

    def cantFichas(self) -> int:
        n = self.nMax
        return int( 0.5*( n + 1 )*( n + 2 ) )

    def jugar(self):
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
        fichas = [ dpc( self.jugadores[i].fichas ) for i in range(self.nJug) ]

        self.encFichas = Encoder( self.fichas )
        self.encNum = Encoder( range(self.nMax+1) )
        
        if DEBUG : 
            # print( self.fichas )
            # print( fichas )
            # fTemp = Ficha( 0,4 )
            # print( self.fichas.index( fTemp ) )
            print( self.encFichas.encode( fichas[0] ) )

        while not acabar:
            self.tablero, ficha, acabar, pasar = self.jugadores[idx].jugar( self.tablero, fichas )

            if DEBUG : print(f'Turno {k:d}, el Jugador {idx:d} juega la Ficha {ficha}')

            if pasar: nPas += 1
            else: nPas = 0

            if nPas == self.nJug: acabar = True

            if DEBUG : self.printJugadores() 
            if DEBUG : self.printTablero()

            idx += 1
            idx %= self.nJug
            k += 1


        idx = (idx-1)%self.nJug
        r=calcular_recompenza(idx,0.9,)
        self.jugadores[0].policy.reward_episode.extend(r)



        update_policy(self.jugadores[0].policy, self.optim)

        if DEBUG and nPas < self.nJug: print(f'Se acabó el Juego, ganó {idx:d}!!!')
        if DEBUG and nPas == self.nJug: print('Se cerró el Juego :(')

    def repartir(self):
        numeros = range(self.nMax+1)

        fichas = list( combinations( numeros, 2) )
        for i in numeros: fichas.append( (i,i) )

        jugIdx = [ i%self.nJug for i in range(len(fichas)) ]
        rnd.shuffle( jugIdx )
        for ficha, jug in zip( fichas, jugIdx ): 
            f = Ficha(ficha[0],ficha[1])
            self.fichas.append( dpc( f ) )
            self.jugadores[jug].agregarFicha( f )
            #if DEBUG:print( f'{i:d}: {f}' )
    def reset(self):
         self.tablero = []
         self.fichas = []
         for i in range(len(self.jugadores)):
            self.jugadores[i].fichas=[]

    def printJugadores(self):
        for j in self.jugadores: print(j)


    def printTablero(self):
        s = f'Tablero:\n\t'
        for ficha in self.tablero: s += str(ficha) + "  "
        print(s+"\n")
        
class Jugador:
    def __init__(self, id:int, nFichas:int, nMax:int, typeAgent):
        self.id = id
        self.fichas = []
        self.nMax = nMax
        
        self.typeAgent = typeAgent
        self.policy = None if typeAgent == 'random' else defs.Policy( 147 )

    def __str__(self):
        s = f'Jugador {self.id:d}:\n\t'
        for ficha in self.fichas: s += str(ficha) + "  "
        return s

    def agregarFicha( self, ficha ):
        self.fichas.append( ficha )

    def jugarRandom( self, tablero ):
        if not tablero:
            ficha =  Ficha(self.nMax,self.nMax)
            self.fichas.remove( ficha )
            tablero.append( ficha )
        else:
            ficha = None
            nJug1, nJug2 = tablero[0].n1, tablero[-1].n2
            idx1, idx2 = False, False

            for f in self.fichas:
                if nJug1 in f : ficha, idx1 = f, True
                if nJug2 in f : ficha, idx2 = f, True
                if idx1 or idx2: break

            if ficha is not None:
                self.fichas.remove( ficha )
                if idx1: 
                    if ficha.n2 == nJug1: tablero = [ficha] + tablero
                    else : tablero = [ficha.inv()] + tablero
                else :
                    if ficha.n1 == nJug2: tablero = tablero + [ficha]
                    else : tablero = tablero + [ficha.inv()]

        return tablero, ficha, len( self.fichas ) == 0, ficha is None

    def jugar( self, tablero, fichas=[],encoder_to_fichas=None,encoder_number=None) :
        if self.typeAgent == 'random': return self.jugarRandom( tablero )
        else :
            # Código para Policy Gradient
            state=encoder_to_fichas.encode(tablero)
            for ficha in fichas:
                state.extend(encoder_to_fichas.encode(ficha))
            nJug1, nJug2 = tablero[0].n1, tablero[-1].n2
            state.extend(encoder_number.encode([nJug1,nJug2]))
            state = np.array(state)
            # state = torch.from_numpy(state).type(torch.FloatTensor)
            state = self.policy(Variable(state))
            c = Categorical(state)
            action = c.sample()

            # Add log probability of our chosen action to our history

            if self.policy.policy_history.dim() != 0:
                self.policy.policy_history = torch.cat([self.policy.policy_history, c.log_prob(action)])
            else:
                self.policy.policy_history = (c.log_prob(action))

            nJug1, nJug2 = tablero[0].n1, tablero[-1].n2

            # Asumiendo que devuelve un vector de 28X2 con 1 en las fichas que toene el numero nJug1 o el nJug2
            fichas_posibles=np.array(encoder_number.encode([nJug1, nJug2]))
            idx1, idx2 = False, False
            if fichas_posibles[action]!=1:
                ficha=None
            else:



            if ficha is not None:
                self.fichas.remove( ficha )
                if idx1:
                    if ficha.n2 == nJug1: tablero = [ficha] + tablero
                    else : tablero = [ficha.inv()] + tablero
                else :
                    if ficha.n1 == nJug2: tablero = tablero + [ficha]
                    else : tablero = tablero + [ficha.inv()]

            return


class Ficha:
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

class Encoder: 
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
