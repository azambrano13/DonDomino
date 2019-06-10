from itertools import combinations 
import random as rnd
import definitions as defs

DEBUG = True

class Juego:
    def __init__(self, nMax:int, nJug:int) :
        self.nMax = nMax
        assert self.cantFichas() % nJug == 0, "No se pueden repartir las fichas!!!"
        self.nJug = nJug

        nFichas = int(self.cantFichas()/nJug)
        types = ['random','random','random','random']
        self.jugadores = [ Jugador( i, nFichas, nMax, types[i] ) for i in range(nJug) ]

        self.tablero = []

        self.jugar()

    def cantFichas(self) -> int:
        n = self.nMax
        return int( 0.5*( n + 1 )*( n + 2 ) )

    def jugar(self):
        self.repartir()
        if DEBUG : self.printJugadores()            

        idx = -1
        for i,j in enumerate(self.jugadores):
            if Ficha(self.nMax,self.nMax,0) in j.fichas:
                idx = i
                break

        k = 0
        nPas = 0
        acabar = False

        while not acabar:
            self.tablero, ficha, acabar, pasar = self.jugadores[idx].jugar( self.tablero )

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
        if DEBUG and nPas < self.nJug: print(f'Se acabó el Juego, ganó {idx:d}!!!')
        if DEBUG and nPas == self.nJug: print('Se cerró el Juego :(')

    def repartir(self):
        numeros = range(self.nMax+1)

        fichas = list( combinations( numeros, 2) )
        for i in numeros: fichas.append( (i,i) )

        jugIdx = [ i%self.nJug for i in range(len(fichas)) ]
        rnd.shuffle( jugIdx )
        for i, (ficha, jug) in enumerate( zip( fichas, jugIdx ) ): 
            f = Ficha(ficha[0],ficha[1],i)
            self.jugadores[jug].agregarFicha( f )
            #if DEBUG:print( f'{i:d}: {f}' )

    def printJugadores(self):
        for j in self.jugadores: print(j)
        print()

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
            ficha =  Ficha(self.nMax,self.nMax,0)
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

    def jugar( self, tablero, fichas=[] ) :
        if self.typeAgent == 'random': return self.jugarRandom( tablero )
        else :
            # Código para Policy Gradient
            pass

class Ficha:
    def __init__(self, n1:int, n2:int, id:int):
        self.n1 = n1
        self.n2 = n2
        self.id = id

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


j = Juego(6,4)
