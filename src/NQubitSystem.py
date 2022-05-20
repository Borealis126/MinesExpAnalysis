# Must be python 2.7 compatible!
import numpy as np
import csv
from math import floor
from sympy import Matrix, zeros, eye, symbols
from sympy.physics.quantum import Dagger, TensorProduct


class Transition(object):
    def __init__(self, states):
        self.states = states

    @property
    def transitionStr(self):
        return transitionString(state_str(self.states[0]), state_str(self.states[1]))


class Qubit(object):
    def __init__(self, index, otherQubitIndices):
        self.index = index
        self.name = "Q" + str(self.index)
        self.otherQubitIndices = otherQubitIndices
        self.numQubits = len(self.otherQubitIndices)+1

    @staticmethod
    def indexFromName(name):
        return int(name[1:])


class NQubitSystem(object):
    def __init__(self, paramsFilePath):  # Q0Pio2X=[CF,tF,CROT,tROT]
        self.N = 0
        self.transitions = {}
        self.twoQubitValues = {}
        self.numPhotons = 0
        self.numLevels = 0

        self.qubits = []  # Initialized in loadQsysParams
        self.data = dict()
        self.loadQSysParams(paramsFilePath)  # All prior values are updated in loadQSysParams

    @property
    def dim(self):
        return  self.numLevels ** self.N

    def stateList(self, excitationList):
        """Excitation list is of the form [[i,n],[j,m],...]
        where i,j are component indices, and m,n are the excitations."""
        s = [0] * self.N
        for i in excitationList:
            s[i[0]] = i[1]
        return s

    def loadQSysParams(self, paramsFilePath):  # Takes as parameter the relevant NQS class
        with open(str(paramsFilePath)) as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                self.data[row[0]] = returnCorrectType(row[1])
        self.N = int(self.data["NumQubits"])
        self.numPhotons = int(self.data["NumPhotons"])# Only consider states with this total number of photons
        self.numLevels = int(self.data["NumLevels"])
        self.qubits = [Qubit(index=i, otherQubitIndices=[j for j in range(self.N) if j != i]) for i in range(self.N)]
        transitionStringList = [transitionString(state_str(i[0]), state_str(i[1])) for i in
                                transitionListTrunc(transitionList(self.N, self.numLevels), self.numPhotons)]
        for transitionStr in transitionStringList:
            states = transitionFromString(transitionStr)
            self.transitions[transitionStr] = Transition(states)

    @property
    def allStates(self):
        return allStates(self.N, self.numLevels)

    @property
    def transitionList(self):
        return transitionList(self.N, self.numLevels)


def allStates(N, numLevels):
    def baseRepresentation(num, base, places):
        numBaseList = [0] * places
        for i in range(places):
            nthPlace = base ** (places - i - 1)
            numBaseList[i] = int(floor(num / nthPlace))
            num = num - nthPlace * numBaseList[i]
        return numBaseList

    numAllStates = numLevels ** N
    return [baseRepresentation(i, numLevels, N) for i in range(numAllStates)]


def state_str(stateList):
    return '-'.join([str(i) for i in stateList])


def transitionString(stateStr1, stateStr2):
    return stateStr1 + "|" + stateStr2


def transitionFromString(transitionStr):
    return [[int(i) for i in stateStr.split("-")] for stateStr in transitionStr.split("|")]


def transitionList(N, numLevels):  # Calculates the transitions for given #qubits and #levels
    allStatesList = [state for state in allStates(N, numLevels)]
    transitions = []
    for state1 in allStatesList:
        for state2 in allStatesList:
            keepPair = True
            for index in range(N):
                if state2[index] < state1[index]:
                    keepPair = False
            if state1 == state2:
                keepPair = False
            if keepPair:
                transitions.append([state1, state2])
    return transitions


def transitionListIndex(N, numLevels, state1, state2):
    transList = transitionList(N, numLevels)
    return transList.index([state1, state2])


def transitionListTrunc(transitions, numPhotons):
    return [t for t in transitions if sum(t[0]) <= numPhotons and sum(t[1]) <= numPhotons]


def transitionListTruncIndex(N, numLevels, numPhotons, state1, state2):
    transList = transitionListTrunc(transitionList(N, numLevels), numPhotons)
    return transList.index([state1, state2])


def twoQubitPairs(N):
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append([i, j])
    return pairs


def twoQubitPairs_ordered(N):
    pairs = []
    for i in range(N):
        for j in range(N):
            pairs.append([i, j])
    return pairs


def twoQubitIndex(N, q0Index, q1Index):
    return N * q0Index + q1Index


def twoQubitMaxIndex(N, numPhotons):
    return numPhotons ** N


def returnCorrectType(strVal):
    try:
        return float(strVal)
    except ValueError:
        return str(strVal)
