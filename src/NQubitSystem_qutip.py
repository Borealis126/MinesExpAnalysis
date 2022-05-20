import sys
sys.path.append("O:\\68707\\JoelHoward\\PulseShaping")
import NQubitSystem_PulseGen as NQS_PG
from qutip import basis, tensor, qeye
from sympy import Matrix, zeros, eye, symbols
from sympy.physics.quantum import Dagger, TensorProduct
import numpy as np
from scipy.linalg import expm, sqrtm
from cmath import phase


class Transition(NQS_PG.Transition):
    def __init__(self, states):
        super(Transition, self).__init__(states)
        self.dSym = symbols('d_' + self.transitionStr, real=True, nonnegative=True)
        self.freq = symbols('omega_' + self.transitionStr, real=True, nonnegative=True)


class NQubitSystem_QT(NQS_PG.NQubitSystem_PG):
    def __init__(self, paramsFilePath):   # Q0Pio2X=[CF,tF,CROT,tROT]
        super(NQubitSystem_QT, self).__init__(paramsFilePath)
        for transitionStr, transition in self.transitions.items():
            tempTransition = Transition(transition.states)
            self.transitions[transitionStr] = tempTransition  # Upgrade Transition class.

        self.numLevels = 2
        self.ket0 = self.npState([0])# Do everything via numpy
        self.ket1 = self.npState([1])
        self.sx = self.ket1 * self.ket0.conj().T + self.ket0 * self.ket1.conj().T
        self.sy = (-1j * (self.ket0 * self.ket1.conj().T - self.ket1 * self.ket0.conj().T))
        self.sz = -(self.ket1 * self.ket1.conj().T - self.ket0 * self.ket0.conj().T)
        self.s_plus = 1/2*(self.sx-1j*self.sy)
        self.s_minus = 1 / 2 * (self.sx + 1j * self.sy)

        self.ZI = np.kron(self.sz, qeye(self.numLevels))
        self.IZ = np.kron(qeye(self.numLevels), self.sz)
        self.ZZ = np.kron(self.sz, self.sz)

        self.X_zero = np.kron(self.sx, self.ket0 * self.ket0.conj().T)
        self.zero_X = np.kron(self.ket0 * self.ket0.conj().T, self.sx)
        self.X_one  = np.kron(self.sx, self.ket1 * self.ket1.conj().T)
        self.one_X  = np.kron(self.ket1 * self.ket1.conj().T, self.sx)
        self.Y_zero = np.kron(self.sy, self.ket0 * self.ket0.conj().T)
        self.zero_Y = np.kron(self.ket0 * self.ket0.conj().T, self.sy)
        self.Y_one  = np.kron(self.sy, self.ket1 * self.ket1.conj().T)
        self.one_Y  = np.kron(self.ket1 * self.ket1.conj().T, self.sy)

    def transitionPaths(self, state, path=None):
        paths = []
        if path is None:
            path = []
        path.append(state)
        childTransitions = []
        for transitionStr, transition in self.transitions.items():
            if transition.states[1] == state:
                childTransitions.append(transition)
        if childTransitions:
            for transition in childTransitions:
                paths.extend(self.transitionPaths(transition.states[0], path[:]))
        else:
            paths.append(path)
        return paths

    def stateEnergy(self, state):
        allPaths = self.transitionPaths(state)
        E_vals = []
        for path in allPaths:
            E = 0
            for i in range(len(path) - 1):
                E += self.transitions[NQS_PG.transitionString(NQS_PG.state_str(path[i + 1]), NQS_PG.state_str(path[i]))].freq
            E_vals.append(E)
        return E_vals

    def npState(self, state):
        return self.qutipState(state).full()

    def qutipState(self, state):# state is like [1,0]. Standard basis
        return tensor([basis(self.numLevels, i) for i in state])

    def sympyState(self, state):
        return TensorProduct(*[self.sympyBasis(i) for i in state])

    def sympySingleQubitOperator(self, qubitIndex, operator):
        tensorArray = [eye(self.numLevels) for i in range(self.N)]
        tensorArray[qubitIndex] = operator
        return TensorProduct(*tensorArray)

    def sympyBasis(self, numExcitations):
        basisState = zeros(self.numLevels, 1)
        basisState[numExcitations] = 1
        return basisState

    def H_2Q(self, parts, q0x, q0y, q1x, q1y):
        """Values are in the range -1 to 1, which is normalized to the maxAmpStrength."""
        H0, Hx0, Hy0, Hx1, Hy1 = parts
        return H0 + q0x * Hx0 + q0y * Hy0 + q1x * Hx1 + q1y * Hy1

    @property
    def H_2Q_parts(self):
        # print('R Values: ', self.data['0-0|0-1 Dipole Strength (MHz)'],
        #                     self.data['0-0|1-0 Dipole Strength (MHz)'],
        #                     self.data['0-1|1-1 Dipole Strength (MHz)'],
        #                     self.data['1-0|1-1 Dipole Strength (MHz)'])
        r0 = self.data['0-1|1-1 Dipole Strength (MHz)'] / self.data['0-0|1-0 Dipole Strength (MHz)']
        r1 = self.data['1-0|1-1 Dipole Strength (MHz)'] / self.data['0-0|0-1 Dipole Strength (MHz)']

        maxR_w_q0 = self.qubits[0].maxAmpStrength * 2 * np.pi
        maxR_w_q1 = self.qubits[1].maxAmpStrength * 2 * np.pi

        gz_w = self.twoQubitValues[0][1]['ZZ'] * 2 * np.pi
        H0 = gz_w / 4 * (-self.ZI - self.IZ + self.ZZ)
        Hx0 = maxR_w_q0 / 2 * (self.X_zero + r0 * self.X_one)
        Hy0 = maxR_w_q1 / 2 * (self.Y_zero + r0 * self.Y_one)
        Hx1 = maxR_w_q0 / 2 * (self.zero_X + r1 * self.one_X)
        Hy1 = maxR_w_q1 / 2 * (self.zero_Y + r1 * self.one_Y)
        return [H0, Hx0, Hy0, Hx1, Hy1]

    def pulseToUnitary_shape(self, Q0Pulse, Q1Pulse):
        numSegs = 300
        tVals = np.linspace(0, Q0Pulse.duration, numSegs)  # Pulses should have the same duration
        segLength = Q0Pulse.duration / numSegs
        U_tot = np.identity(4)
        q0x_unshaped = Q0Pulse.amp * np.cos(Q0Pulse.phase)
        q0y_unshaped = Q0Pulse.amp * np.sin(Q0Pulse.phase)
        q1x_unshaped = Q1Pulse.amp * np.cos(Q1Pulse.phase)
        q1y_unshaped = Q1Pulse.amp * np.sin(Q1Pulse.phase)

        H_segs = []
        for i, t in enumerate(tVals):
            q0x = q0x_unshaped * Q0Pulse.shapeFunc(t)
            q0y = q0y_unshaped * Q0Pulse.shapeFunc(t)
            q1x = q1x_unshaped * Q1Pulse.shapeFunc(t)
            q1y = q1y_unshaped * Q1Pulse.shapeFunc(t)

            H_seg = self.H_2Q(self.H_2Q_parts, q0x, q0y, q1x, q1y)

            U_seg = unitaryOp(H_seg, segLength)
            H_segs.append(H_seg)
            U_tot = np.matmul(U_seg, U_tot)
        return U_tot

    def pulseToUnitary(self, Q0Pulse, Q1Pulse):

        q0x = Q0Pulse.amp * np.cos(Q0Pulse.phase)
        q0y = Q0Pulse.amp * np.sin(Q0Pulse.phase)

        q1x = Q1Pulse.amp * np.cos(Q1Pulse.phase)
        q1y = Q1Pulse.amp * np.sin(Q1Pulse.phase)
        H_seg = self.H_2Q(self.H_2Q_parts, q0x, q0y, q1x, q1y)

        return unitaryOp(H_seg, Q0Pulse.duration)

    def gateExpSliceToUnitaryOpList(self, gateExpSlice):
        numSegs = len(gateExpSlice.opList[0].pulseList)
        unitaryOpList = []
        for i in range(numSegs):
            Q0Pulse = gateExpSlice.opList[0].pulseList[i]
            Q1Pulse = gateExpSlice.opList[1].pulseList[i]

            unitaryOpList.append(self.pulseToUnitary_shape(Q0Pulse, Q1Pulse))
        return unitaryOpList

    def rotationX(self, theta):
        return expm(-1j * theta / 2 * self.sx)

    def rotationY(self, theta):
        return expm(-1j * theta / 2 * self.sy)

    def rotationZ(self, theta):
        return expm(-1j * theta / 2 * self.sz)


def unitaryFromOpList(unitaryOpList):
    # List is ordered from first applied to last, reverse of equation.
    fullOp = unitaryOpList[0]
    for op in unitaryOpList[1:]:
        fullOp = op.dot(fullOp)
    return fullOp


def evolution(U, rho):
    return U.dot(rho).dot(U.conj().T)


def unitaryOp(H, t):
    return expm(-1j*H*t)


def fidelity(A, B):
    return np.trace(sqrtm(sqrtm(A).dot(B).dot(sqrtm(A)))).real**2


def purity(A):
    return np.trace(A.dot(A)).real
