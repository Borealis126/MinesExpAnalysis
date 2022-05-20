#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 17:23:27 2021

@author: alex
"""
# imports
import torch
import numpy as np
from itertools import product
from pathlib import Path
from qutip import tensor, qeye
from scipy.linalg import expm, sqrtm

# Pauli Matricies
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
identity = np.array([[1, 0], [0, 1]])


def SUgroup(N): # Unitary group generation
    SU = []
    pauli_int = [1, 2, 3, 4]  # eq to [sx,sy,sz,identity]
    perms = list(product(pauli_int, repeat=N))  # all permutations of paulis
    for p in perms:  # mapping integers to pauli
        unitary = 1
        for pauli in p:
            if pauli == 1:
                unitary = torch.tensor(np.kron(unitary, sx), dtype=torch.cdouble)
            elif pauli == 2:
                unitary = torch.tensor(np.kron(unitary, sy), dtype=torch.cdouble)
            elif pauli == 3:
                unitary = torch.tensor(np.kron(unitary, sz), dtype=torch.cdouble)
            elif pauli == 4:
                unitary = torch.tensor(np.kron(unitary, identity), dtype=torch.cdouble)
        SU.append(unitary)
    return SU


def NielsenFidelity(N, U_Exp, target_gate):
    fidelity = 0
    d = 2 ** N
    SU = SUgroup(N)
    for i in range(0, len(SU)):
        eps_U = torch.matmul(torch.matmul(U_Exp, SU[i]), (U_Exp.conj().T))
        target_U = torch.matmul(torch.matmul(target_gate, (SU[i].conj().T)), (target_gate.conj().T))
        fidelity = fidelity + torch.trace(torch.matmul(target_U, eps_U))
    fidelity = abs(fidelity + d * d) / (d * d * (d + 1))
    return fidelity


def gate(qSys, gate_name):
    ket00 = qSys.npState([0, 0])
    ket01 = qSys.npState([0, 1])
    ket10 = qSys.npState([1, 0])
    ket11 = qSys.npState([1, 1])
    eye = qeye(qSys.numLevels)
    dt = torch.cdouble
    if gate_name == "Q0_Pi":
        return torch.tensor(np.kron(qSys.rotationY(np.pi), eye), dtype=dt)
    elif gate_name == "Q1_Pi":
        return torch.tensor(np.kron(eye, qSys.rotationY(np.pi)), dtype=dt)
    elif gate_name == "Q0_Pio2":
        return torch.tensor(np.kron(qSys.rotationY(np.pi / 2), eye), dtype=dt)
    elif gate_name == "Q1_Pio2":
        return torch.tensor(np.kron(eye, qSys.rotationY(np.pi / 2)), dtype=dt)
    elif gate_name == "CNOT":
        return torch.tensor(
            ket00 * ket00.conj().T + ket01 * ket01.conj().T + ket10 * ket11.conj().T + ket11 * ket10.conj().T,
            dtype=dt)
    elif gate_name == "iSWAP":
        return torch.tensor(expm(1j*np.pi/4*(np.kron(qSys.sx, qSys.sx)+np.kron(qSys.sy, qSys.sy))), dtype=dt)
    elif gate_name == "sqrtSWAP":
        SWAP = ket00 * ket00.conj().T + ket01 * ket10.conj().T + ket10 * ket01.conj().T + ket11 * ket11.conj().T
        sqrtSWAP = sqrtm(SWAP)
        return torch.tensor(sqrtSWAP, dtype=dt)
    elif gate_name == "SWAP":
        return torch.tensor(
            ket00 * ket00.conj().T + ket01 * ket10.conj().T + ket10 * ket01.conj().T + ket11 * ket11.conj().T,
            dtype=dt)
    else:
        return None