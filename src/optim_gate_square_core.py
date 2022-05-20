import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import csv
from scipy.linalg import expm
from optimHelperFunctions import NielsenFidelity, gate
from math import log
from itertools import product

sys.path.append('O:\\68707\\JoelHoward\\PulseShaping')
from WaveformConstructorPrimitives_JH import *
import AdvancedWaveforms_JH as wfm_adv

dt = torch.cdouble  # datatype and precision
rdt = torch.double


def R_vals(driveType, numSegs):
    if driveType == ('X', 'X') or driveType == ('XY', '') or driveType == ('', 'XY'):
        R = torch.rand([numSegs, 2], dtype=rdt) * 2 * np.pi  # Random initialization (between 0 and 2pi)
    elif driveType == ('XY', 'XY'):
        R = torch.rand([numSegs, 4], dtype=rdt) * 2 * np.pi
    R.requires_grad = True  # set flag so we can backpropagate
    return R


def get_optimizer(opt, R, lr):
    if opt == 'ADAM':
        return torch.optim.Adam([R], lr=lr, weight_decay=1e-6)
    elif opt == 'ADAMW':
        return torch.optim.AdamW([R], lr=lr, weight_decay=0.01)
    elif opt == 'ADAMax':
        return torch.optim.Adamax([R], lr=lr, weight_decay=0.01)
    elif opt == 'RMSprop':
        return torch.optim.RMSprop([R], lr=lr, momentum=0.2)
    elif opt == 'Rprop':
        return torch.optim.Rprop([R], lr=lr)
    elif opt == 'Adadelta':
        return torch.optim.Adadelta([R], lr=lr)
    elif opt == 'Adagrad':
        return torch.optim.Adagrad([R], lr=lr)
    elif opt == 'SGD':
        return torch.optim.SGD([R], lr=lr, momentum=0.99, nesterov=True)
    elif opt == 'ASGD':
        return torch.optim.ASGD([R], lr=lr)
    else:
        return None


def get_scheduler(sched, optimizer, N_iter):
    if sched == 'Step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=N_iter / 10, gamma=0.9)
    elif sched == 'Exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    elif sched == 'Plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                          verbose=False, min_lr=0.03, factor=0.3,
                                                          patience=20)
    else:
        return None


def get_loss_in(sched):
    if sched == 'Plateau':
        return True
    else:
        return False


def optim(qSys, targetGate_name, gateT, driveType, maxRabis, numSegs, lr, N_iter, seed):
    targetGate_torch = gate(qSys, targetGate_name)
    H_parts = [torch.tensor(i, dtype=dt) for i in qSys.H_2Q_parts]
    rabiFracs = [maxRabis[i] / qSys.qubits[i].maxAmpStrength for i in range(qSys.N)]

    dim = qSys.numLevels ** 2
    gateT = torch.tensor(gateT, dtype=torch.double)
    segT = gateT/numSegs

    torch.manual_seed(seed)
    R = R_vals(driveType, numSegs)

    opt = 'SGD'
    optimizer = get_optimizer(opt, R, lr)
    sched = 'Plateau'
    loss_in = get_loss_in(sched)
    scheduler = get_scheduler(sched, optimizer, N_iter)

    infidelity_list = torch.zeros(N_iter)

    for ii in range(N_iter):  # optimization iterations
        optimizer.zero_grad()
        U = torch.eye(dim, dtype=dt)
        if driveType == ('XY', 'XY'):
            cosR = torch.cos(R) * torch.tensor([rabiFracs[0]] * 2 + [rabiFracs[1]] * 2)
        for m in range(numSegs):  # Hamiltonian evolution given pulses
            if driveType == ('XY', 'XY'):
                H = qSys.H_2Q(H_parts, cosR[m, 0], cosR[m, 1], cosR[m, 2], cosR[m, 3])
            elif driveType == ('', 'XY'):
                H = qSys.H_2Q(H_parts, 0, 0, cosR[m, 0], cosR[m, 1])
            elif driveType == ('XY', ''):
                H = qSys.H_2Q(H_parts, cosR[m, 0], cosR[m, 1], 0, 0)
            elif driveType == ('X', 'X'):
                H = qSys.H_2Q(H_parts, cosR[m, 0], 0, cosR[m, 1], 0)
            else:
                raise ValueError
            U = torch.matmul(torch.matrix_exp(-1j * H * segT), U)

        IF = 1 - NielsenFidelity(qSys.N, U, targetGate_torch)

        infidelity_list[ii] = IF.detach()

        IF.backward()  # use torch to calculate the gradient

        if optimizer is not None and scheduler is None:  # Update R
            optimizer.step()
            optimizer.zero_grad()
        elif optimizer is not None and scheduler is not None:
            optimizer.step()
            if loss_in:
                scheduler.step(IF)
            else:
                scheduler.step()
            optimizer.zero_grad()
        else:
            R.data.sub_(lr * R.grad.data)  # using data avoids overwriting tensor object
            R.grad.data.zero_()  # and it's respective grad info
            #    lr = decay_r*lr

    IF = infidelity_list.min().item()
    cosRData = cosR.detach().numpy().tolist()

    op_0 = Op()
    op_1 = Op()
    segT_val = segT.item()
    for seg in cosRData:
        if driveType == ('XY', 'XY'):
            q0x, q0y, q1x, q1y = seg
        elif driveType == ('X', 'X'):
            q0x, q1x = seg
            q0y = q1y = 0
        elif driveType == ('XY', ''):
            q0x, q0y = seg
            q1x = q1y = 0
        else:
            raise ValueError

        R_0, phi_0 = wfm_adv.pulseCartToPolar(q0x, q0y)
        op_0.pulseList.append(Pulse(R_0, segT_val, phi_0))

        R_1, phi_1 = wfm_adv.pulseCartToPolar(q1x, q1y)
        op_1.pulseList.append(Pulse(R_1, segT_val, phi_1))

    return ExpSlice(name="G", opList=[op_0, op_1]), IF


def optim_dumb(qSys, targetGate_name, gateT, driveType, numSegs, N_trials, seed):
    targetGate_torch = gate(qSys, targetGate_name)
    H_parts = [torch.tensor(i, dtype=dt) for i in qSys.H_2Q_parts]
    dim = qSys.numLevels ** 2
    gateT = torch.tensor(gateT, dtype=torch.double)
    segT = gateT/numSegs

    paramsPerSeg = sum([len(i) for i in driveType])
    numParams = numSegs*paramsPerSeg
    print('NumParams:', numParams)
    numDivs = int(N_trials ** (1/numParams))
    print('Resolution: ', numDivs)
    N_trials = int(numDivs ** numParams)
    print('Actual number of trials: ', N_trials)
    div = np.linspace(0, 2*np.pi, numDivs)
    trials = np.cos(np.array([i for i in product(*[div] * numParams)])).reshape(N_trials, numSegs, paramsPerSeg)
    if driveType == ('X', 'X'):
        trials = np.insert(np.insert(trials, 1, 0, axis=2), 3, 0, axis=2)

    entries = [[]]*N_trials
    for n in range(N_trials):
        if n % 1e4 == 0:
            print(str(n)+'/'+str(N_trials)+' complete')
        U = torch.eye(dim, dtype=dt)
        for m in range(numSegs):  # Hamiltonian evolution given pulses
            H = qSys.H_2Q(H_parts, trials[n, m, 0], trials[n, m, 1], trials[n, m, 2], trials[n, m, 3])
            U = torch.matmul(torch.matrix_exp(-1j * H * segT), U)
        IF = 1 - NielsenFidelity(qSys.N, U, targetGate_torch)
        entries[n] = [trials[n, :, :], IF]
    IFs = [i[1] for i in entries]
    minIndex = IFs.index(min(IFs))
    bestEntry = entries[minIndex]
    bestPulseData, bestIF = bestEntry

    op_0 = Op()
    op_1 = Op()
    segT_val = segT.item()
    for seg in bestPulseData:
        q0x, q0y, q1x, q1y = seg

        R_0, phi_0 = wfm_adv.pulseCartToPolar(q0x, q0y)
        op_0.pulseList.append(Pulse(R_0, segT.item(), phi_0))

        R_1, phi_1 = wfm_adv.pulseCartToPolar(q1x, q1y)
        op_1.pulseList.append(Pulse(R_1, segT_val, phi_1))

    return ExpSlice(name="G", opList=[op_0, op_1]), bestIF