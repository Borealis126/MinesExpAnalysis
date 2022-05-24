from src import qdpm
from qutip import identity, Qobj, basis
from qutip.qip.operations import rx, ry
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import src.NQubitSystem_qutip as NQS_QT
import src.AdvancedWaveforms_JH as wfm_adv
from src.pulseGenFuncs import Paths
from src.optimHelperFunctions import NielsenFidelity, gate
import torch
from copy import deepcopy

preparation = [identity(2),
                   rx(np.pi),
                   ry(np.pi / 2),
                   ry(-np.pi / 2),
                   rx(-np.pi / 2),
                   rx(np.pi / 2)]
rotation = [identity(2),
            rx(-np.pi / 2),
    ry(np.pi / 2)]
init_state = [basis(2, 0), basis(2, 0)]


def gateData(gateName, basePath, date, gateIndices):
    qSys = NQS_QT.NQubitSystem_QT(basePath / 'G0' / 'results' / 'gate' / date)
    qSys.numLevels = 2

    TheoryGate = gate(qSys, gateName).numpy()

    gates = wfm_adv.loadGates(qSys, basePath / 'gates.json')
    U_vals = [NQS_QT.unitaryFromOpList(qSys.gateExpSliceToUnitaryOpList(gates[g_i])) for g_i in gateIndices]
    G_T = [sum([pulse.duration for pulse in gates[g_i].opList[0].pulseList]) for g_i in gateIndices]
    G_T_theory = np.array(G_T) / (np.pi / (qSys.twoQubitValues[0][1]['ZZ'] * 2 * np.pi))
    return {'U_vals': U_vals,
            'G_T': G_T,
            'G_T_theory': G_T_theory,
            'U_theory': TheoryGate}


def getTomos(basePath, date, gateIndices, gateDataVals):


    studyPath_I = Path('.') / 'ExpData' / 'ProcessTomographyTest' / 'id'
    paths_I = Paths(studyPath_I)

    R_I = qdpm.ProcessTomography(qdpm.Experiment(paths_I.resultsPath(date)), init_state, preparation, rotation)._R_mle

    tomos = list()

    for g_i in gateIndices:
        studyPath = Paths(basePath / ('G' + str(g_i)))
        tomo = qdpm.ProcessTomography(qdpm.Experiment(studyPath.resultsPath(date)), init_state, preparation, rotation)
        U = gateDataVals['U_vals'][g_i]
        R_U = tomo._R_mle
        R_spamFree = qdpm.R_SPAMfree(U, R_U, R_I)
        tomo._R_mle = R_spamFree.astype('float64')
        tomos.append(tomo)
    return tomos


def IFs(basePath, date, gateIndices, gateDataVals, tomos):
    qSys = NQS_QT.NQubitSystem_QT(basePath / 'G0' / 'results' / 'gate' / date)
    qSys.numLevels = 2

    TheoryGate = gateDataVals['U_theory']

    G_IF_Exp_Theory = list()
    G_IF_Theory_TheoryGate = list()
    G_IF_Exp_TheoryGate = list()
    for g_i in gateIndices:
        tomo = tomos[g_i]

        G_IF_Exp_Theory.append(1 - tomo.fidelity(Qobj(gateDataVals['U_vals'][g_i], dims=[[2, 2], [2, 2]])))
        G_IF_Theory_TheoryGate.append(1 - NielsenFidelity(qSys.N,
                                                          torch.tensor(gateDataVals['U_vals'][g_i],
                                                                       dtype=torch.cdouble),
                                                          torch.tensor(TheoryGate,
                                                                       dtype=torch.cdouble)).item())
        G_IF_Exp_TheoryGate.append(1 - tomo.fidelity(Qobj(TheoryGate, dims=[[2, 2], [2, 2]])))
    return {'G_IF_Exp_Theory': G_IF_Exp_Theory,
            'G_IF_Theory_TheoryGate': G_IF_Theory_TheoryGate,
            'G_IF_Exp_TheoryGate': G_IF_Exp_TheoryGate}

    # if g_i == 15:
    #     tomo.plot()
    #     plt.savefig('CNOT75hist_exp.png', dpi=500)

    #         tomo.plot_theory(Qobj(TheoryGate, dims=[[2,2],[2,2]]))
    #         plt.savefig('CNOT75hist_theory.png', dpi=500)


def fidelityCurve(gateName, rabiStrengthMHz, gateDataVals, IFdict):
    plt.plot(gateDataVals['G_T_theory'], IFdict['G_IF_Exp_Theory'], 'bo-', label='IF Exp/Theory')
    plt.plot(gateDataVals['G_T_theory'], IFdict['G_IF_Theory_TheoryGate'], 'go-', label='IF Theory/' + gateName)
    plt.plot(gateDataVals['G_T_theory'], IFdict['G_IF_Exp_TheoryGate'], 'ro-', label='IF Exp/' + gateName)

    plt.legend()
    plt.title(gateName + ' Q0-XY Q1-XY square 4segs $\Omega=$' + str(rabiStrengthMHz) + 'MHz')
    plt.xlabel('Gate T (units of pi/gz)')
    plt.ylabel('Infidelity')
    plt.savefig(gateName + '_' + str(rabiStrengthMHz) + '_curves.png', format='png')
    plt.show()


def fidelityCurve_noise(gateName, rabiStrengthMHz, gateDataVals, IFdict, noiseDictIFs):
    plt.errorbar(x=gateDataVals['G_T_theory'],
                 y=noiseDictIFs['G_IF_Exp_Theory_Mean'],
                 yerr=noiseDictIFs['G_IF_Exp_Theory_STD'],
                 fmt='b-', label='IF Exp/Theory')
    plt.errorbar(x=gateDataVals['G_T_theory'],
                 y=noiseDictIFs['G_IF_Exp_TheoryGate_Mean'],
                 yerr=noiseDictIFs['G_IF_Exp_TheoryGate_STD'],
                 fmt='r-', label='IF Exp/' + str(gateName))
    plt.plot(gateDataVals['G_T_theory'], IFdict['G_IF_Theory_TheoryGate'], 'go-', label='IF Theory/CNOT')
    plt.legend()
    plt.title(gateName + ' Q0-XY Q1-XY square 4segs $\Omega=$' + str(rabiStrengthMHz) + 'MHz')
    plt.xlabel('Gate T (units of pi/gz)')
    plt.ylabel('Infidelity')
    plt.savefig(gateName + '_' + str(rabiStrengthMHz) + '_curves_noise.png', format='png')
    plt.show()


def saveIFs(gateName, rabiStrengthMHz, gateDataVals, IFdict):
    numGates = len(gateDataVals['G_T_theory'])
    np.savetxt(gateName + '_' + str(rabiStrengthMHz) + 'MHz_expData_theory.csv',
               np.array(
                   [[gateDataVals['G_T_theory'][i], IFdict['G_IF_Theory_TheoryGate'][i]] for i in range(numGates)]),
               delimiter=",")
    np.savetxt(gateName + '_' + str(rabiStrengthMHz) + 'MHz_expData_exp.csv',
               np.array([[gateDataVals['G_T_theory'][i], np.real(IFdict['G_IF_Exp_TheoryGate'][i])] for i in
                         range(numGates)]), delimiter=",")
    np.savetxt(gateName + '_' + str(rabiStrengthMHz) + 'MHz_expData_expVstheory.csv',
               np.array(
                   [[gateDataVals['G_T_theory'][i], np.real(IFdict['G_IF_Exp_Theory'][i])] for i in range(numGates)]),
               delimiter=",")


def IFs_gaussianNoise(basePath, date, gateIndices, gateDataVals, tomos, numNoiseTrials):
    np.random.seed(0)
    N = 500
    G_IF_Exp_Theory_Mean = list()
    G_IF_Exp_Theory_STD = list()
    G_IF_Exp_TheoryGate_Mean = list()
    G_IF_Exp_TheoryGate_STD = list()
    for g_i in gateIndices:
        tomoArray = list()
        for noiseIndex in range(numNoiseTrials):
            tomo_noise = deepcopy(tomos[g_i])
            noiseArray = 1 / np.sqrt(N) * np.array([np.random.normal(0, 1) for _ in range(16 * 16)]).reshape(16, 16)
            tomo_noise._R_mle += noiseArray
            tomoArray.append(tomo_noise)
        IFdict = IFs(basePath, date, range(numNoiseTrials), {'U_vals': [gateDataVals['U_vals'][g_i]] * numNoiseTrials,
                                                             'U_theory': gateDataVals['U_theory']}, tomoArray)

        G_IF_Exp_Theory_Mean.append(np.mean(np.array(IFdict['G_IF_Exp_Theory'])))
        G_IF_Exp_Theory_STD.append(np.std(np.array(IFdict['G_IF_Exp_Theory'])))

        G_IF_Exp_TheoryGate_Mean.append(np.mean(np.array(IFdict['G_IF_Exp_TheoryGate'])))
        G_IF_Exp_TheoryGate_STD.append(np.std(np.array(IFdict['G_IF_Exp_TheoryGate'])))
    return {'G_IF_Exp_Theory_Mean': G_IF_Exp_Theory_Mean,
            'G_IF_Exp_Theory_STD': G_IF_Exp_Theory_STD,
            'G_IF_Exp_TheoryGate_Mean': G_IF_Exp_TheoryGate_Mean,
            'G_IF_Exp_TheoryGate_STD': G_IF_Exp_TheoryGate_STD}


def plotPTMMap(tomo):
    tomo.plot()


def plotPTMMap_theory(gateDataVals):
    """The tomo object we use here can be anything so long as it has the correct dimensions, etc.
    Requiring a tomo object is just an unfortunate byproduct of how QDPM is organized."""

    studyPath_I = Path('.') / 'ExpData' / 'ProcessTomographyTest' / 'id'
    paths_I = Paths(studyPath_I)
    dummyTomo = qdpm.ProcessTomography(qdpm.Experiment(paths_I.resultsPath('2022-03-11_0')),
                                       init_state, preparation, rotation)

    dummyTomo.plot_theory(Qobj(gateDataVals['U_theory'], dims=[[2,2],[2,2]]))