from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import qdpm
from qutip import identity, Qobj, tensor, basis
from qutip.qip.operations import rx, ry, cphase

def ramsey(path):
    exp_path = Path(path)
    img_path = exp_path.joinpath('ramsey.png')
    ramsey_exp = qdpm.Experiment(path)
    clf = qdpm.SingleQubitClassifier(ramsey_exp, qubit=1)
    prediction = clf.predict()
    population = prediction.mean(axis=1)
    tau = np.linspace(0, 50, ramsey_exp.scan_size[0])
    fig, ax = plt.subplots()
    ramsey_fit = qdpm.RamseyFit(tau, population, two_freqs=False, make_plots=True, ax=ax)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    fig.savefig(img_path)
    return [str(img_path), str(ramsey_fit.fit_params['tau'])]
#######################################################

# e = qdpm.Experiment(r'O:\68707\BF2222\LABVIEWdata\Cooldown20200221_Nb_SWIPHT_MB_W8_180nm\Heterodyne\QST\Free evolution CZ_Pulse Train\YPiO2 X YPiO2\2020_03_01_23_44_29')
# rotation = [identity(2),
#             rx(-np.pi/2),
#             ry(np.pi/2)]
# initial_state = tensor((basis(2, 0) + basis(2, 1)).unit(), (basis(2, 0) + basis(2, 1)).unit())
# target = cphase(np.pi) * initial_state
# tomo = qdpm.StateTomography(e, rotation, theory_state=target)
# fig, ax = plt.subplots()
# print(tomo.fidelity)
# tomo.plot(part='real', ax=ax)
# plt.show()

########################################################
def pt(path):
    e = qdpm.Experiment(path)
    preparation = [identity(2),
                   rx(np.pi),
                   ry(np.pi/2),
                   ry(-np.pi/2),
                   rx(-np.pi/2),
                   rx(np.pi/2)]
    rotation = [ry(np.pi/2),
                rx(-np.pi/2),
                identity(2)]
    init_state = [basis(2, 0), basis(2, 0)]
    theory_gate = cphase(np.pi)
    tomo = qdpm.ProcessTomography(e, init_state, preparation, rotation, theory_gate)
    print(tomo.fidelity)
    tomo.plot_theory()
    plt.show()
    return 'done'

pt(r'O:\68707\BF2222\LABVIEWdata\Cooldown20201021\Heterodyne\Nb_SWIPHT_MB_W8_180nm\QPT\2020_12_01_21_28_44')

########################################################
# ramsey_exp = qdpm.Experiment(r'O:\68707\BF2222\LABVIEWdata\Cooldown20200714\Heteordyne\PTC\Ramsey\Q2\2020_07_18_15_37_02')
# qubit_idx = 1
# clf = qdpm.ClassifierOneQubit(ramsey_exp, qubit=qubit_idx)
# a, b = clf._set_assembly()
# print(a.shape)