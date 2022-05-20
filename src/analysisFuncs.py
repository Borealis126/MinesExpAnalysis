import sys
sys.path.append("O:\\68707\\JoelHoward\\DataAnalysis")
import qdpm

from qutip import identity, tensor, basis,bell_state,fidelity,Qobj, ket2dm, qeye, qzero
from qutip.qip.operations import rx, ry, cphase
import numpy as np
from itertools import product
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.pyplot as plt


def fidelities(rhos, targetStates):
    return [fidelity(Qobj(rho,[[2,2],[2,2]]),targetStates[rhoIndex])**2 for rhoIndex, rho in enumerate(rhos)]


def tomoAnalysis(dataFolder, numStudies, runName):
    rotation = [identity(2), rx(-np.pi/2), ry(np.pi/2)]
    runFolder=dataFolder / runName
    train_exp = qdpm.Experiment(runFolder)
    c_matrix = qdpm.TwoQubitClassifier(train_exp).c_matrix()
    qst = qdpm.QST_chain(train_exp, rotation, theory_state=tensor(basis(2,0),basis(2,0)), num_of_qubit=2, num_of_QST=numStudies)
    rho = qst.rho_mle
    return rho, qst, c_matrix


def purity(rho):
    return np.real(np.trace(np.array(rho)**2))


def plot_rho(rho, title):
    ticklabel = []
    for idx in product(range(2), repeat=2):
        ticklabel.append('|' + ''.join(map(str, idx)) + r'$\rightangle$')

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.tight_layout(pad=6.0)
    fig.suptitle(title, fontsize=16)
    for i in range(2):
        vals = rho.real
        ax = axs[i]
        plot_title = "Real part"
        if i == 1:
            vals = rho.imag
            plot_title = "Imaginary part"

        img = axs[i].imshow(vals, cmap='seismic')
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(ticklabel, fontsize=12)
        ax.set_yticklabels(ticklabel, fontsize=12)
        ax.set_xticks(np.arange(-0.5, 4), minor=True)
        ax.set_yticks(np.arange(-0.5, 4), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
        ax.set_title(plot_title, fontsize=14)
        img.set_clim(-1, 1)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size='5%', pad=0)
        plt.colorbar(img, ticks=np.linspace(-1, 1, 11), cax=cax)