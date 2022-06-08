from itertools import product
from matplotlib.pyplot import pause
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip import sigmax, sigmay, sigmaz, identity, tensor, basis, operator_to_vector, ket2dm, Qobj, fidelity, average_gate_fidelity
import cvxpy as cp
import joblib
from ..analysis.classifier import TwoQubitClassifier

from ..experiment import Experiment
from forest.benchmarking.operator_tools import pauli_liouville2chi, kraus2chi, chi2pauli_liouville

class StateTomography(object):

    def __init__(self, exp, rotation, theory_state=None, num_of_qubit=2):
        if not isinstance(exp, Experiment):
            raise TypeError('Input is not a qdpm.Experiment object')

        self._exp = exp
        self._rotation = rotation
        self._rot_per_qubit = len(rotation)
        self._num_of_qubit = int(num_of_qubit)
        self._com_base = 2 ** num_of_qubit
        self._total_rot = self._rot_per_qubit ** num_of_qubit

        try:
            if theory_state.type == 'ket':
                self._theory_dm = ket2dm(theory_state)
            else:
                self._theory_dm = theory_state
        except:
            self._theory_dm = None
        
        self._prob, self._rho = self._do_optimization()

    @property
    def rho_mle(self):
        return self._rho.value

    @property
    def purity(self):
        rho = self._rho

        return np.matmul(rho.value, rho.value).trace().real

    @property
    def fidelity(self):
        if self._theory_dm is None:
            raise ValueError('Need a theory state to calculate fidelity.')

        rho = Qobj(self.rho_mle, dims=[[2, 2], [2, 2]])
        sigma = self._theory_dm

        return fidelity(rho, sigma)

    def _state_projection(self):
        clf = TwoQubitClassifier(self._exp, (0, 1))

        state_projection = clf.predict().T.reshape((-1, self._total_rot))

        confusion_matrix = clf.c_matrix()

        return state_projection, confusion_matrix

    def _pauli_2Q_assembly(self):
        pauli_1Q_rescaled = [identity(2)/np.sqrt(2),
                             sigmax()/np.sqrt(2),
                             sigmay()/np.sqrt(2),
                             sigmaz()/np.sqrt(2)]

        pauli_2Q_rescaled = []
        for i, j in product(range(self._com_base), repeat=2):
            pauli_2Q_rescaled.append(tensor(pauli_1Q_rescaled[i], pauli_1Q_rescaled[j]))

        return pauli_2Q_rescaled

    def _2Q_rotation_assembly(self):
        rotation_1Q = self._rotation
        rotation_2Q = []
        for i, j in product(range(self._rot_per_qubit), repeat=2):
            rotation_2Q.append(tensor(rotation_1Q[i], rotation_1Q[j]))

        return rotation_2Q

    def _C_matrix_assembly(self):
        _, confusion_matrix = self._state_projection()
        pauli_2Q_rescaled = self._pauli_2Q_assembly()
        rotation_2Q = self._2Q_rotation_assembly()

        bitstring_projector = []
        for i, j in product(range(2), repeat=2):
            bitstring_projector.append(tensor(basis(2, i), basis(2, j)))

        N_projector = []
        for j in range(self._com_base):
            N_projector_int = 0
            for k in range(self._com_base):
                N_projector_int += confusion_matrix[j, k] * bitstring_projector[k].proj()
            N_projector.append(N_projector_int)

        pi_projector = np.zeros((self._com_base, self._com_base ** self._num_of_qubit))
        for j, l in product(range(self._com_base), range(self._com_base ** self._num_of_qubit)):
            pi_projector[j, l] = (N_projector[j]*pauli_2Q_rescaled[l]).tr()


        R_projector = [np.zeros((self._com_base ** self._num_of_qubit, self._com_base ** self._num_of_qubit)) for _ in range(self._total_rot)]
        for k in range(self._total_rot):
            for r, m in product(range(self._com_base ** self._num_of_qubit), repeat=2):
                Lambda = rotation_2Q[k] * pauli_2Q_rescaled[m] * rotation_2Q[k].dag()
                Lambda_tilde = operator_to_vector(Lambda)
                R_projector[k][r, m] = (operator_to_vector(pauli_2Q_rescaled[r]).dag() * Lambda_tilde)[0, 0].real

        C_matrix = np.zeros((self._com_base, self._total_rot, self._com_base ** self._num_of_qubit))
        for j, k, m in product(range(self._com_base), range(self._total_rot), range(self._com_base ** self._num_of_qubit)):
            for r in range(self._com_base ** self._num_of_qubit):
                C_matrix[j, k, m] += pi_projector[j, r] * R_projector[k][r, m]

        return C_matrix

    def _histogram(self):
        state_projection, _ = self._state_projection()
        histogram = np.zeros((self._com_base, self._total_rot))
        for j, k in product(range(self._com_base), range(self._total_rot)):
            histogram[j, k] = sum(state_projection[:, k] == j)

        return histogram

    def _do_optimization(self):
        if self._exp.path.joinpath('assets/rho_mle.joblib').exists():
            prob, rho = joblib.load(self._exp.path.joinpath('assets/rho_mle.joblib'))
        else:
            pauli_2Q_rescaled = self._pauli_2Q_assembly()
            C_matrix = self._C_matrix_assembly()
            histogram = self._histogram()

            rho_m = cp.Variable((self._com_base) ** 2)

            likelihood = 0
            for j, k in product(range(self._com_base), range(self._total_rot)):
                log_term = 0
                for m in range(self._com_base ** 2):
                    log_term += C_matrix[j, k, m] * rho_m[m]
                likelihood += histogram[j, k] * cp.log(log_term)

            obj = cp.Maximize(likelihood)

            rho = 0
            for m in range(self._com_base ** 2):
                rho += rho_m[m] * pauli_2Q_rescaled[m].full()

            constraints = [cp.trace(rho) == 1,
                        rho >> 0]

            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.SCS)
            joblib.dump([prob, rho], self._exp.path.joinpath('assets/rho_mle.joblib'))
        return prob, rho

    def plot(self, part, ax=None, colorbar=True, dressed_state=False):
        if part == 'real':
            data = self.rho_mle.real
            plot_title = r'Real part of MLE density matrix $\rho$'
        elif part == 'imag':
            data = self.rho_mle.imag
            plot_title = r'Imaginary part of MLE density matrix $\rho$'

        ticklabel = []
        for idx in product(range(2), repeat=self._num_of_qubit):
            ticklabel.append('|' + ''.join(map(str, idx)) + r'$\rightangle$')

        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        fig = plt.gcf()
        img = ax.imshow(data, cmap='seismic')
        ax.set_xticks(range(self._com_base))
        ax.set_yticks(range(self._com_base))
        ax.set_xticklabels(ticklabel, fontsize=14)
        ax.set_yticklabels(ticklabel, fontsize=14)
        ax.set_xticks(np.arange(-0.5, self._com_base), minor=True)
        ax.set_yticks(np.arange(-0.5, self._com_base), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
        ax.set_title(plot_title, fontsize=14)
        if colorbar:
            fig.colorbar(img, ticks=np.linspace(-1, 1, 11))
        img.set_clim(-1, 1)

        return fig, ax, img

    def plot_3D(self, part, limits=None, ax=None):
        if part == 'real':
            data = self.rho_mle.real
            plot_title = r'Real part of MLE density matrix $\rho$'
        elif part == 'imag':
            data = self.rho_mle.imag
            plot_title = r'Imaginary part of MLE density matrix $\rho$'

        ticklabel = []
        for idx in product(range(2), repeat=self._num_of_qubit):
            ticklabel.append('|' + ''.join(map(str, idx)) + r'$\rightangle$')

        n = np.size(data)
        xpos, ypos = np.meshgrid(range(self._com_base), range(self._com_base))
        xpos = xpos.T.flatten() - 0.5
        ypos = ypos.T.flatten() - 0.5
        zpos = np.zeros(n)
        dx = dy = 0.8 * np.ones(n)
        dz = data.flatten()

        if limits and type(limits) is list and len(limits) == 2:
            z_min = limits[0]
            z_max = limits[1]
        else:
            z_min = min(dz)
            z_max = max(dz)
            if z_min == z_max:
                z_min -= 0.1
                z_max += 0.1

        norm = mpl.colors.Normalize(z_min, z_max)
        cmap = mpl.cm.get_cmap('seismic')
        colors = cmap(norm(dz))

        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig, azim=-40, elev=30)

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, alpha=0.8, color=colors)
        ax.set_zlim3d(min(z_min, 0), z_max)
        ax.set_xticks(range(self._com_base))
        ax.set_yticks(range(self._com_base))
        ax.set_xticklabels(ticklabel, fontsize=14)
        ax.set_yticklabels(ticklabel, fontsize=14)
        ax.set_title(plot_title, fontsize=14)

    def plot_theory(self, part, ax=None):
        if self._theory_dm is None:
            raise ValueError('Need a theory state to plot theory density matrix.')

        if part == 'real':
            data = self._theory_dm.full().real
            plot_title = r'Real part of theory density matrix $\rho$'
        elif part == 'imag':
            data = self._theory_dm.full().imag
            plot_title = r'Imaginary part of theory density matrix $\rho$'

        ticklabel = []
        for idx in product(range(2), repeat=self._num_of_qubit):
            ticklabel.append('|' + ''.join(map(str, idx)) + r'$\rightangle$')

        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        fig = plt.gcf()
        img = ax.imshow(data, cmap='seismic')
        ax.set_xticks(range(self._com_base))
        ax.set_yticks(range(self._com_base))
        ax.set_xticklabels(ticklabel, fontsize=14)
        ax.set_yticklabels(ticklabel, fontsize=14)
        ax.set_xticks(np.arange(-0.5, self._com_base), minor=True)
        ax.set_yticks(np.arange(-0.5, self._com_base), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
        ax.set_title(plot_title, fontsize=14)
        fig.colorbar(img, ticks=np.linspace(-1, 1, 11))
        img.set_clim(-1, 1)

        return fig, ax, img 

class ProcessTomography(object):

    def __init__(self, exp, init_state, preparation, rotation, qubit=(0, 1)):
        if not isinstance(exp, Experiment):
            raise TypeError('Input is not a qdpm.Experiment object')

        self._exp = exp
        self._init_state = init_state
        self._preparation = preparation
        self._prep_per_qubit = len(preparation)
        self._rotation = rotation
        self._rot_per_qubit = len(rotation)
        self._qubit = qubit
        self._num_of_qubit = len(qubit)
        self._com_base = 2 ** self._num_of_qubit
        self._total_rot = self._rot_per_qubit ** self._num_of_qubit
        self._total_prep = self._prep_per_qubit ** self._num_of_qubit
        self._2Q_init_state = tensor(init_state[0], init_state[1])
        self._init_den_mat = ket2dm(self._2Q_init_state)
        self._R_mle = self._do_optimization()

    @property
    def R_mle(self):
        return self._R_mle

    def _state_projection(self):
        clf = TwoQubitClassifier(self._exp, self._qubit, calib_seq=(0, 1, 2, 3))

        state_projection = clf.predict().T.reshape(
            (-1, self._total_rot, self._total_prep), order='F')

        confusion_matrix = clf.c_matrix()

        return state_projection, confusion_matrix

    def _pauli_2Q_assembly(self):
        pauli_1Q_rescaled = [identity(2)/np.sqrt(2),
                             sigmax()/np.sqrt(2),
                             sigmay()/np.sqrt(2),
                             sigmaz()/np.sqrt(2)]

        pauli_2Q_rescaled = []
        for i, j in product(range(self._com_base), repeat=2):
            pauli_2Q_rescaled.append(tensor(pauli_1Q_rescaled[i], pauli_1Q_rescaled[j]))

        return pauli_2Q_rescaled

    def _2Q_rotation_assembly(self):
        rotation_1Q = self._rotation
        rotation_2Q = []
        for i, j in product(range(self._rot_per_qubit), repeat=2):
            rotation_2Q.append(tensor(rotation_1Q[i], rotation_1Q[j]))

        return rotation_2Q

    def _2Q_preparation_assembly(self):
        preparation_1Q = self._preparation
        preparation_2Q = []
        for i, j in product(range(self._prep_per_qubit), repeat=2):
            preparation_2Q.append(tensor(preparation_1Q[i], preparation_1Q[j]))

        return preparation_2Q

    def _B_matrix_assembly(self, confusion_matrix):
        if self._exp.path.joinpath('assets/B_matrix.joblib').exists():
            B_matrix = joblib.load(self._exp.path.joinpath('assets/B_matrix.joblib'))
        else:
            # _, confusion_matrix = self._state_projection()
            pauli_2Q_rescaled = self._pauli_2Q_assembly()
            rotation_2Q = self._2Q_rotation_assembly()
            preparation_2Q = self._2Q_preparation_assembly()
            bitstring_projector = []
            for i, j in product(range(2), repeat=2):
                bitstring_projector.append(tensor(basis(2, i), basis(2, j)))
            N_projector = []
            for j in range(self._com_base):
                N_projector_int = 0
                for k in range(self._com_base):
                    N_projector_int += confusion_matrix[j, k] * bitstring_projector[k].proj()
                N_projector.append(N_projector_int)
            pi_projector = np.zeros((self._com_base, self._com_base ** self._num_of_qubit))
            for j, l in product(range(self._com_base), range(self._com_base ** self._num_of_qubit)):
                pi_projector[j, l] = (N_projector[j]*pauli_2Q_rescaled[l]).tr()
            R_projector_rot = [np.zeros((self._com_base ** self._num_of_qubit, self._com_base ** self._num_of_qubit)) for _ in range(self._total_rot)]
            for k in range(self._total_rot):
                for r, m in product(range(self._com_base ** self._num_of_qubit), repeat=2):
                    R_projector_rot[k][r, m] = (pauli_2Q_rescaled[r] * rotation_2Q[k] * pauli_2Q_rescaled[m] * rotation_2Q[k].dag()).tr().real
            R_projector_prep = [np.zeros((self._com_base ** self._num_of_qubit, self._com_base ** self._num_of_qubit)) for _ in range(self._total_prep)]
            for l in range(self._total_prep):
                for n, q in product(range(self._com_base ** self._num_of_qubit), repeat=2):
                    R_projector_prep[l][n, q] = (pauli_2Q_rescaled[n] * preparation_2Q[l] * pauli_2Q_rescaled[q] * preparation_2Q[l].dag()).tr().real
            rho_0 = []
            for q in range(self._com_base ** self._num_of_qubit):
                rho_0.append((pauli_2Q_rescaled[q] * self._init_den_mat).tr())

            B_matrix = np.zeros((self._com_base, self._total_rot, self._total_prep,
                                self._com_base ** self._num_of_qubit, self._com_base ** self._num_of_qubit))
            for j, k, l, m, n in product(range(self._com_base), range(self._total_rot), range(self._total_prep),
                                        range(self._com_base ** self._num_of_qubit), range(self._com_base ** self._num_of_qubit)):
                B_matrix[j, k, l, m, n] = sum([pi_projector[j, r] * R_projector_rot[k][r, m] * R_projector_prep[l][n, q] * rho_0[q]
                                               for r, q in product(range(self._com_base ** self._num_of_qubit), repeat=2)])

            joblib.dump(B_matrix, self._exp.path.joinpath('assets/B_matrix.joblib'))

        return B_matrix

    def _histogram(self, state_projection):
        # state_projection, _ = self._state_projection()

        histogram = np.zeros((self._com_base, self._total_rot, self._total_prep))
        for j, k, l in product(range(self._com_base), range(self._total_rot), range(self._total_prep)):
            histogram[j, k, l] = sum(state_projection[:, k, l] == j)

        return histogram

    def _do_optimization(self):
        if self._exp.path.joinpath('assets/r_mle.joblib').exists():
            r_mle = joblib.load(self._exp.path.joinpath('assets/r_mle.joblib'))
        else:
            pauli_2Q_rescaled = self._pauli_2Q_assembly()

            state_projection, confusion_matrix = self._state_projection()

            B_matrix = self._B_matrix_assembly(confusion_matrix)

            histogram = self._histogram(state_projection)

            r = cp.Variable(((self._com_base) ** 2, (self._com_base) ** 2))

            # likelihood = 0
            # for j, k, l in product(range(self._com_base), range(self._total_rot), range(self._total_prep)):
            #     log_term = sum([B_matrix[j, k, l, m, n] * r[m, n] for m, n in product(range(self._com_base ** 2), repeat=2)])
            #     # log_term = 0
            #     # for m, n in product(range(self._com_base ** 2), repeat=2):
            #     #     log_term += B_matrix[j, k, l, m, n] * r[m, n]
            #     likelihood += histogram[j, k, l] * cp.log(log_term)

            likelihood = sum([histogram[j, k, l]
                              * cp.log(sum([B_matrix[j, k, l, m, n] * r[m, n]
                                            for m, n in product(range(self._com_base ** 2), repeat=2)]))
                              for j, k, l in product(range(self._com_base), range(self._total_rot), range(self._total_prep))])

            obj = cp.Maximize(likelihood)

            choi_matrix = 0
            for i, j in product(range((self._com_base) ** 2), repeat=2):
                choi_matrix += r[i, j] * tensor(pauli_2Q_rescaled[j].trans(), pauli_2Q_rescaled[i]).full()

            constraints = [r[0, 0] == 1,
                           r[0, 1:] == 0,
                        choi_matrix >> 0]

            prob = cp.Problem(obj, constraints)

            prob.solve()

            r_mle = r.value

            joblib.dump(r_mle, self._exp.path.joinpath('assets/r_mle.joblib'))

        return r_mle

    def R_theory(self, theory_gate):
        pauli_2Q_rescaled = self._pauli_2Q_assembly()

        r_theory = np.zeros(((self._com_base) ** 2, (self._com_base) ** 2))
        for m, n in product(range((self._com_base) ** 2), repeat=2):
            r_theory[m, n] = (pauli_2Q_rescaled[m] * (theory_gate * pauli_2Q_rescaled[n] * theory_gate.dag())).tr().real

        return r_theory

    def fidelity(self, theory_gate):
        fidelity = (np.matmul(self.R_mle.T, self.R_theory(theory_gate)).trace()/self._com_base + 1)/(self._com_base + 1)
        return fidelity

    @property
    def purity(self):
        
        purity = (np.matmul(self.R_mle.T, self.R_mle).trace()/self._com_base + 1)/(self._com_base + 1)

        return purity

    def plot(self, ax=None):
        ticklabel = []
        for idx in product('IXYZ', repeat=self._num_of_qubit):
            ticklabel.append(''.join(map(str, idx)))

        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        data = self.R_mle
        fig = plt.gcf()
        img = ax.imshow(data, cmap='seismic')
        ax.set_xticks(range(16))
        ax.set_yticks(range(16))
        ax.set_xticklabels(ticklabel, fontsize=10)
        ax.set_yticklabels(ticklabel, fontsize=10)
        ax.set_xlabel('Input Pauli Operator')
        ax.set_ylabel('Output Pauli Operator')
        ax.grid(color='k', alpha=0.2, linestyle='-', linewidth=1)
        # ax.set_title('QPT', fontsize=14)
        # fig.colorbar(img, ticks=np.linspace(-1, 1, 11))
        img.set_clim(-1, 1)
        return fig, ax, img

    def plot_theory(self, theory_gate, ax=None):
        ticklabel = []
        for idx in product('IXYZ', repeat=self._num_of_qubit):
            ticklabel.append(''.join(map(str, idx)))

        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        data = self.R_theory(theory_gate)
        fig = plt.gcf()
        img = ax.imshow(data, cmap='seismic')
        ax.set_xticks(range(16))
        ax.set_yticks(range(16))
        ax.set_xticklabels(ticklabel, fontsize=10)
        ax.set_yticklabels(ticklabel, fontsize=10)
        ax.set_xlabel('Input Pauli Operator')
        ax.set_ylabel('Output Pauli Operator')
        ax.grid(color='k', alpha=0.2, linestyle='-', linewidth=1)
        # ax.set_title('QPT', fontsize=14)
        # fig.colorbar(img, ticks=np.linspace(-1, 1, 11))
        img.set_clim(-1, 1)
        return fig, ax, img

def R_SPAMfree(U, R_U, R_I):
    chi_exp_U = pauli_liouville2chi(R_U)
    chi_exp_I = pauli_liouville2chi(R_I)

    pauli_1Q = [i.full() for i in [identity(2), sigmax(), sigmay(), sigmaz()]]
    pauli_2Q = [np.kron(i[0], i[1]) for i in product(pauli_1Q, pauli_1Q)]

    def E(m):
        return pauli_2Q[m]

    T = np.zeros([16, 16], dtype=complex)
    for m in range(16):
        for n in range(16):
            T[m, n] = np.trace(np.matmul(np.matmul(E(m).conj().T, E(n)), U.conj().T)) / 4

    chi_err_exp_U = np.matmul(np.matmul(T, chi_exp_U), T.conj().T)

    V = np.zeros([16, 16], dtype=complex)
    for m in range(16):
        for n in range(16):
            V[m, n] = np.trace(np.matmul(np.matmul(E(m).conj().T, E(n)), np.eye(4).conj().T)) / 4

    chi_err_exp_I = np.matmul(np.matmul(V, chi_exp_I), V.conj().T)

    chi_I = kraus2chi(np.kron(np.eye(2), np.eye(2)))

    chi_err = chi_err_exp_U - (chi_err_exp_I - chi_I)

    chi = np.matmul(np.matmul(np.linalg.inv(T), chi_err), np.linalg.inv(T.conj().T))
    return chi2pauli_liouville(chi)


class QST_chain(StateTomography):
    def __init__(self, exp, rotation, theory_state, num_of_qubit, num_of_QST):
        self._num_of_QST = num_of_QST
        self._calib_seq = (0, 1, 2, 3)

        super().__init__(exp, rotation, theory_state=theory_state, num_of_qubit=num_of_qubit)

    def _state_projection(self):
        clf = TwoQubitClassifier(self._exp, (0, 1), calib_seq=self._calib_seq)

        state_projection = clf.predict().T.reshape((-1, self._total_rot*self._num_of_QST))

        confusion_matrix = clf.c_matrix()

        return state_projection, confusion_matrix

    def _histogram(self):
        state_projection, _ = self._state_projection()
        histogram = np.zeros((self._com_base, self._total_rot*self._num_of_QST))
        for j, k in product(range(self._com_base), range(self._total_rot*self._num_of_QST)):
            histogram[j, k] = sum(state_projection[:, k] == j)

        return histogram

    def _do_optimization(self):
        if self._exp.path.joinpath('assets/rho_mle.joblib').exists():
            prob_all, rho_all = joblib.load(self._exp.path.joinpath('assets/rho_mle.joblib'))
        else:
            pauli_2Q_rescaled = self._pauli_2Q_assembly()
            C_matrix = self._C_matrix_assembly()
            prob_all, rho_all = [], []

            h = self._histogram()
            for piece in range(self._num_of_QST):
                histogram = h[:, piece*self._total_rot: (piece+1)*self._total_rot]
                rho_m = cp.Variable((self._com_base) ** 2)

                likelihood = 0
                for j, k in product(range(self._com_base), range(self._total_rot)):
                    log_term = 0
                    for m in range(self._com_base ** 2):
                        log_term += C_matrix[j, k, m] * rho_m[m]
                    likelihood += histogram[j, k] * cp.log(log_term)

                obj = cp.Maximize(likelihood)

                rho = 0
                for m in range(self._com_base ** 2):
                    rho += rho_m[m] * pauli_2Q_rescaled[m].full()

                constraints = [cp.trace(rho) == 1,
                            rho >> 0]

                prob = cp.Problem(obj, constraints)
                prob.solve(solver=cp.SCS)

                prob_all.append(prob)
                rho_all.append(rho)
            joblib.dump([prob_all, rho_all], self._exp.path.joinpath('assets/rho_mle.joblib'))
        return prob_all, rho_all

    @property
    def rho_mle(self):
        return [a.value for a in self._rho]

    @property
    def purity(self):
        rho = self._rho

        return [np.matmul(a.value, a.value).trace().real for a in rho]

    @property
    def fidelity(self):
        if self._theory_dm is None:
            raise ValueError('Need a theory state to calculate fidelity.')
        
        sigma = self._theory_dm
        fidelity_all = []
        for i in range(self._num_of_QST):
            rho = Qobj(self.rho_mle[i], dims=[[2, 2], [2, 2]])
            fidelity_all.append(fidelity(rho, sigma)  )

        return fidelity_all

    def plot(self, part, QST_idx, ax=None, colorbar=False):
        if part == 'real':
            data = self.rho_mle[QST_idx].real
            plot_title = r'Real part of MLE density matrix $\rho$'
        elif part == 'imag':
            data = self.rho_mle[QST_idx].imag
            plot_title = r'Imaginary part of MLE density matrix $\rho$'

        ticklabel = []
        for idx in product(range(2), repeat=self._num_of_qubit):
            ticklabel.append('|' + ''.join(map(str, idx)) + r'$\rightangle$')

        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        fig = plt.gcf()
        img = ax.imshow(data, cmap='seismic')
        ax.set_xticks(range(self._com_base))
        ax.set_yticks(range(self._com_base))
        ax.set_xticklabels(ticklabel, fontsize=14)
        ax.set_yticklabels(ticklabel, fontsize=14)
        ax.set_xticks(np.arange(-0.5, self._com_base), minor=True)
        ax.set_yticks(np.arange(-0.5, self._com_base), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
        ax.set_title(plot_title, fontsize=14)
        if colorbar:
            fig.colorbar(img, ticks=np.linspace(-1, 1, 11))
        img.set_clim(-1, 1)

    def plot_3D(self, part, QST_idx, limits=None, ax=None):
        if part == 'real':
            data = self.rho_mle[QST_idx].real
            plot_title = r'Real part of MLE density matrix $\rho$'
        elif part == 'imag':
            data = self.rho_mle[QST_idx].imag
            plot_title = r'Imaginary part of MLE density matrix $\rho$'

        ticklabel = []
        for idx in product(range(2), repeat=self._num_of_qubit):
            ticklabel.append('|' + ''.join(map(str, idx)) + r'$\rightangle$')

        n = np.size(data)
        xpos, ypos = np.meshgrid(range(self._com_base), range(self._com_base))
        xpos = xpos.T.flatten() - 0.5
        ypos = ypos.T.flatten() - 0.5
        zpos = np.zeros(n)
        dx = dy = 0.8 * np.ones(n)
        dz = data.flatten()

        if limits and type(limits) is list and len(limits) == 2:
            z_min = limits[0]
            z_max = limits[1]
        else:
            z_min = min(dz)
            z_max = max(dz)
            if z_min == z_max:
                z_min -= 0.1
                z_max += 0.1

        norm = mpl.colors.Normalize(z_min, z_max)
        cmap = mpl.cm.get_cmap('seismic')
        colors = cmap(norm(dz))

        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig, azim=-40, elev=30)

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, alpha=0.8, color=colors)
        ax.set_zlim3d(min(z_min, 0), z_max)
        ax.set_xticks(range(self._com_base))
        ax.set_yticks(range(self._com_base))
        ax.set_xticklabels(ticklabel, fontsize=14)
        ax.set_yticklabels(ticklabel, fontsize=14)
        ax.set_title(plot_title, fontsize=14)

    def plot_theory(self, part, ax=None, colorbar=False):
        if self._theory_dm is None:
            raise ValueError('Need a theory state to plot theory density matrix.')

        if part == 'real':
            data = self._theory_dm.full().real
            plot_title = r'Real part of theory density matrix $\rho$'
        elif part == 'imag':
            data = self._theory_dm.full().imag
            plot_title = r'Imaginary part of theory density matrix $\rho$'

        ticklabel = []
        for idx in product(range(2), repeat=self._num_of_qubit):
            ticklabel.append('|' + ''.join(map(str, idx)) + r'$\rightangle$')

        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        fig = plt.gcf()
        img = ax.imshow(data, cmap='seismic')
        ax.set_xticks(range(self._com_base))
        ax.set_yticks(range(self._com_base))
        ax.set_xticklabels(ticklabel, fontsize=14)
        ax.set_yticklabels(ticklabel, fontsize=14)
        ax.set_xticks(np.arange(-0.5, self._com_base), minor=True)
        ax.set_yticks(np.arange(-0.5, self._com_base), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
        ax.set_title(plot_title, fontsize=14)
        if colorbar:
            fig.colorbar(img, ticks=np.linspace(-1, 1, 11))
        img.set_clim(-1, 1)   

