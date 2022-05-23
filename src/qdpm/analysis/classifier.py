from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn import svm
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from ..experiment import Experiment

# TODO: Modify TwoQubitClassifier to be a child class of QubitClassifier

class QubitClassifier(object):
    """A generic class to classify qubit state.

    Parameters
    ----------
    exp : Experiment
        the Experiment object to do classification with
    qubit : int or tuple of int
        qubit index or tuple of qubit indices to be classified
    calib_seq : tuple, optional
        the index of the calibration pulses, in the order of
        (|00>, |01>, |10>, |11>), by default None if no
        calibration pulse is included
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, optional
            SVM kernel used, by default 'rbf'

    Raises
    ------
    TypeError
        if 'exp' is not an Experiment object
    TypeError
        if the experiment is not a single-shot experiment
    ValueError
        the qubit to be classified has out-of-range index
        
    """
    def __init__(self, exp, qubit, calib_seq=None, kernel='rbf'):
        if not isinstance(exp, Experiment):
            raise TypeError('Input is not a Experiment object')

        if exp.scan_setup.loc['Target', 0] != 'Sequencer':
            raise TypeError(
                'Input experiment is not a single-shot experiment')

        # if not set(qubit).issubset(range(exp.number_of_readout)):
        #     raise ValueError('Qubit index out of range')

        self._exp = exp
        self._qubit = qubit
        self._calib_seq = calib_seq
        self._SVM_kernel = kernel

        if 'Sequence' in exp.scan_setup.loc['Target'].values:
            self._data = self._seq_concatenate()

            self._calib_seq = []
            for seq in range(self._seq_piece):
                if calib_seq is not None:
                    self._calib_seq += list(map(lambda x, seq=seq: x + seq * self._exp.scan_size[0],
                                                calib_seq))
        else:
            self._data = exp.data

    def _set_assembly(self):
        """Raise error if not implemented in child class."""
        raise NotImplementedError("Not implemented!")

    def _train_set_assembly(self):
        """Raise error if not implemented in child class."""
        raise NotImplementedError("Not implemented!")

    def _seq_concatenate(self):
        """Concatenate 'Sequence' experiments.

        Returns
        -------
        list of list of numpy.ndarray
            concatenated data list
        """
        # find the index of 'Sequence' sweep
        seq_idx = self._exp.scan_setup.isin(['Sequence']).any().to_numpy().nonzero()[0][0]

        # determines number of sequence piece
        self._seq_piece = self._exp.scan_size[seq_idx]

        # concatenate sequence
        if isinstance(self._qubit, tuple):
            data = [[None for _ in range(2)] for _ in range(2)]
            for res, quad in product(range(2), range(2)):
                data_temp = self._exp.data[self._qubit[res]][quad].transpose((2, 0, 1))
                data[res][quad] = data_temp.reshape(-1, self._exp.scan_size[1])
        elif isinstance(self._qubit, int):
            data = [None for _ in range(2)]
            for quad in range(2):
                data_temp = self._exp.data[self._qubit][quad].transpose((2, 0, 1))
                data[quad] = data_temp.reshape(-1, self._exp.scan_size[1])            

        return data
    
    def _classify_boundary(self):
        """Carry out classification.

        Returns
        -------
        sklearn.estimator
            best estimator determined by sklearn.GridSearchCV
        """
        if self._exp.path.joinpath(f'assets/Projector_Q{self._qubit}.joblib').exists():
            clf = joblib.load(self._exp.path.joinpath(f'assets/Projector_Q{self._qubit}.joblib'))
        else:
            whole_set_scaled, label = self._train_set_assembly()

            # prepare search grid
            C_range = np.logspace(-1.1, 3.1, num=5, base=2)
            gamma_range = np.logspace(-4.1, 1.1, num=6, base=2)
            param_grid = dict(gamma=gamma_range, C=C_range)

            # stratified shuffle for cross-validation
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)

            # do SVM classification 
            grid = GridSearchCV((svm.SVC(kernel=self._SVM_kernel)),
                                param_grid=param_grid,
                                cv=cv)
            grid.fit(whole_set_scaled, label)

            clf = grid.best_estimator_

            joblib.dump(clf, self._exp.path.joinpath(f'assets/Projector_Q{self._qubit}.joblib'))

        return clf

    def c_matrix(self):
        """Calculate classification confusion matrix.

        Returns
        -------
        numpy.ndarray of shape (n_classes, n_classes)
            Confusion matrix whose i-th row and j-th column entry indicates
            the number of samples with true label being i-th class and
            predicted label being j-th class.
        """
        clf = self._classify_boundary()
        whole_set_scaled, label = self._train_set_assembly()

        predicted_label = clf.predict(whole_set_scaled)

        c_matrix = confusion_matrix(label,
                                    predicted_label, normalize='true').T

        return c_matrix

class SingleQubitClassifier(QubitClassifier):
    """Classify single qubit state."""
    def __init__(self, exp, qubit, calib_seq=None):
        super().__init__(exp, qubit, calib_seq)

        self._clf = self._classify_boundary()

    def _set_assembly(self):
        """Assemble training data set.

        This method is to put together the training set for supervised
        machine learning. If the experiment includes calibration pulses,
        those will be used directly. Otherwise, the program will infer the
        training set to its best by simple analysis of the data.

        Returns
        -------
        whole_set : numpy.ndarray
            the training set for SVM to use
        set_length : list of int
            the length of two training subsets
        """
        raw = self._data[self._qubit]
        mean_data = np.mean(self._data, axis=self._exp.average_axis)[self._qubit]

        # put together a rough training set when no calibration experiments
        # are present
        if not self._calib_seq:
            # find a rough center of all points
            x_center = np.mean(mean_data[0])
            poly_fit = np.polynomial.Polynomial.fit(mean_data[0], mean_data[1], 1)
            y_center = poly_fit(x_center)

            # rotate data on IQ plane so that span along I quadrature is maximum
            theta = np.pi - np.angle((mean_data[0][0]-x_center)+1j*(poly_fit(mean_data[0][0])-y_center))
            x_rotated = np.cos(theta)*(raw[0]-x_center) - np.sin(theta)*(raw[1]-y_center)

            # find two experiments with roughly most and least ground state
            ground_ratio = np.sum(x_rotated < 0, axis=1)/x_rotated.shape[1]
            idx1 = np.argmax(ground_ratio)
            idx2 = np.argmin(ground_ratio)

            # put together two training sets
            ground_state_hist = np.stack((raw[0][idx1, :],
                                          raw[1][idx1, :])).T
            excited_state_hist = np.stack((raw[0][idx2, :],
                                           raw[1][idx2, :])).T
        else:
            ground_state_hist = np.stack((raw[0][self._calib_seq[0], :],
                                          raw[1][self._calib_seq[0], :])).T
            excited_state_hist = np.stack((raw[0][self._calib_seq[1], :],
                                           raw[1][self._calib_seq[1], :])).T

        # combine the two sets
        whole_set = np.concatenate((ground_state_hist, excited_state_hist), axis=0)
        set_length = [ground_state_hist.shape[0], excited_state_hist.shape[0]]

        return whole_set, set_length

    def _train_set_assembly(self):
        """Normalize train set and make labels.

        Returns
        -------
        whole_set_scaled : numpy.ndarray
            normalized training set, rangeing (-2, 2)
        label : numpy.ndarray
            training set label, by default [0, ..., 0, 1, ..., 1]
        """
        whole_set, set_length = self._set_assembly()
        whole_set_scaled = minmax_scale(whole_set,
                                        feature_range=(-2, 2), axis=0)

        label = np.concatenate((np.full(set_length[0], 0, dtype=int),
                                np.full(set_length[1], 1, dtype=int)))

        return whole_set_scaled, label

    def plot_boundary(self):
        """Plot the training set and classification boundary.
        """
        _, set_length = self._set_assembly()
        whole_set_scaled, _ = self._train_set_assembly()

        fidelity = self.fidelity()

        fig, ax = plt.subplots(figsize=(9, 9), dpi=100)
        ax.scatter(whole_set_scaled[:set_length[0], 0],
                   whole_set_scaled[:set_length[0], 1],
                   c='r', marker='.', s=20)
        ax.scatter(whole_set_scaled[set_length[0]:, 0],
                   whole_set_scaled[set_length[0]:, 1],
                   c='b', marker='.', s=20)
        ax.scatter(self._clf.support_vectors_[:, 0],
                   self._clf.support_vectors_[:, 1],
                   s=20, linewidth=0.5,
                   facecolors='none', edgecolors='k')

        x_min, x_max = ax.get_xlim()[0], ax.get_xlim()[1]
        y_min, y_max = ax.get_ylim()[0], ax.get_ylim()[1]

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.005),
                             np.arange(y_min, y_max, 0.005))
        xy = np.vstack([xx.ravel(), yy.ravel()]).T
        Z = self._clf.predict(xy).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=[-1, 0, 1, 2], alpha=0.2, colors=['red', 'blue'])
        ax.set_title('Scatter Diagram with the Optimal Readout Boundary\n' +
                     f'Fidelity={fidelity*100:.2f}%', fontsize=16)

        fig.savefig(self._exp.path.joinpath('Single shot measurement and optimal ' +
                                            f'readout boundary on Q{self._qubit} IQ plane.png'))
        plt.show()

    def fidelity(self):
        """Calculate readout fidelity based on confusion matrix.

        Returns
        -------
        float
            possibility of correctly predicting qubit state
        """
        c_matrix = self.c_matrix()
        infidelity = np.trace(np.fliplr(c_matrix))
        fidelity = 1 - infidelity

        return fidelity

    def _scale_to_train_set(self):
        whole_set, _ = self._set_assembly()
        minimum = np.min(whole_set, axis=0)
        maximum = np.max(whole_set, axis=0)
        scale = [4/(max - min) for (max, min) in zip(maximum, minimum)]
        data_rescaled = []
        for idx, quad in enumerate(range(2)):
            data_rescaled.append(self._data[self._qubit][quad]*scale[idx]-2-minimum[idx]*scale[idx])

        return data_rescaled

    def predict(self, exclude_calib=True):
        """Predict qubit state of all input data.

        Returns
        -------
        numpy.ndarray
            qubit state of all experiment data based on the model
            obtained from training set, with 0 being ground state,
            1 being excited state 
        """
        # rescale all data points according to training set
        data_rescaled = self._scale_to_train_set()
        xy = np.vstack([data_rescaled[0].ravel(),
                        data_rescaled[1].ravel()]).T
        prediction = self._clf.predict(xy).reshape(data_rescaled[0].shape)

        if exclude_calib and (not self._calib_seq is None):
            prediction = np.delete(prediction, self._calib_seq, axis=0)

        return prediction

class TwoQubitClassifier(QubitClassifier):

    def __init__(self, exp, qubit, calib_seq=(0, 1, 2, 3)):
        super().__init__(exp, qubit, calib_seq)

        if len(qubit) != 2:
            raise ValueError('Qubit number to be classified is wrong')

        if not set(qubit).issubset(range(exp.number_of_readout)):
            raise ValueError('Qubit index to be classified is wrong')

        if 'Sequence' in exp.scan_setup.loc['Target'].values:
            self._data = self._seq_concatenate()

            self._calib_seq = []
            for seq in range(self._seq_piece):
                self._calib_seq += list(map(lambda x, seq=seq: x + seq * self._exp.scan_size[0],
                                            calib_seq))
        else:
            self._data = exp.data
        
        self._clf = self._classify_boundary()

    def _set_assembly(self):
        whole_set = []
        set_length = []
        for calib in self._calib_seq[0:4]:
            whole_set_temp = []
            for res, quad in product(self._qubit, range(2)):
                whole_set_temp.append(self._data[res][quad][calib, :])
            whole_set.append(whole_set_temp)
            set_length.append(len(whole_set_temp[0]))
        whole_set = np.array(whole_set).transpose((1, 2, 0)).reshape((4, -1), order='F').T

        return whole_set, set_length

    def _train_set_assembly(self):
        whole_set, set_length = self._set_assembly()
        whole_set_scaled = minmax_scale(whole_set, feature_range=(-2, 2), axis=0)

        label = np.concatenate((np.full(set_length[0], 0, dtype=int),
                                np.full(set_length[1], 1, dtype=int),
                                np.full(set_length[2], 2, dtype=int),
                                np.full(set_length[3], 3, dtype=int)))

        return whole_set_scaled, label

    def _scale_to_train_set(self):
        whole_set, _ = self._set_assembly()
        minimum = np.min(whole_set, axis=0)
        maximum = np.max(whole_set, axis=0)
        scale = [4/(max - min) for (max, min) in zip(maximum, minimum)]
        data_rescaled = []
        for idx, value in enumerate(product(self._qubit, range(2))):
            res = value[0]
            quad = value[1]
            data_rescaled.append(self._data[res][quad]*scale[idx] -2 - minimum[idx]*scale[idx])

        return data_rescaled

    def predict(self, exclude_calib=True):
        data_rescaled = self._scale_to_train_set()
        xy = np.vstack([data_rescaled[0].ravel(),
                        data_rescaled[1].ravel(),
                        data_rescaled[2].ravel(),
                        data_rescaled[3].ravel(),]).T
        prediction = self._clf.predict(xy).reshape(data_rescaled[0].shape)

        if exclude_calib:
            prediction = np.delete(prediction, self._calib_seq, axis=0)

        return prediction
