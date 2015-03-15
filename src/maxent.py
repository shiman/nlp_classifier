from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import logsumexp
from numpy import array, exp, zeros, ones, argmax
from cPickle import dump, load, HIGHEST_PROTOCOL as HIGHEST_PICKLE_PROTOCOL

from classifier import Classifier
from codebook import Codebook


class MaxEnt(Classifier):
    """A Maximum Entropy classifier"""

    def __init__(self):
        super(MaxEnt, self).__init__()
        self.weights = array([])
        self.features = Codebook(start=1)
        self.labels = Codebook(decode=True)
        self.data = list()

    @property
    def model(self):
        return self.weights

    @model.setter
    def model(self, model):
        self.weights = model

    def save(self, file):
        """Pack and save the current model to the given file."""
        if isinstance(file, basestring):
            with open(file, "wb") as file:
                self.save(file)
        else:
            to_save = (self.model, self.features, self.labels)
            dump(to_save, file, HIGHEST_PICKLE_PROTOCOL)

    def load(self, file):
        """Load a saved model from the given file and unpack."""
        if isinstance(file, basestring):
            with open(file, "rb") as file:
                self.load(file)
        else:
            pack = load(file)
            self.model, self.features, self.labels = pack

    def _get_normalize_vec(self, feat_vec, weights=None):
        """
        Get the normalize vector of a posterior (Z) without its exp()
        :param feat_vec: feature vector
        :param weights: weights (2-d array)
        :return: Z vector
        """
        if weights is None:
            weights = self.weights
        return array([row[0] + row[feat_vec].sum() for row in weights])

    def _objective_func(self, weights):
        weights = weights.reshape((len(self.labels), len(self.features)+1))
        prior = self.regularize * (weights ** 2 / (2 * self.gaussian_sigma ** 2)).sum()
        numerator = 0.0
        denominator = 0.0
        for d in self.data:
            z_vec = self._get_normalize_vec(d[1], weights)
            numerator += z_vec[d[0]] + weights[d[0], 0]
            denominator += logsumexp(z_vec)
        likelihood = numerator - denominator - prior
        return -likelihood

    def _get_expected_counts(self, weights):
        expected_counts = zeros(weights.shape)
        for d in self.data:
            z_vec = self._get_normalize_vec(d[1], weights)
            posterior_vec = exp(z_vec - logsumexp(z_vec))
            expected_counts[:, 0] += posterior_vec
            for label_i in range(len(self.labels)):
                expected_counts[label_i][d[1]] += posterior_vec[label_i]
        return expected_counts

    def _get_gradient(self, weights):
        weights = weights.reshape((len(self.labels), len(self.features)+1))
        prior = self.regularize * (weights / self.gaussian_sigma ** 2)
        expected_counts = self._get_expected_counts(weights)
        gradient = self.observed_counts - expected_counts - prior
        return -gradient.ravel()

    def _get_observed_counts(self):
        self.observed_counts = zeros((len(self.labels), len(self.features)+1))
        for d in self.data:
            self.observed_counts[d[0]][d[1]] += 1
            self.observed_counts[d[0], 0] += 1

    def train(self, instances, regularize=False, gaussian_sigma=1.0, disp=False):
        """
        Train with given instances and optimize the parameters with L-BFGS
        algorithm.
        :param instances: the instance list
        :param regularize: whether you want to regularize the parameters
        :param gaussian_sigma: the variance of regularization
        :param disp: whether you want to show the optimization details.
        """
        for d in instances:
            self.features.add(d.features().items())
            self.labels.add(d.label)
            self.data.append((self.labels.encode(d.label),
                              self.features.encode(d.features().items())))
        self._get_observed_counts()
        self.weights = ones((len(self.labels), len(self.features)+1))
        self.gaussian_sigma = gaussian_sigma
        if regularize: self.regularize = 1
        else: self.regularize = 0

        best, _, d = fmin_l_bfgs_b(self._objective_func, self.weights,
                                   fprime=self._get_gradient, disp=disp)
        self.weights = best.reshape((len(self.labels), len(self.features)+1))
        if disp:
            for k, v in d.items():
                print k, v

    def classify(self, instance):
        feat_vec = self.features.encode(instance.features().items())
        label_index = argmax(self._get_normalize_vec(feat_vec))
        return self.labels.decode(label_index)
