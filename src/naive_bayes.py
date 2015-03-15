# -*- mode: Python; coding: utf-8 -*-

from math import log
from classifier import Classifier


def laplace_smooth(conditional_count, prior_count):
    """Static method for laplace smoothing.
    :param conditional_count: a dict of conditional counts
    :param prior_count: a dict of prior counts
    """

    for k in conditional_count:
        conditional_count[k] += 1
    for k in prior_count:
        prior_count[k] += len(prior_count)


class NaiveBayes(Classifier):
    u"""A naïve Bayes classifier."""

    def __init__(self, model={}):
        super(NaiveBayes, self).__init__(model)
        self.OOV = 0  # a special key for unknown words

    def get_model(self):
        return self._model

    def set_model(self, model):
        self._model = model

    model = property(get_model, set_model)

    def train(self, instances):
        """
        Construct a statistical model from labeled instances.
        :param instances: iterable collection of labeled Document instances
        """
        labels = set([inst.label for inst in instances])
        self.model[-1] = labels  # special key to store labels

        prior_count = dict()
        total_doc = 0.0
        conditional_count = dict()
        for instance in instances:
            total_doc += 1
            the_label = instance.label
            prior_count[the_label] = prior_count.get(the_label, 0.0) + 1
            for label in labels:
                conditional_count[(self.OOV, label)] = 0.0
            for feature in instance.features().items():
                for label in labels:
                    key = (feature, label)
                    if label == the_label:
                        conditional_count[key] = conditional_count.get(key, 0.0)\
                                                 + 1
                    else:
                        conditional_count[key] = conditional_count.get(key, 0.0)

        laplace_smooth(conditional_count, prior_count)

        for label in prior_count:
            self.model[label] = log(prior_count[label] / total_doc)

        for key in conditional_count:
            feature, label = key
            self.model[key] = log(conditional_count[key] / prior_count[label])

    def classify(self, instance):
        """
        Classify an instance and return a hypothesis label.
        :param instance: an Document instance
        """
        argmax = float("-inf")
        guess = None
        for label in self.model[-1]:
            prior = self.model[label]
            conditional = list()
            for feature in instance.features().items():
                key = (feature, label)
                if key in self.model:
                    conditional.append(self.model[key])
                else:
                    oov_key = (self.OOV, label)
                    conditional.append(self.model[oov_key])
            conditional = sum(conditional)
            likelihood = prior + conditional
            if likelihood > argmax:
                argmax = likelihood
                guess = label
        return guess
