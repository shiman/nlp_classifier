# -*- mode: Python; coding: utf-8 -*-

from numpy import zeros, array, argmax
from classifier import Classifier, DefaultCodeBook, CodeBook
from cPickle import dump, load, HIGHEST_PROTOCOL as HIGHEST_PICKLE_PROTOCOL


class HMM(Classifier):
    """A Hidden Markov Model classifier."""

    def __init__(self):
        self.vocabulary = DefaultCodeBook([])
        self.states = CodeBook([])
        self.emission_probabilities = array([])
        self.transition_probabilities = array([])
        self.initial_probabilities = array([])

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def load(self, file):
        """Load a saved model from the given file and unpack."""
        if isinstance(file, basestring):
            with open(file, "rb") as file:
                self.load(file)
        else:
            pack = load(file)
            self.vocabulary, self.states, \
                self.initial_probabilities, \
                self.emission_probabilities, \
                self.transition_probabilities = pack

    def save(self, file):
        """Pack and save the current model to the given file."""
        if isinstance(file, basestring):
            with open(file, "wb") as file:
                self.save(file)
        else:
            to_save = (self.vocabulary, self.states,
                       self.initial_probabilities,
                       self.emission_probabilities,
                       self.transition_probabilities)
            dump(to_save, file, HIGHEST_PICKLE_PROTOCOL)

    def train(self, instances, initial_probabilities=None,
              transition_probabilities=None,
              emission_probabilities=None,
              states=None, vocabulary=None):
        if initial_probabilities and transition_probabilities and \
           emission_probabilities and states and vocabulary:
            self.initial_probabilities = array(initial_probabilities)
            self.transition_probabilities = array(transition_probabilities).T
            self.emission_probabilities = array(emission_probabilities).T
            self.states = CodeBook(states)
            self.vocabulary = CodeBook(vocabulary)
            return

        for instance in instances:
            for ob in instance.features():
                self.vocabulary.add(ob)
            for label in instance.label:
                self.states.add(label)
        self.emission_probabilities = zeros((len(self.vocabulary),
                                             len(self.states)))
        self.transition_probabilities = zeros((len(self.states),
                                               len(self.states)))
        self.initial_probabilities = zeros(len(self.states))

        # Get emission and transition counts
        for instance in instances:
            if len(instance.features()) == 0: continue
            initial_state_index = self.states.get(instance.label[0])
            self.initial_probabilities[initial_state_index] += 1
            for i in range(len(instance.features()) - 1):
                this_ob = self.vocabulary.get(instance.features()[i])
                this_state = self.states.get(instance.label[i])
                next_state = self.states.get(instance.label[i+1])
                self.emission_probabilities[this_ob, this_state] += 1.0
                self.transition_probabilities[next_state, this_state] += 1.0

        # Laplace smoothing
        self.initial_probabilities += 1.0
        self.transition_probabilities += 1.0
        self.emission_probabilities += 1.0

        # Get real emission and transition probabilities
        self.initial_probabilities /= self.initial_probabilities.sum()
        self.emission_probabilities /= self.emission_probabilities.sum(axis=0)
        self.transition_probabilities /= self.transition_probabilities.sum(axis=0)

    def likelihood(self, instance):
        """Use forward algorithm to get the likelihood of an observation"""
        if len(instance.features()) == 0:
            return 0.0
        observation = [self.vocabulary.get(x) for x in instance.features()]
        trellis = zeros((len(self.states), len(instance.features())))
        trellis[:, 0] = self.initial_probabilities * \
            self.emission_probabilities[observation[0]]
        for i in range(len(observation)-1):
            next_ob = observation[i+1]
            for state in range(trellis.shape[0]):
                trellis[:, i+1] += self.transition_probabilities[:, state] * \
                    self.emission_probabilities[next_ob] * \
                    trellis[state, i]
        return trellis[:, -1].sum()

    def classify(self, instance):
        """Use viterbi algorithm to decode the instance into states"""
        if len(instance.features()) == 0:
            return []
        observation = [self.vocabulary.get(x) for x in instance.features()]
        trellis = zeros((len(self.states), len(instance.features())))
        trellis[:, 0] = self.initial_probabilities * \
            self.emission_probabilities[observation[0]]
        backpointers = zeros(trellis.shape)
        for i in range(len(observation)-1):
            next_ob = observation[i+1]
            for state in range(trellis.shape[0]):
                for next_state in range(trellis.shape[0]):
                    edge_value = self.transition_probabilities[next_state, state] * \
                        self.emission_probabilities[next_ob, next_state] * \
                        trellis[state, i]
                    if edge_value > trellis[next_state, i+1]:
                        trellis[next_state, i+1] = edge_value
                        backpointers[next_state, i+1] = state
        labels = [''] * len(observation)
        time = len(observation) - 1
        state = argmax(trellis[:, -1])
        labels[time] = self.states.name(state)
        while time > 0:
            state = backpointers[state, time]
            labels[time-1] = self.states.name(state)
            time -= 1
        return labels
