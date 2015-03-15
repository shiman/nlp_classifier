# -*- mode: Python; coding: utf-8 -*-

"""A simple framework for text classification."""

from abc import ABCMeta, abstractmethod, abstractproperty
from cPickle import dump, load, HIGHEST_PROTOCOL as HIGHEST_PICKLE_PROTOCOL

class Classifier(object):
    """An abstract text classifier.

    Subclasses must provide training and classification methods, as well as
    an implementation of the model property. The internal representation of
    a classifier's model is entirely up to the subclass, but the read/write
    model property must return/accept a single object (e.g., a list of
    probability distributions)."""

    __metaclass__ = ABCMeta

    def __init__(self, model=None):
        if isinstance(model, (basestring, file)):
            self.load(model)
        else:
            self.model = model

    def get_model(self): return None
    def set_model(self, model): pass
    model = abstractproperty(get_model, set_model)

    def save(self, file):
        """Save the current model to the given file."""
        if isinstance(file, basestring):
            with open(file, "wb") as file:
                self.save(file)
        else:
            dump(self.model, file, HIGHEST_PICKLE_PROTOCOL)

    def load(self, file):
        """Load a saved model from the given file."""
        if isinstance(file, basestring):
            with open(file, "rb") as file:
                self.load(file)
        else:
            self.model = load(file)

    @abstractmethod
    def train(self, instances):
        """Construct a statistical model from labeled instances."""
        pass

    @abstractmethod
    def classify(self, instance):
        """Classify an instance and return the expected label."""
        return None

class CodeBook(object):
    """A bi-directional map between names and auto-generated indices.
    Useful for both features and labels in classifiers."""

    def __init__(self, names):
        self.names = dict((index, name) for index, name in enumerate(names))
        self.index = dict((name, index) for index, name in enumerate(names))

    def __contains__(self, name):
        return name in self.index

    def __getitem__(self, name):
        return self.index[name]

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return len(self.index)

    def __repr__(self):
        return "<%s with %d entries>" % (self.__class__.__name__, len(self))

    def add(self, name):
        """Add the given name with a generated index."""
        if name not in self:
            index = len(self)
            self.names[index] = name
            self.index[name] = index
        return name

    def get(self, name, default=None):
        """Return the index associated with the given name."""
        return self.index.get(name, default)

    def name(self, index):
        """Return the name associated with the given index."""
        return self.names[index]

class DefaultCodeBook(CodeBook):
    """A codebook with a unique default object."""

    def __init__(self, names):
        super(DefaultCodeBook, self).__init__(names)
        self.default = self.add(object())

    def __getitem__(self, name):
        try:
            return super(DefaultCodeBook, self).__getitem__(name)
        except KeyError:
            return self[self.default]

    def get(self, name, default=None):
        """A default value of None denotes the default object, not itself."""
        index = super(DefaultCodeBook, self).get(name, default)
        return self.get(self.default) if index is None else index
