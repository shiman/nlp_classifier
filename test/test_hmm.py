# -*- mode: Python; coding: utf-8 -*-

from __future__ import division

from corpus import Document
from hmm import HMM

from nltk.corpus import treebank as tagged_corpus # could also use brown, etc.
# from nltk.corpus import brown as tagged_corpus
from nltk.stem.wordnet import WordNetLemmatizer
global lemmatizer
lemmatizer = WordNetLemmatizer()

import sys
from unittest import TestCase, main, skip

class IceCreamCones(Document):
    def features(self):
        """How many ice cream cones were consumed on each day?"""
        return self.data # counts

class IceCreamHMM(TestCase):
    def setUp(self):
        """Initialize Eisner ice cream HMM (J & M, Figure 6.3)"""
        self.hmm = HMM()
        # These variables have many aliases. J & M call them π, A, B, Q, and V.
        # You don't need to use these names, but you do need to provide a way
        # of initializing them.
        self.hmm.train([],
                       initial_probabilities=[.8, .2], # P(Hot, Cold)
                       transition_probabilities=[[.7, .3],  # P(Hot|Hot, Cold)
                                                 [.4, .6]], # P(Cold|Hot, Cold)
                       emission_probabilities=[[.2, .4, .4],  # P(1, 2, 3|Hot)
                                               [.5, .4, .1]], # P(1, 2, 3|Cold)
                       states=("Hot", "Cold"),
                       vocabulary=(1, 2, 3))

    def test_likelihood(self):
        """Test likelihood for Eisner ice cream HMM (J & M, Figure 6.7)"""
        # Figure 6.7 of J & M (slide 15 of Lecture6_Handout.pdf, 2014-10-15)
        # has a known erratum in the computation of alpha_2(2): .7*.2 = .14,
        # not .014.
        self.assertAlmostEqual(self.hmm.likelihood(IceCreamCones([3, 1])),
                               (.32*.14 + .02*.08) + (.32*.15 + .02*.30))

    def test_decoding(self):
        """Test decoding of Eisner ice cream HMM (J & M, Section 6.4)"""
        # The same error occurs in Figure 6.10, but the value given for the
        # Viterbi variable v_2(2) is .0448, which is correct (as you should
        # verify manually and perhaps add a test for here).
        self.assertEqual(self.hmm.classify(IceCreamCones([3, 1, 3])),
                         ["Hot", "Hot", "Hot"])

class TaggedSentence(Document):
    """Features are words, labels are part-of-speech tags."""

    def __init__(self, data, label=None, *args, **kwargs):
        if data and not label:
            # Data is assumed to be NLTK-style (word, tag) pairs.
            # If you'd like to collapse the tag set, this is the place.
            label = [tag for word, tag in data] # e.g., tag[0]
            data = [word.lower() for word, tag in data]
        super(TaggedSentence, self).__init__(data, label, *args, **kwargs)

    def features(self):
        # return [lemmatizer.lemmatize(x) for x in self.data]  # Lemmatize
        return self.data # words

class TagHMM(TestCase):
    """Train and test an HMM POS tagger."""

    def setUp(self):
        self.train, self.test = self.split_sents()
        self.hmm = HMM()
        self.hmm.train(self.train)

    def split_sents(self, train=0.95, total=3500,
                    document_class=TaggedSentence):
        sents = tagged_corpus.tagged_sents()[:total]
        total = len(sents) if total is None else total
        i = int(round(train * total))
        j = i + int(round(total - train * total))
        return (map(document_class, sents[0:i]),
                map(document_class, sents[i:j]))

    def accuracy(self, test_sents, verbose=sys.stderr):
        """Compute accuracy of the HMM tagger on the given sentences."""
        total = correct = 0
        for sent in test_sents:
            tags = self.hmm.classify(sent)
            total += len(tags)
            for guess, tag in zip(tags, sent.label):
                correct += (guess == tag)
        if verbose:
            print >> verbose, "%.2d%% " % (100 * correct / total),
        return correct / total

    # @skip("too slow")
    def test_tag_train(self):
        """Tag the training data"""
        self.assertGreater(self.accuracy(self.train), 0.85)

    def test_tag(self):
        """Tag the test data"""
        self.assertGreater(self.accuracy(self.test), 0.85)

if __name__ == "__main__":
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)
