# -*- mode: Python; coding: utf-8 -*-

from __future__ import division

from corpus import Document, BlogsCorpus, NamesCorpus
from maxent import MaxEnt
from numpy import zeros

import sys
from random import shuffle, seed
from unittest import TestCase, main, skip

from nltk.corpus import stopwords

global STOPWORDS
STOPWORDS = set(stopwords.words())


class EvenOdd(Document):
    def features(self):
        """Is the data even or odd?"""
        return {"even": self.data % 2 == 0}

class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""
        return dict((word, True) for word in self.data.split())

class MyInstance(Document):
    def features(self):
        feat = dict((word.lower(), True) for word in self.data.split()
                    if word not in STOPWORDS)
        return feat

class Name(Document):
    def features(self, letters="abcdefghijklmnopqrstuvwxyz"):
        """Based on NLTK's names_demo_features."""
        name = self.data.lower()
        features = {}
        for letter in letters:
            features["startswith(%s)" % letter] = name[0] == letter
            features["endswith(%s)" % letter] = name[-1] == letter
            features["has(%s)" % letter] = letter in name
        return features

def accuracy(classifier, test, verbose=sys.stderr):
    cm = zeros((len(classifier.labels) + 1, len(classifier.labels) + 4))
    for x in test:
        hypo = classifier.labels.encode(classifier.classify(x))
        gold = classifier.labels.encode(x.label)
        cm[hypo, gold] += 1
    cm[:,-4] = cm.sum(axis=1)
    cm[-1,:] = cm.sum(axis=0)
    for i in range(len(cm) - 1):
        pr = cm[i, i] / cm[i, -4]
        re = cm[i, i] / cm[-1, i]
        f1 = 2 * pr * re / (pr + re)
        cm[i, -3] = pr * 100
        cm[i, -2] = re * 100
        cm[i, -1] = f1 * 100
    cm_string = "\n\nConfusion Matrix\n" + \
                "(row = hypothesis; column = gold)\n" + \
                " " * 8 + ''.join(["%-8s" % str(classifier.labels.decode(x))
                                   for x in range(len(cm)-1)]) + \
                '%-8s' % 'total' + \
                '%-8s' % 'Pr' + '%-8s' % 'Re' + '%-8s' % 'F1' + '\n'
    for i in range(len(cm)-1):
        cm_string += "%-8s" % classifier.labels.decode(i) + \
                     ''.join(["%-8s" % str(int(x)) for x in cm[i]]) + '\n'
    cm_string += '%-8s' % 'total' + \
                 ''.join(["%-8s" % str(int(x)) for x in cm[-1]]) + '\n'
    acc = sum([cm[i, i] for i in range(len(cm)-1)]) * 1.0 / cm[-1, -4]
    cm_string += "Accuracy: %.4f\n" % acc
    if verbose:
        # print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
        print >> verbose, cm_string
    return acc

class NaiveBayesTest(TestCase):
    u"""Tests for the naïve Bayes classifier."""

    def test_even_odd(self):
        """Classify numbers as even or odd"""
        classifier = MaxEnt()
        classifier.train([EvenOdd(0, True), EvenOdd(1, False)])
        test = [EvenOdd(i, i % 2 == 0) for i in range(2, 1000)]
        self.assertEqual(accuracy(classifier, test), 1.0)

    def split_names_corpus(self, document_class=Name):
        """Split the names corpus into training and test sets"""
        names = NamesCorpus(document_class=document_class)
        self.assertEqual(len(names), 5001 + 2943) # see names/README
        seed(hash("names"))
        shuffle(names)
        return (names[:6000], names[6000:])

    def test_names_nltk(self):
        """Classify names using NLTK features"""
        train, test = self.split_names_corpus()
        classifier = MaxEnt()
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.70)

    def split_blogs_corpus(self, document_class):
        """Split the blog post corpus into training and test sets"""
        blogs = BlogsCorpus(document_class=document_class)
        self.assertEqual(len(blogs), 3227)
        seed(hash("blogs"))
        shuffle(blogs)
        return (blogs[:3000], blogs[3000:])

    def test_my_features(self):
        """Classify blog authors using my features"""
        train, test = self.split_blogs_corpus(MyInstance)
        classifier = MaxEnt()
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.55)

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)
