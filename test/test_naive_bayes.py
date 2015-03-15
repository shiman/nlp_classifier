# -*- mode: Python; coding: utf-8 -*-

from __future__ import division

from corpus import Document, BlogsCorpus, NamesCorpus
from naive_bayes import NaiveBayes

import sys
from random import shuffle, seed
from unittest import TestCase, main, skip

from nltk.tokenize import word_tokenize
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
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
    return sum(correct) / len(correct)

class NaiveBayesTest(TestCase):
    u"""Tests for the naïve Bayes classifier."""

    def test_even_odd(self):
        """Classify numbers as even or odd"""
        classifier = NaiveBayes()
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
        classifier = NaiveBayes()
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.70)

    def split_blogs_corpus(self, document_class):
        """Split the blog post corpus into training and test sets"""
        blogs = BlogsCorpus(document_class=document_class)
        self.assertEqual(len(blogs), 3227)
        seed(hash("blogs"))
        shuffle(blogs)
        return (blogs[:3000], blogs[3000:])

    def test_blogs_bag(self):
        """Classify blog authors using bag-of-words"""
        train, test = self.split_blogs_corpus(BagOfWords)
        classifier = NaiveBayes()
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.55)

    def test_my_features(self):
        """Classify blog authors using my features"""
        train, test = self.split_blogs_corpus(MyInstance)
        classifier = NaiveBayes()
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.55)

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)
