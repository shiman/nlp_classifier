#!/usr/bin/python
# coding: utf-8
from __future__ import unicode_literals
from collections import defaultdict
from nltk import Tree
from nltk import word_tokenize


class Production(object):
    """
    A production rule
    """

    def __init__(self, lhs, rhs):
        """
        :param lhs: left hand side, a single non-terminal
        :param rhs: right hand side, a list of terminals or non-terminals
        """
        self.lhs = lhs
        self.rhs = rhs

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, Production):
            return False
        return self.lhs == other.lhs and self.rhs == other.rhs

    def __str__(self):
        return self.lhs + ' -> ' + ' '.join(self.rhs)

    def match(self, tokens):
        """Check if a list of nodes matches this rule"""
        symbols = list()
        for tok in tokens:
            if isinstance(tok, Tree): symbols.append(tok.label())
            else: symbols.append(tok)
        return symbols == self.rhs

    def is_branching(self):
        return len(self.rhs) > 1


class CKYParser(object):
    """
    A probablistic CKY parser.
    """

    def __init__(self, rules=None):
        """
        :param rules: a file or a file path
        """
        self.productions = self._get_rules(rules)
        self.symbols = [prod.lhs for prod in self.productions]
        self.start_symbol = 'S'

    def _get_rules(self, rules):
        """Parse rule file into ``Production``s"""
        if isinstance(rules, basestring):
            with open(rules) as f:
                return self._get_rules(f)
        else:
            productions = dict()
            for line in rules.readlines():
                if line.startswith('#'): continue
                if line.strip() == '': continue
                tokens = line.strip().split('\t')
                if len(tokens) != 3:
                    raise Exception("Unable to parse grammar file")
                lhs = tokens[0].strip()
                rhs = tokens[1].strip().split()
                prob = float(tokens[2].strip())
                productions[Production(lhs, rhs)] = prob
            return productions

    def _buildup(self, tokens, d):
        """
        Recursively build up all possible subtrees from given tokens (mainly
        for unary nodes).
        e.g. book -> NN | book -> NN -> NP | book -> V | book -> V -> VP -> S

        :param tokens: the elements that you want to build with
        :param d: the dict (representing the 3rd dimension of the table) that
                  you want to put the subtrees into
        """
        for prod in self.productions:
            if prod.match(tokens):
                tree = Tree(prod.lhs, tokens)
                tree.prob = self.productions[prod]
                for tok in tokens:
                    if isinstance(tok, Tree):
                        tree.prob *= tok.prob
                original = d[tree.label()]
                if original is None or tree.prob > original.prob:
                    d[tree.label()] = tree
                self._buildup([tree], d)

    def parse(self, sentence, pos_tagger=None, prob=True):
        """
        Parse a sentence into an NLTK Tree instance
        :param sentence: a sentence string
        :param pos_tagger: an external POS tagger, if available
        :rtype: (nltk.Tree, float)
        """
        # initialize the table
        tokens = word_tokenize(sentence)
        size = len(tokens)
        table = [None] * size
        for i in range(size): table[i] = [None] * size
        for i in range(size):
            for j in range(size):
                table[i][j] = dict([(x, None) for x in self.symbols])

        # fill the leaf level POS subtrees
        if pos_tagger is not None:
            tags = pos_tagger.classify(tokens)
            for i in range(size):
                tree = Tree(tags[i], [tokens[i]])
                tree.prob = 1.0
                self._buildup([tree], table[i][i])
        else:
            for i in range(size):
                self._buildup([tokens[i]], table[i][i])

        # build up the whole pyramid
        for j in range(1, size):  # for each word index / column
            for i in range(j-1, -1, -1):  # for each row
                for k in range(i, j):  # for each split point
                    for prod in self.productions:
                        if prod.is_branching():
                            daughterA = table[i][k][prod.rhs[0]]
                            daughterB = table[k+1][j][prod.rhs[1]]
                            self._buildup((daughterA, daughterB), table[i][j])
        parsed = table[0][-1][self.start_symbol]
        if not prob:
            return parsed
        if parsed is None:
            return None, 0.0
        return parsed, parsed.prob


if __name__ == '__main__':
    parser = CKYParser(rules='grammar.txt')
    with open('input.txt') as f:
        print "Parsing 5 sentences..."
        for line in f:
            tree, prob = parser.parse(line.strip())
            print line
            print tree, prob
            print
