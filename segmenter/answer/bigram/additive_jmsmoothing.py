# -*- coding: UTF-8 -*-
from __future__ import print_function
from heapq import heappush, heappop
import sys, codecs, optparse, os, math, unicodedata

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
optparser.add_option("-k", "--lambda", dest="jm_lambda", default='0.9', help="Lambda for Jelinek-Mercer Smoothing")
optparser.add_option("-d", "--delta", dest="delta", default=0.001, help="coeficient for bigram additive smoothing")
(opts, _) = optparser.parse_args()

class Pdist(dict):
    "A probability distribution estimated from counts in unigram."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0
        self.V = 0  # the number of words in vocabulary

        for line in file(filename):
            self.V += 1
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        # Heuristics for digits (also try normalizing full-width digits)
        tempKey = unicodedata.normalize(
            'NFKC',
            # Replace full-width dots with 0 to form consecutive digits.
            unicode(key.encode("utf8").replace('\xc2\xb7', '0'), "utf8")
        )
        if tempKey.isdigit():
            return 1./self.N

        if key in self:
            return float(self[key])/float(self.N)
        else:
            return self.missingfn(key, self)


class Segmenter:
    "Perform segmentation to the input"

    def __init__(self, pwUnigram, pwBigram, pwCombine, maxlen=None):
        self.pwUnigram = pwUnigram
        self.pwBigram = pwBigram
        self.pwCombine = pwCombine
        self.maxlen = maxlen or self.pwUnigram.maxlen

    def segment(self, input):
        self.scores = {}
        self.prev = {}
        self.heap = []
        self.ans = []

        # entry: (start_index, word)
        # (Note: start/end_index are of the CURRENT word.)
        # end_index = start_index + len(word) - 1
        # scores[entry] = score of the entry
        # prev[entry] = previous_entry

        # Initialize the heap
        for i in range(min(len(input), self.maxlen)):
            word = input[0: i + 1]
            if self.pwUnigram(word) is not None:
                entry = (0, word)
                heappush(self.heap, entry)
                self.scores[entry] = math.log(self.pwUnigram(word))
                self.prev[entry] = None

        # Store the last entry which has the largest score so that we can backtrack.
        lastEntry = None
        # Dynamic Programming
        while self.heap:
            entry = heappop(self.heap)
            endindex = entry[0] + len(entry[1]) - 1

            if endindex == len(input) - 1 and \
                    (lastEntry is None or self.scores[entry] > self.scores[lastEntry]):
                lastEntry = entry

            for i in range(min(len(input) - 1 - endindex, self.maxlen)):
                newword = input[endindex + 1: endindex + 2 + i]
                if self.pwUnigram(newword) is not None:
                    newentry = (endindex + 1, newword)
                    newscore = self.scores[entry] + \
                        math.log(self.pwCombine(self.pwUnigram, self.pwBigram, newword, entry[1]))
                    if newentry not in self.heap:
                        heappush(self.heap, newentry)
                        self.scores[newentry] = newscore
                        self.prev[newentry] = entry
                    else:
                        if newscore > self.scores[newentry]:
                            self.scores[newentry] = newscore
                            self.prev[newentry] = entry

        # Get the best segmentation
        entry = lastEntry
        while entry is not None:
            self.ans = [entry[1]] + self.ans
            entry = self.prev[entry]

        return self.ans


# Note: using naive_limit_to_one as missingfn always gives worse result.
def naive_punish_long_words(key, Pw):
    # Cut off extremely long (and also unseen) words, which are super unlikely.
    if len(key) > 10:
        return None
    # 1 is also the minimal count in unigram.
    return (1. / Pw.N) ** len(key)


def bigram_missingfn(key, Pw):
    return 0


def additive_smoothing(cab, ca, ttl):
    delta = float(opts.delta)
    return float(cab + delta) / float(ca + delta * ttl)


# Lambda = 0.9 gives the best result, which is the default.
def cPw_jm_smoothing(Pw, P2w, current_word, prev_word):
    Lambda = float(opts.jm_lambda)
    cab = 0
    if prev_word in Pw:
        cab = Pw[prev_word]
    ca = 0
    if prev_word + ' ' + current_word in P2w:
        ca = P2w[prev_word + ' ' + current_word]

    p2 = additive_smoothing(cab, ca, Pw.V)

    return p2 * Lambda + Pw(current_word) * (1 - Lambda)


pwUnigram = Pdist(opts.counts1w, missingfn=naive_punish_long_words)
pwBigram = Pdist(opts.counts2w, missingfn=bigram_missingfn)
segmenter = Segmenter(pwUnigram, pwBigram, cPw_jm_smoothing)
old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)

# perform segmentation and print result
with open(opts.input) as f:
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        print(" ".join(segmenter.segment(utf8line)))
sys.stdout = old