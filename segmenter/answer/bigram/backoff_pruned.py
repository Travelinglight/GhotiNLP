# -*- coding: UTF-8 -*-
from __future__ import print_function
from heapq import heappush, heappop
import sys, codecs, optparse, os, math, unicodedata

## k = 0 unigram, k = 1 bigram, the version combine unigram and bigram
k = float(0.4)
backoff = float(0)
lamf = 0.0026
nxy = {}
optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
optparser.add_option("-u", "--unibigram", dest="unibigram", default=0.4, help="u == 0: pure unigram")
(opts, _) = optparser.parse_args()

class PdistUnigram(dict):
    "A probability distribution estimated from counts in unigram."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0
        for line in file(filename):
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N*(lamf**(k-1)))

    def __call__(self, key):
        tempKey = unicode(key.encode("utf8").replace('\xc2\xb7', '1'),"utf8")
        if unicodedata.normalize('NFKC', tempKey).isdigit():
            return len(unicodedata.normalize('NFKC', tempKey))/float(self.N)

        if key in self: return float(self[key])/float(self.N)
        elif len(key) <= 3: return self.missingfn(len(key), self.N)
        else: return None

class PdistBigram(dict):
    "A probability distribution estimated from counts in bigram and unigram."
    def __init__(self, filename, pwUnigram, sep='\t', N=None, missingfn=None):
        self.maxlen = 0 
        self.pwUnigram = pwUnigram
        for line in file(filename):
            (key, freq) = line.split(sep)
            (key1, key2) = key.split()
            try:
                utf8key1 = unicode(key1, 'utf-8')
                utf8key2 = unicode(key2, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key1,utf8key2] = self.get((utf8key1,utf8key2), 0) + int(freq)
            self.maxlen = max(len(utf8key1), len(utf8key2), self.maxlen)
            if utf8key1 in nxy:
                nxy[utf8key1] += 1
            else:
                nxy[utf8key1] = 1
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        if key in self: return float(self[key]-backoff)/(float(self.N) * float(self.pwUnigram(key[0])))
        elif key[0] in nxy: return backoff*nxy[key[0]] * self.pwUnigram(key[1])
        else: return self.pwUnigram(key[1])

class Segmenter:
    "Perform segmentation to the input"

    def __init__(self, pwUnigram, pwBigram, maxlen, unibigram):
        self.pwUnigram = pwUnigram
        self.pwBigram = pwBigram
        self.maxlen = maxlen
        self.ub = unibigram

    def __call__(self, input):
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
                    biprobability = self.pwBigram((entry[1], newword))
                    uniprobability = self.pwUnigram(newword)
                    probability = self.ub * biprobability + (1 - self.ub) * uniprobability

                    newentry = (endindex + 1, newword)
                    newscore = self.scores[entry] + math.log(probability, 2)
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



pwUnigram  = PdistUnigram(opts.counts1w)
pwBigram = PdistBigram(opts.counts2w, pwUnigram)
segmenter = Segmenter(pwUnigram, pwBigram, pwUnigram.maxlen, float(opts.unibigram))
old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)

# perform segmentation and print result
with open(opts.input) as f:
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        print(" ".join(segmenter(utf8line)))
sys.stdout = old