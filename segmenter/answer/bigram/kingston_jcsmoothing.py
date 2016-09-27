# -*- coding: UTF-8 -*-
from __future__ import print_function
import sys, codecs, optparse, os, heapq, math, unicodedata

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
optparser.add_option("-u", "--unibigram", dest="unibigram", default=0.0, help="coefficient for combining bigram/unigram prob")
optparser.add_option("-d", "--delta", dest="delta", default=0.0, help="coeficient for bigram additive smoothing")
(opts, _) = optparser.parse_args()

class PdistUnigram(dict):
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
        tempKey = unicode(key.encode("utf8").replace('\xc2\xb7', '1'),"utf8")
        if unicodedata.normalize('NFKC', tempKey).isdigit():
            return len(unicodedata.normalize('NFKC', tempKey))/float(self.N)

        if key in self: return float(self[key])/float(self.N)
        elif len(key) == 1: return self.missingfn(key, self.N)
        else: return None

class PdistBigram(dict):
    "A probability distribution estimated from counts in bigram and unigram."

    def __init__(self, filename, pwUnigram, sep='\t', delta=0.0, N=None, missingfn=None):
        self.maxlen = 0
        self.pwUnigram = pwUnigram
        self.V = 0
        self.delta = delta

        for line in file(filename):
            self.V += 1
            (key, freq) = line.split(sep)
            (key1, key2) = key.split()
            try:
                utf8key1 = unicode(key1, 'utf-8')
                utf8key2 = unicode(key2, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key1,utf8key2] = self.get((utf8key1,utf8key2), 0) + int(freq)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        if key in self: return (float(self[key])/float(self.N) + self.delta) / (float(self.pwUnigram(key[0])) + self.delta * self.V)
        else: return self.delta / (float(self.pwUnigram(key[0])) + self.delta * self.V)

class Segmenter:
    "Perform segmentation to the input"

    def __init__(self, pwUnigram, pwBigram, maxlen, ub):
        self.pwUnigram = pwUnigram
        self.pwBigram = pwBigram
        self.maxlen = maxlen
        self.ub = ub    # ub: ub * bigram_prob + (1 - ub) * unigram_prob

    def segment(self, input):
        chart = []
        finalindex = len(input) - 1
        h = []
        res = []

        ## Initialize the heap ##
        for i in range(0, min(self.maxlen,len(input))):
            if not self.pwUnigram(input[0:i+1]) is None:
                entry = [0, input[0:i+1], math.log(self.pwUnigram(input[0:i+1]), 2), None]
                if not entry in h:
                    heapq.heappush(h, entry)

        ## Iteratively fill in chart[i] for all i ##
        for i in range(len(input)):
            chart.append(None)

        while(len(h) > 0):
            entry = heapq.heappop(h)
            endindex = entry[0] + len(entry[1]) - 1

            if(chart[endindex] is None):
                chart[endindex] = entry
            else:
                preventry = chart[endindex]
                if(entry[2] > preventry[2]):
                    chart[endindex] = entry
                else:
                    continue

            for i in range(endindex+1, min(endindex+1+self.maxlen,len(input))):
                if self.pwUnigram(input[endindex+1:i+1]) is not None:
                    biprobability = self.pwBigram((entry[1], input[endindex+1 : i+1]))
                    uniprobability = self.pwUnigram(input[endindex+1:i+1])
                    probability = math.log(self.ub * biprobability + (1 - self.ub) * uniprobability, 2)
                    newentry = [endindex+1, input[endindex+1:i+1], entry[2] + probability, entry]
                    heapq.heappush(h, newentry)

        ## Get the best segmentation ##
        finalentry = chart[finalindex]
        currentry = finalentry
        while(not(currentry is None)): 
            res.append(currentry[1])
            currentry = currentry[3]
        res = reversed(res)
        finalRes =  " ".join(res)

        return finalRes 

pwUnigram  = PdistUnigram(opts.counts1w)
pwBigram = PdistBigram(opts.counts2w, pwUnigram, delta=float(opts.delta))
segmenter = Segmenter(pwUnigram, pwBigram, pwUnigram.maxlen, float(opts.unibigram))
old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)

# perform segmentation and print result
with open(opts.input) as f:
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        print(segmenter.segment(utf8line))
sys.stdout = old