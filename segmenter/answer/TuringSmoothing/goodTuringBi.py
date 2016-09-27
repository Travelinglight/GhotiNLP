# -*- coding: UTF-8 -*-
from __future__ import print_function
import sys, codecs, optparse, os, heapq, math, unicodedata

## k = 0 unigram, k = 1 bigram, the version combine unigram and bigram
k = float(1)

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()

class PdistUnigram(dict):
    "A probability distribution estimated from counts in unigram."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.count = {}
        self.maxlen = 0
        for line in file(filename):
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self.count[int(freq)] = self.count.get(int(freq), 0) + 1
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        tempKey = unicode(key.encode("utf8").replace('\xc2\xb7', '1'),"utf8")
        if unicodedata.normalize('NFKC', tempKey).isdigit():
            return len(unicodedata.normalize('NFKC', tempKey))/float(self.N)

        if key in self:
            return float(self[key])/float(self.N)
        elif len(key) == 1:
            return self.missingfn(key, self.N)
        else:
            return None

class PdistBigram(dict):
    "A probability distribution estimated from counts in bigram and unigram."

    def __init__(self, filename, PwUnigram, sep='\t', N=None, missingfn=None):
        self.maxlen = 0
        self.PwUniBigram = PwUnigram
        self.count = {}

        for line in file(filename):
            (key, freq) = line.split(sep)
            (key1, key2) = key.split()
            try:
                utf8key1 = unicode(key1, 'utf-8')
                utf8key2 = unicode(key2, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self.count[int(freq)] = self.count.get(int(freq), 0) + 1
            self[utf8key1,utf8key2] = self.get((utf8key1,utf8key2), 0) + int(freq)
            self.maxlen = max(len(utf8key1), len(utf8key2), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        if key in self:
            if int(self[key]) > 3:
                closest_key_nr = min(PwUnigram.count, key=lambda y:abs(float(y-int(self[key]))))
                closest_key_nrp = min(PwUnigram.count, key=lambda y:abs(float(y-int(self[key]-1))))
                Nr = PwUnigram.count[closest_key_nr]
                Nrpuls = PwUnigram.count[closest_key_nrp]
                probability = float((int(self[key]) + 1) * Nrpuls) / float(self.N * Nr * PwUnigram(key[0]))
                return probability
            else:
                return float(self[key])/(float(self.N) * float(PwUnigram(key[0])))
        else:
            return PwUnigram(key[1])

class segmenter:
    "Perform segmentation to the input"

    def __init__(self, PwUnigram, PwBigram, maxlen):
        self.PwUnigram = PwUnigram
        self.PwBigram = PwBigram
        self.maxlen = maxlen

    def __call__(self, input):
        chart = []
        finalindex = len(input) - 1
        h = []
        res = []

        ## Initialize the heap ##
        for i in range(0, min(self.maxlen,len(input))):
            if not PwUnigram(input[0:i+1]) is None:
                entry = [0, input[0:i+1], math.log(PwUnigram(input[0:i+1]), 2), None]
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
                if not PwUnigram(input[endindex+1:i+1]) is None:
                    biprobability = PwBigram((entry[1], input[endindex+1:i+1]))
                    uniprobability = PwUnigram(input[endindex+1:i+1])
                    probability = math.log(k * biprobability + (1-k) * uniprobability, 2)
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

PwUnigram  = PdistUnigram(opts.counts1w)
PwBigram = PdistBigram(opts.counts2w, PwUnigram)
Segmenter = segmenter(PwUnigram, PwBigram, PwUnigram.maxlen)
old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)

# perform segmentation and print result
with open(opts.input) as f:
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        print(Segmenter(utf8line))
sys.stdout = old