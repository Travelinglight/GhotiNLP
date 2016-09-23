import pdb
import sys, codecs, optparse, os
import math
from heapq import heappush, heappop

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()

class Pdist1(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0
        self.alpha = 0
        self.nLine = 0

        for line in file(filename):
            self.nLine += 1
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
        if key in self: return (float(self[key]) + self.alpha)/ (float(self.N) + self.alpha * self.nLine)
        #else: return self.alpha / (self.alpha * self.nLine)
        #else: return self.missingfn(key, self.N)
        elif len(key) == 1: return self.missingfn(key, self.N)
        else: return 0

class Pdist2(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0
        self.alpha = 0
        self.nLine = 0

        for line in file(filename):
            self.nLine += 1
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))

            self[utf8key] = self.get(utf8key, 0) + int(freq)

        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        if key in self: return float(self[key])/float(self.N)
        #else: return self.missingfn(key, self.N)
        else: return 0

class Segmenter():
    def __init__(self, pw1, pw2):
        self.pw1 = pw1
        self.pw2 = pw2
        self.maxlen = pw1.maxlen

    def segment(self, input, k):
        self.chart = [[] for i in range(len(input))]
        self.heap = []
        self.ans = []

        # Initialize the heap
        for i in range(min(len(input), self.maxlen)):
            word = input[0: i + 1]
            if self.pw1(word) is not None:
                heappush(self.heap, (0, math.log(self.pw1(word)), word, None))

        # Iteratively fill in chart[i] for all i
        while self.heap:
            entry = heappop(self.heap)
            endindex =  entry[0] + len(entry[2]) - 1

            if len(self.chart[endindex]) == k:
                if entry[1] > self.chart[endindex][0][1][1]:
                    heappop(self.chart[endindex])
                    heappush(self.chart[endindex], (entry[1], entry))
                else:
                    continue
            else:
                heappush(self.chart[endindex], (entry[1], entry))

            for i in range(min(len(input) - 1 - endindex, self.maxlen)):
                newword = input[endindex + 1 : endindex + 2 + i]
                if self.pw1(newword) is not None:
                    prob = self.pw2(entry[2] + ' ' + newword)
                    if prob is None:
                        prob = self.pw1(newword) * self.pw1(entry[2])
                    newentry = (endindex + 1,
                                entry[1] + math.log(prob) - math.log(self.pw1(entry[2])),
                                newword,
                                entry)

                    if not newentry in self.heap:
                        heappush(self.heap, newentry)

        # Get the best segmentation

        maxprob = self.chart[len(input) - 1][0][0]
        for i in self.chart[len(input) - 1]:
            if (i[0] > maxprob):
                maxprob = i[0]
                entry = i[1]

        while entry is not None:
            self.ans = [entry[2]] + self.ans
            entry = entry[3]

        return self.ans

Pw1 = Pdist1(opts.counts1w)
Pw2 = Pdist2(opts.counts2w)
seg = Segmenter(Pw1, Pw2)

old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
# ignoring the dictionary provided in opts.counts
with open(opts.input) as f:
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        output = seg.segment(utf8line, 5)
        print " ".join(output)
sys.stdout = old
