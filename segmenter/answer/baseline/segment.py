import sys, codecs, optparse, os
import math
from heapq import heappush, heappop

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."

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
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        if key in self: return float(self[key])/float(self.N)
        elif len(key) == 1: return self.missingfn(key, self.N)
        else: return None

class Segmenter():
    def __init__(self, pw, maxlen=None):
        self.pw = pw
        if maxlen is not None:
            self.maxlen = maxlen
        else:
            self.maxlen = pw.maxlen

    def segment(self, input):
        self.chart = [None for i in range(len(input))]
        self.scores = {}
        self.heap = []
        self.ans = []

        # entry: (start_index, word)
        # (Note: start/end_index are of the CURRENT word.)
        # end_index = start_index + len(word) - 1
        # scores[entry] = score of the entry
        # chart[end_index] = entry

        # Initialize the heap
        for i in range(min(len(input), self.maxlen)):
            word = input[0: i + 1]
            if self.pw(word) is not None:
                heappush(self.heap, (0, word))
                self.scores[(0, word)] = math.log(self.pw(word))

        # Iteratively fill in chart[i] for all i
        while self.heap:
            entry = heappop(self.heap)
            endindex = entry[0] + len(entry[1]) - 1

            if self.chart[endindex] is not None:
                if self.scores[entry] > self.scores[self.chart[endindex]]:
                    self.chart[endindex] = entry
                else:
                    continue
            else:
                self.chart[endindex] = entry

            for i in range(min(len(input) - 1 - endindex, self.maxlen)):
                newword = input[endindex + 1: endindex + 2 + i]
                if self.pw(newword) is not None:
                    newentry = (endindex + 1, newword)
                    newscore = self.scores[entry] + math.log(self.pw(newword))
                    if newentry not in self.heap:
                        heappush(self.heap, newentry)
                        self.scores[newentry] = newscore
                    else:
                        self.scores[newentry] = max(self.scores[newentry], newscore)

        # Get the best segmentation
        entry = self.chart[len(input) - 1]
        while True:
            self.ans = [entry[1]] + self.ans
            if entry[0] == 0:
                break
            else:
                entry = self.chart[entry[0] - 1]

        return self.ans

Pw  = Pdist(opts.counts1w)
seg = Segmenter(Pw)

old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
# ignoring the dictionary provided in opts.counts
with open(opts.input) as f:
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        output = seg.segment(utf8line)
        print " ".join(output)
sys.stdout = old
