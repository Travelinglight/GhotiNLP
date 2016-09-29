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
        self.missingfn = missingfn or (lambda k, Pw: 1. / Pw.N)

    def __call__(self, key):
        if key in self:
            return float(self[key])/float(self.N)
        else:
            return self.missingfn(key, self)


class Segmenter():
    def __init__(self, pw, maxlen=None):
        self.pw = pw
        if maxlen is not None:
            self.maxlen = maxlen
        else:
            self.maxlen = pw.maxlen

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
            if self.pw(word) is not None:
                entry = (0, word)
                heappush(self.heap, entry)
                self.scores[entry] = math.log(self.pw(word))
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
                if self.pw(newword) is not None:
                    newentry = (endindex + 1, newword)
                    newscore = self.scores[entry] + math.log(self.pw(newword))
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


# 80.44 local; 88.55 leaderboard
def naive_limit_to_one(key, Pw):
    if len(key) == 1:
        return 1. / Pw.N
    else:
        return None


# 86.08 local; 91.47 leaderboard
def naive_punish_long_words(key, Pw):
    # 1 is also the minimal count in unigram
    return (1. / Pw.N) ** len(key)


Pw = Pdist(opts.counts1w, missingfn=naive_punish_long_words)
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
