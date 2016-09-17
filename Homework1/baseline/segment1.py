import sys, codecs, optparse, os
import math

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
optparser.add_option("-m", "--maxwordlength", dest="maxlen", default=15, help="maximum possible word length")
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
        #else: return self.missingfn(key, self.N)
        elif len(key) == 1: return self.missingfn(key, self.N)
        else: return None

class Segmenter():
    def __init__(self, pw, maxlen):
        self.pw = pw
        self.maxlen = maxlen

    def segment(self, input):
        self.chart = [None for i in range(len(input))]
        self.ans = []

        # Iteratively fill in chart[i] for all i
        for i in range(len(input)):
            for j in range(min(len(input) - i, self.maxlen)):
                word = input[i : i + j + 1]
                if self.pw(word) is not None:
                    if i == 0:
                        probability = math.log(self.pw(word))
                    else:
                        probability = math.log(self.pw(word)) + self.chart[i - 1][1]

                    newentry = [i, probability, word]
                    if self.chart[i + j] is None or self.chart[i + j][1] < probability:
                        self.chart[i + j] = newentry

        # Get the best segmentation
        index = len(input) - 1
        while index >= 0:
            self.ans = [self.chart[index][2]] + self.ans
            index = self.chart[index][0] - 1

        return self.ans

Pw  = Pdist(opts.counts1w)
seg = Segmenter(Pw, opts.maxlen)

old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
# ignoring the dictionary provided in opts.counts
with open(opts.input) as f:
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        output = seg.segment(utf8line)
        print " ".join(output)
sys.stdout = old
