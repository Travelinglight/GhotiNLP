#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict
import math

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

sys.stderr.write("Training with baseline algorithm...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

t_old = defaultdict(float)
t_new = defaultdict(float)
f_voc = defaultdict(int)    # French vocabulary, later convert to vocabulary size
count_fe = defaultdict(float)
count_e = defaultdict(float)

# calculate French vocabulary size
for (n, (f, e)) in enumerate(bitext):
    for f_i in set(f):
        f_voc[f_i] += 1
f_voc = len(f_voc.items())

# initialize t0 uniformly
for (n, (f, e)) in enumerate(bitext):
    for f_i in set(f):
        for e_j in set(e):
            t_old[(f_i, e_j)] = float(1) / f_voc

# calculate probabilities
L_old = -float('inf')
while True:
    count_fe.clear()
    count_e.clear()
    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            z = 0
            for e_j in set(e):
                z += t_old[(f_i, e_j)]
            for e_j in set(e):
                c = t_old[(f_i, e_j)] / z
                count_fe[(f_i, e_j)] += c
                count_e[e_j] += c

    for (f_i, e_j) in count_fe:
        t_new[(f_i, e_j)] += count_fe[(f_i, e_j)] / count_e[e_j]

    Pr = defaultdict(lambda:1.0)

    L_new = 0.0
    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            pr = 0.0
            for e_j in set(e):
                pr += t_new[f_i, e_j]

            Pr[n] *= pr
        L_new += Pr[n]
    L_new = math.log(L_new)
    if L_new - L_old < 10**(-4):
        break

    sys.stderr.write(str(L_new))
    sys.stderr.write("\n")
    L_old = L_new


# decode
a = defaultdict(int)
for (n, (f, e)) in enumerate(bitext):
    for i, f_i in enumerate(f):
        bestp = 0
        bestj = 0
        for j, e_j in enumerate(e):
            if t_new[(f_i, e_j)] > bestp:
                bestp = t_new[(f_i, e_j)]
                bestj = j
        a[i] = bestj

for (f, e) in bitext:
    for (i, f_i) in enumerate(f):
        sys.stdout.write("%i-%i " % (i, a[i]))
    sys.stdout.write("\n")
