#!/usr/bin/env python
import optparse, sys, os, logging, math
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-i", "--max_iterations", dest="max_iters", default=5, type="int", help="number of iterations for training")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

bitext = [[sentence.strip().lower().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

# calculate French vocabulary size
f_voc = set()
for (f, e) in bitext:
    for f_i in f:
        f_voc.add(f_i)
V_f = len(f_voc)

# a magic string that never appears in the text
null_word = '__NULL__'

# add_n value for smoothing
add_n = 0.01

t_fe = defaultdict(float)
q_fe = defaultdict(float)
count_fe = defaultdict(float)
count_e = defaultdict(float)
count_ji = defaultdict(float)
count_i = defaultdict(float)

# initialize t0 and q0 uniformly
sys.stderr.write("Initializing t and q...\n")
for (n, (f, e)) in enumerate(bitext):
    for i in range(1, len(f) + 1):
        for j in range(0, len(e) + 1):
            if i == 0:
                e_j = null_word
            else:
                e_j = e[j - 1]

            f_i = f[i - 1]
            t_fe[(f_i, e_j)] = 1.0 / V_f
            q_fe[(j - 1, i - 1, n)] = 1.0 / (len(e) + 1)

# calculate probabilities
sys.stderr.write("Training IBM Model 2 with one null word and smoothing...\n")
epoch = 0
while epoch < opts.max_iters:
    sys.stderr.write("Iteration %d " % (epoch))

    count_fe.clear()
    count_e.clear()
    for (n, (f, e)) in enumerate(bitext):
        for i, f_i in enumerate(f):
            # null_word is specially calculated instead of using
            # `e + [null_word]` for better performance
            z = t_fe[(f_i, null_word)] * q_fe[(0, i, n)]
            for j, e_j in enumerate(e):
                z += t_fe[(f_i, e_j)] * q_fe[(j, i, n)]
            c = t_fe[(f_i, null_word)] * q_fe[(0, i, n)] / z

            count_fe[(f_i, null_word)] += c
            count_e[null_word] += c
            count_ji[(-1, i, n)] += c
            count_i[(i, n)] += c
            for e_j in e:
                c = t_fe[(f_i, e_j)] * q_fe[(j, i, n)] / z
                count_fe[(f_i, e_j)] += c
                count_e[e_j] += c
                count_ji[(j, i, n)] += c
                count_i[(i, n)] += c

    for (f_i, e_j) in count_fe:
        t_fe[(f_i, e_j)] = (count_fe[(f_i, e_j)] + add_n) / (count_e[e_j] + add_n * V_f)
    for (j, i, n) in count_ji:
        q_fe[(j, i, n)] = (count_ji[(j, i, n)] + add_n) / (count_i[(i, n)] + add_n * (len(bitext[n][1]) + 1))

    L = 0.0
    for (n, (f, e)) in enumerate(bitext):
        Pr = 1.0
        for i, f_i in enumerate(f):
            pr = t_fe[(f_i, null_word)] * q_fe[(-1, i, n)]
            for j, e_j in enumerate(e):
                pr += t_fe[f_i, e_j] * q_fe[(j, i, n)]

            Pr *= pr

        L += math.log(Pr)

    sys.stderr.write("Iteration %d, L=%f\n" % (epoch, L))
    sys.stderr.write("done\n")
    epoch += 1

# decode
sys.stderr.write("Decoding...\n")
for (n, (f, e)) in enumerate(bitext):
    for i, f_i in enumerate(f):
        bestp = t_fe[(f_i, null_word)] * q_fe[(-1, i, n)]
        bestj = -1
        for j, e_j in enumerate(e):
            if t_fe[(f_i, e_j)] * q_fe[(j, i, n)] > bestp:
                bestp = t_fe[(f_i, e_j)] * q_fe[(j, i, n)]
                bestj = j
        if bestj >= 0:
            sys.stdout.write("%i-%i " % (i, bestj))
    sys.stdout.write("\n")
