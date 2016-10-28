#!/usr/bin/env python
import optparse, sys, os, logging
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

sys.stderr.write("Training bidirectional IBM Model 1 with one null word and smoothing...\n")
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


def calculate_t(bitext):
    t_fe = defaultdict(float)
    count_fe = defaultdict(float)
    count_e = defaultdict(float)

    # initialize t0 uniformly
    for (f, e) in bitext:
        for f_i in set(f):
            for e_j in set(e).union({null_word}):
                t_fe[(f_i, e_j)] = 1.0 / V_f

    # calculate probabilities
    epoch = 0
    while epoch < opts.max_iters:
        sys.stderr.write("Iteration %d " % (epoch))

        count_fe.clear()
        count_e.clear()
        for (f, e) in bitext:
            for f_i in f:
                # null_word is specially calculated instead of using
                # `e + [null_word]` for better performance
                z = t_fe[(f_i, null_word)]
                for e_j in e:
                    z += t_fe[(f_i, e_j)]
                c = t_fe[(f_i, null_word)] / z
                count_fe[(f_i, null_word)] += c
                count_e[null_word] += c
                for e_j in e:
                    c = t_fe[(f_i, e_j)] / z
                    count_fe[(f_i, e_j)] += c
                    count_e[e_j] += c

        for (f_i, e_j) in count_fe:
            t_fe[(f_i, e_j)] = (count_fe[(f_i, e_j)] + add_n) / (count_e[e_j] + add_n * V_f)

        sys.stderr.write("done\n")
        epoch += 1

    return t_fe


sys.stderr.write("Calculating t_fe...\n")
t_fe = calculate_t(bitext)
# reverse bitext
for i in xrange(len(bitext)):
    tmp = bitext[i][0]
    bitext[i][0] = bitext[i][1]
    bitext[i][1] = tmp
sys.stderr.write("Calculating t_ef...\n")
t_ef = calculate_t(bitext)


def decode(t_fe, f, e, inverse):
    ans = []
    for i, f_i in enumerate(f):
        bestp = t_fe[(f_i, null_word)]
        bestj = -1
        for j, e_j in enumerate(e):
            if t_fe[(f_i, e_j)] > bestp:
                bestp = t_fe[(f_i, e_j)]
                bestj = j
        if bestj >= 0:
            if not inverse:
                ans.append("%i-%i" % (i, bestj))
            else:
                ans.append("%i-%i" % (bestj, i))
    return ans

# decode
sys.stderr.write("Decoding...\n")
# NOTE: bitext has been reversed now! Hence it is (e, f) instead of (f, e)!
for (e, f) in bitext:
    align_fe = decode(t_fe, f, e, False)
    align_ef = decode(t_ef, e, f, True)
    align = set(align_fe).intersection(align_ef)
    print ' '.join(align)
    #print ' '.join(align_fe)
