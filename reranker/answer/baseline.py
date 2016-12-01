#!/usr/bin/env python
import optparse, sys, os
import random
import pickle
import numpy as np
from collections import namedtuple
from math import fabs

# Add the parent directory into search paths so that we can import perc
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import bleu


optparser = optparse.OptionParser()
optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("data", "train.nbest"), help="N-best file")
optparser.add_option("-r", "--reference", dest="reference", default=os.path.join("data", "train.en"), help="English reference sentences")
optparser.add_option("-t", "--tau", dest="tau", default=5000, help="samples generated from n-best list per input sentence")
optparser.add_option("-a", "--alpha", dest="alpha", default=0.1, help="sampler acceptance cutoff")
optparser.add_option("-x", "--xi", dest="xi", default=100, help="training data generated from the samples tau")
optparser.add_option("-s", "--step", dest="eta", default=0.1, help="perceptron learning rate")
optparser.add_option("-e", "--epochs", dest="epochs", default=5, help="number of epochs for perceptron training")
(opts, _) = optparser.parse_args()


# initialization
print >> sys.stderr, "Initializing training data"
bleu_dump = opts.nbest + '.baseline.feats'
candidate = namedtuple("candidate", "sentence, features, bleu, smoothed_bleu")
if os.path.isfile(bleu_dump):
    sys.stderr.write("Loading features from %s... " % bleu_dump)
    with open(bleu_dump, 'rb') as f:
        nbests = pickle.load(f)
    sys.stderr.write("Done.\n")
else:
    ref = [line.strip().split() for line in open(opts.reference)]
    nbests = []
    for n, line in enumerate(open(opts.nbest)):
        (i, sentence, features) = line.strip().split("|||")
        i = int(i)
        sentence = sentence.strip()
        features = np.array([float(h) for h in features.strip().split()])

        # calculate bleu score and smoothed bleu score
        stats = tuple(bleu.bleu_stats(sentence.split(), ref[i]))
        bleu_score = bleu.bleu(stats)
        smoothed_bleu_score = bleu.smoothed_bleu(stats)

        while len(nbests) <= i:
            nbests.append([])
        nbests[i].append(candidate(sentence, features, bleu_score, smoothed_bleu_score))

        if n % 2000 == 0:
            sys.stderr.write(".")
    sys.stderr.write("\nSaving features to %s... " % bleu_dump)
    with open(bleu_dump, 'wb') as f:
        pickle.dump(nbests, f)
    sys.stderr.write("Done.\n")

# set weights to default
w = np.array([1.0/len(nbests[0][0].features)] * len(nbests[0][0].features))

# training
random.seed()
for i in range(opts.epochs):
    print >> sys.stderr, "Training epoch %d:" % i
    delta = np.zeros(len(w))
    mistakes = 0
    for nbest in nbests:
        if len(nbest) < 2:
            continue

        sample = []
        for j in xrange(opts.tau):
            (s1, s2) = (nbest[k] for k in random.sample(range(len(nbest)), 2))
            if fabs(s1.smoothed_bleu - s2.smoothed_bleu) > opts.alpha:
                if s1.smoothed_bleu > s2.smoothed_bleu:
                    sample.append((s1, s2))
                else:
                    sample.append((s2, s1))
            else:
                continue

        for (s1, s2) in sorted(sample, key=lambda s: s[0].smoothed_bleu - s[1].smoothed_bleu)[:opts.xi]:
            if np.dot(w, s1.features) <= np.dot(w, s2.features):
                mistakes += 1
                delta += opts.eta * (s1.features - s2.features)  # this is vector addition!

    print >> sys.stderr, "Number of mistakes: %d" % mistakes
    w += delta

print("\n".join([str(weight) for weight in w]))
