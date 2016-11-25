#!/usr/bin/env python
import optparse, sys, os
import bleu
import random
import numpy as np
from collections import namedtuple
from math import fabs

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
print >> sys.stderr, "Initializing..."
ref = [line.strip().split() for line in open(opts.reference)]
nbests = []
candidate = namedtuple("candidate", "sentence, features, bleu, smoothed_bleu")
for line in open(opts.nbest):
    (i, sentence, features) = line.strip().split("|||")
    features = np.array([float(h) for h in features.strip().split()])
    i = int(i)

    # calculate bleu score and smoothed bleu score
    stats = [0 for k in range(10)]
    stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(sentence.strip().split(), ref[i]))]
    bleu_score = bleu.bleu(stats)
    smoothed_bleu_score = bleu.smoothed_bleu(stats)

    if i >= len(nbests):
        nbests.append([])

    nbests[i].append(candidate(sentence, features, bleu_score, smoothed_bleu_score))

# set weights to default
for line in open(opts.nbest):
    (i, sentence, features) = line.strip().split("|||")
    features = np.array([float(h) for h in features.strip().split()])
    w = np.array([1.0/len(features) for k in range(len(features))])
    break

# training
for i in range(opts.epochs):
    print >> sys.stderr, "Training: epoch[%d]:" % i
    delta = np.array([0.0 for k in range(len(w))])
    mistakes = 0
    for nbest in nbests:
        sample = []
        for j in range(opts.tau):
            if len(nbest) < 2:
                break

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

    print >> sys.stderr, "Number of mistakes: [%d]:" % mistakes
    w += delta

print("\n".join([str(weight) for weight in w]))
