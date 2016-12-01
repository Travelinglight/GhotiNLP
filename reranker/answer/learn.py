#!/usr/bin/env python
import optparse, sys, os
import random
import pickle
import numpy as np
from collections import namedtuple, defaultdict
from math import fabs, log

# Add the parent directory into search paths so that we can import perc
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import bleu


optparser = optparse.OptionParser()
optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("data", "train.nbest"), help="N-best file")
optparser.add_option("-r", "--reference", dest="reference", default=os.path.join("data", "train.en"), help="English reference sentences")
optparser.add_option("-f", "--source", dest="source", default=os.path.join("data", "train.fr"), help="French source sentences")
optparser.add_option("-t", "--tau", dest="tau", default=10000, type="int", help="samples generated from n-best list per input sentence")
optparser.add_option("-a", "--alpha", dest="alpha", default=0.1, type="float", help="sampler acceptance cutoff")
optparser.add_option("-x", "--xi", dest="xi", default=1000, type="int", help="training data generated from the samples tau")
optparser.add_option("-s", "--step", dest="eta", default=0.1, type="float", help="perceptron learning rate")
optparser.add_option("-e", "--epochs", dest="epochs", default=5, type="int", help="number of epochs for perceptron training")
optparser.add_option("-i", "--ibmepochs", dest="ibmepochs", default=5, type="int", help="number of epochs for IBM Model 1")
optparser.add_option("--ibmmodel", dest="t_fe", default=os.path.join("data", "t_fe.pickle"), help="saved t_fe (IBM Model 1)")
# THESE TWO TEST DATA ARE ONLY FOR RERANKING NOT TRAINING!
optparser.add_option("--testnbest", dest="testnbest", default=os.path.join("data", "test.nbest"), help="test N-best file")
optparser.add_option("--testsource", dest="testsource", default=os.path.join("data", "test.fr"), help="test French source sentences")
optparser.add_option("--output", dest="output", default="output", help="test English output after reranking")
(opts, _) = optparser.parse_args()


# a magic string that never appears in the text
null_word = '__NULL__'

# add_n value for smoothing
add_n = 0.01


# Calculate IMB Model 1 coefficients
def calculate_t(bitext):
    # calculate French vocabulary size
    f_voc = set()
    for (f, e) in bitext:
        for f_i in f:
            f_voc.add(f_i)
    v_f = len(f_voc)

    t_fe = defaultdict(float)
    count_fe = defaultdict(float)
    count_e = defaultdict(float)

    # initialize t0 uniformly
    for (f, e) in bitext:
        for f_i in set(f):
            for e_j in set(e).union({null_word}):
                t_fe[(f_i, e_j)] = 1.0 / v_f

    # calculate probabilities
    epoch = 0
    while epoch < opts.ibmepochs:
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
            t_fe[(f_i, e_j)] = (count_fe[(f_i, e_j)] + add_n) / (count_e[e_j] + add_n * v_f)

        sys.stderr.write("done\n")
        epoch += 1

    return t_fe


# get IBM Model 1 score along with other features
def get_more_features(t_fe, f, e):
    untranslated = 0
    score = 0
    for i, f_i in enumerate(f):
        translated = False
        s = 1e-50
        for j, e_j in enumerate(e):
            s += t_fe[(f_i, e_j)]
            if t_fe[(f_i, e_j)] > 0.05:
                translated = True
        score += log(s)
        if not translated or f_i in e:
            untranslated += 1
    return score, len(e), untranslated


# initialization
print >> sys.stderr, "Initializing"
fre = [line.strip().split() for line in open(opts.source)]
ref = [line.strip().split() for line in open(opts.reference)]

if os.path.isfile(opts.t_fe):
    sys.stderr.write("Loading IBM Model 1 from %s... " % opts.t_fe)
    with open(opts.t_fe, 'rb') as f:
        t_fe = pickle.load(f)
    sys.stderr.write("Done.\n")
else:
    print >> sys.stderr, "Calculating IMB Model 1 t_fe from training data..."
    t_fe = calculate_t(zip(fre, ref))

bleu_dump = opts.nbest + '.morefeatures.feats'
candidate = namedtuple("candidate", "sentence, features, bleu, smoothed_bleu")
if os.path.isfile(bleu_dump):
    sys.stderr.write("Loading features from %s... " % bleu_dump)
    with open(bleu_dump, 'rb') as f:
        nbests = pickle.load(f)
    sys.stderr.write("Done.\n")
else:
    print >> sys.stderr, "Calculating all features..."
    nbests = []
    for n, line in enumerate(open(opts.nbest)):
        (i, sentence, features) = line.strip().split("|||")
        i = int(i)
        sentence = sentence.strip()
        words = sentence.split()
        features = [float(h) for h in features.strip().split()]
        more_features = list(get_more_features(t_fe, fre[i], words))
        features = np.array(features + more_features)

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
#w = np.array([1.0/len(nbests[0][0].features)] * len(nbests[0][0].features))
w = np.array([1.0] * len(nbests[0][0].features))
total_w = np.zeros(len(nbests[0][0].features))

# training
random.seed()
for i in range(opts.epochs):
    print >> sys.stderr, "Training epoch %d:" % i
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

        # s[0].bleu - s[1].bleu >= 0; and we want the top (most significant) values.
        sample.sort(key=lambda s: s[0].smoothed_bleu - s[1].smoothed_bleu, reverse=True)
        for (s1, s2) in sample[:opts.xi]:
            if np.dot(w, s1.features) <= np.dot(w, s2.features):
                mistakes += 1
                w += opts.eta * (s1.features - s2.features)  # this is vector addition!

    total_w += w
    print >> sys.stderr, "Number of mistakes: %d" % mistakes

w = total_w / float(opts.epochs)
print("\n".join([str(weight) for weight in w]))

# testing
sys.stderr.write("Reranking test N-best based on learned weights... ")
fre = [line.strip().split() for line in open(opts.testsource)]
translation = namedtuple("translation", "english, score")
nbests = []
for n, line in enumerate(open(opts.testnbest)):
    (i, sentence, features) = line.strip().split("|||")
    i = int(i)
    sentence = sentence.strip()
    words = sentence.split()
    features = [float(h) for h in features.strip().split()]
    more_features = list(get_more_features(t_fe, fre[i], words))
    features = np.array(features + more_features)

    while len(nbests) <= i:
        nbests.append([])

    nbests[i].append(translation(sentence, np.dot(w, features)))

with open(opts.output, 'w') as f:
    for nbest in nbests:
        f.write(max(nbest, key=lambda x: x.score).english + "\n")

sys.stderr.write("Done.\n")
