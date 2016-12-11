#!/usr/bin/env python
import sys, os, math
from collections import namedtuple

def rerank(weights, nbestlist):
    w = None
    if weights is not None:
        weights_file = sys.stdin if weights is "-" else open(weights)
        w = [float(line.strip()) for line in weights_file]
        w = map(lambda x: 1.0 if math.isnan(x) or x == float("-inf") or x == float("inf") or x == 0.0 else x, w)
        w = None if len(w) == 0 else w

    translation = namedtuple("translation", "english, score")
    nbests = []
    for line in nbestlist:
        (i, sentence, features) = line.strip().split("|||")
        if len(nbests) <= int(i):
            nbests.append([])
        features = [float(h) for h in features.strip().split()]
        if w is None or len(w) != len(features):
            w = [1.0/len(features) for _ in xrange(len(features))]
        nbests[int(i)].append(translation(sentence.strip(), sum([x*y for x,y in zip(w, features)])))

    results = [sorted(nbest, key=lambda x: -x.score)[0].english for nbest in nbests]
    return results
