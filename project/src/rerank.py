#!/usr/bin/env python
import sys, os, math
from collections import namedtuple

def rerank(w, nbestlist):
    translation = namedtuple("translation", "english, score")
    nbests = []
    for line in nbestlist:
        (i, sentence, features) = line.strip().split("|||")
        while len(nbests) <= int(i):
            nbests.append([])
        features = [float(h) for h in features.strip().split()]
        if w is None or len(w) != len(features):
            w = [1.0/len(features) for _ in range(len(features))]
        nbests[int(i)].append(translation(sentence.strip(), sum([x*y for x,y in zip(w, features)])))

    results = [sorted(nbest, key=lambda x: -x.score)[0].english for nbest in nbests]
    return results
