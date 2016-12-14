#!/usr/bin/env python
from collections import namedtuple

def rerank(w, nbestlist):
    assert w is not None
    translation = namedtuple("translation", "english, score")
    nbests = []
    for line in nbestlist:
        (i, sentence, features) = line.strip().split("|||")
        while len(nbests) <= int(i):
            nbests.append([])
        features = [float(h) for h in features.strip().split()]
        assert len(w) == len(features)
        nbests[int(i)].append(translation(sentence.strip(), sum([x*y for x,y in zip(w, features)])))

    results = [max(nbest, key=lambda x: x.score).english for nbest in nbests]
    return results
