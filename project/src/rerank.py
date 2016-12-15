#!/usr/bin/env python
import sys
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

    results = []
    for n, nbest in enumerate(nbests):
        if len(nbest) == 0:
            print >> sys.stderr, "WARNING: no translation found for line %d" % (n+1)
            results.append('')
        else:
            results.append(max(nbest, key=lambda x: x.score).english)
    return results
