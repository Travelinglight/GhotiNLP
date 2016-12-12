#!/usr/bin/env python
import sys, os
import bleu


def score(predicted, reference):
    ref = [line.strip().split() for line in open(reference)]
    system = [line.strip().split() for line in predicted]

    stats = [0 for i in range(10)]
    for (r,s) in zip(ref, system):
      stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(s,r))]

    return bleu.bleu(stats)
