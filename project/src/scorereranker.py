#!/usr/bin/env python
import bleu


def score(predicted, references):
    scores = []
    for reference in references:
        ref = [line.strip().split() for line in open(reference)]
        system = [line.strip().split() for line in predicted]

        stats = [0 for i in range(10)]
        for (r,s) in zip(ref, system):
          stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(s,r))]

        scores += [bleu.bleu(stats)]

    return max(scores)
