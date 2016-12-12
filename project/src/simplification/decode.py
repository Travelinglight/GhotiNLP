#!/usr/bin/env python
import os
import sys
from collections import namedtuple 
from math import log, sqrt
from operator import add

# Add the parent directory into search paths so that we can import perc
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import models


def get_candidates(input, tm, lm, weights, s=1):

    alpha = 0.95  #reordering parameter
    french = [tuple(line.strip().split()) for line in open(input).readlines()]

    # tm should translate unknown words as-is with probability 1
    for word in set(sum(french,())):
        if (word,) not in tm:
            tm[(word,)] = [models.phrase(word, [0.0, 0.0, 0.0, 0.0])]


    def generate_phrase_cache(f):
        cache = []
        for i in range(0, len(f)):
            entries = []
            bitstring = 0
            for j in range(i+1, len(f)+1):
                bitstring += 1 << (len(f) - j)
                if f[i:j] in tm:
                    entries.append({'end': j, 'bitstring': bitstring, 'phrase': tm[f[i:j]]})
            cache.append(entries)
        return cache


    def enumerate_phrases(f_cache, coverage):
        for i in range(0, len(f_cache)):
            bitstring = 0
            for entry in f_cache[i]:
                if (entry['bitstring'] & coverage) == 0:
                    yield ((i, entry['end']), entry['bitstring'], entry['phrase'])


    def precalcuate_future_cost(f):
        phraseCheapestTable = {}
        futureCostTable = {}
        for i in range(0,len(f)):
            for j in range(i+1,len(f)+1):
                if f[i:j] in tm:
                    phraseCheapestTable[i,j] = -sys.maxint
                    for phrase in tm[f[i:j]]:
                        if phrase.logprob > phraseCheapestTable[i,j]:
                            phraseCheapestTable[i,j] = phrase.logprob
        for i in range(0,len(f)):
            futureCostTable[i,1] = phraseCheapestTable[i,i+1]
            for j in range(2,len(f)+1-i):
                if (i,i+j) in  phraseCheapestTable:
                    futureCostTable[i,j] = phraseCheapestTable[i,i+j]
                else:
                    futureCostTable[i,j] = -sys.maxint
                for k in range(1, j):
                    if(((i+k,i+j) in phraseCheapestTable) and (futureCostTable[i,j] < futureCostTable[i,k] + phraseCheapestTable[i+k,i+j])):
                        futureCostTable[i,j] = futureCostTable[i,k] + phraseCheapestTable[i+k,i+j]
        return futureCostTable


    def get_future_list(bitstring):
        bitList = bin(bitstring)[2:]
        futureList = []
        count = 0
        index = 0
        findZeroBit = False
        for i in range(len(bitList)):
            if bitList[i] == '0':
                if not findZeroBit:
                    index = i
                findZeroBit = True
                count = count + 1
            else:
                if findZeroBit:
                    futureList.append((index, count))
                findZeroBit = False
                count = 0
        if findZeroBit:
            futureList.append((index, count))
        return futureList


    def get_future_cost(bitList, futureCostTable):
        cost = 0
        for item in bitList:
            cost = cost + futureCostTable[item]
        return cost


    def extract_english(h):
        return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)


    results = []
    sys.stderr.write("Decoding %s...\n" % (input,))
    for n, f in enumerate(french):
        # Generate cache for phrase segmentations.
        f_cache = generate_phrase_cache(f)
        # Pre-calculate future cost table
        #future_cost_table = precalcuate_future_cost(f)

        # logprob = log_lmprob + log_tmprob + distortion_penalty
        # predecessor = previous hypothesis
        # lm_state = N-gram state (the last one or two words)
        # last_frange = (i, j) the range of last translated phrase in f
        # phrase = the last TM phrase object (correspondence to f[last_frange])
        # coverage = bit string representing the translation coverage on f
        # future_cost
        hypothesis = namedtuple("hypothesis", "logprob, features, lm_score, lm_state, predecessor, last_frange, phrase, coverage")
        initial_hypothesis = hypothesis(0.0, [0.0, 0.0, 0.0, 0.0], 0.0, lm.begin(), None, (0, 0), None, 0)
        # stacks[# of covered words in f] (from 0 to |f|)
        stacks = [{} for _ in range(len(f) + 1)]
        # stacks[size][(lm_state, last_frange, coverage)]:
        # recombination based on (lm_state, last_frange, coverage).
        # For different hypotheses with the same tuple, keep the one with the higher logprob.
        # lm_state affects LM; last_frange affects distortion; coverage affects available choices.
        stacks[0][(lm.begin(), None, 0)] = initial_hypothesis
        for i, stack in enumerate(stacks[:-1]):

            # Top-k pruning
            for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:s]:
                for (f_range, delta_coverage, tm_phrases) in enumerate_phrases(f_cache, h.coverage):
                    # f_range = (i, j) of the enumerated next phrase to be translated
                    # delta_coverage = coverage of f_range
                    # tm_phrases = TM entries corresponding to fphrase f[f_range]
                    length = i + f_range[1] - f_range[0]
                    coverage = h.coverage | delta_coverage
                    distance = f_range[0] - h.last_frange[1]

                    # TM might give us multiple candidates for a fphrase.
                    for phrase in tm_phrases:
                        # log_tmprob and distortion
                        features = map(add, h.features, phrase.features)
                        # log_lmprob (N-gram)
                        lm_state = h.lm_state
                        lm_score = h.lm_score
                        for word in phrase.english.split():
                            (lm_state, word_logprob) = lm.score(lm_state, word)
                            lm_score += word_logprob
                        # Don't forget the STOP N-gram if we just covered the whole sentence.
                        lm_score += lm.end(lm_state) if length == len(f) else 0.0

                        # Future cost.
                        #future_list = get_future_list(delta_coverage)
                        #future_cost = get_future_cost(future_list, future_cost_table)

                        logprob = sum(p*q for p, q in zip((features + [lm_score]), weights))
                        new_state = (lm_state, f_range, coverage)
                        new_hypothesis = hypothesis(logprob, features, lm_score, lm_state, h, f_range, phrase, coverage)
                        if new_state not in stacks[length] or \
                            logprob > stacks[length][new_state].logprob:  # recombination
                            stacks[length][new_state] = new_hypothesis

        winner = sorted(stacks[len(f)].itervalues(), key=lambda h: h.logprob, reverse=True)[0:100]
        for i in range(len(winner)):
            results += ["%d ||| %s ||| %f %f %f %f %f" % (n, extract_english(winner[i]), winner[i].features[0],
                winner[i].features[1], winner[i].features[2], winner[i].features[3], winner[i].lm_score)]

    return results
