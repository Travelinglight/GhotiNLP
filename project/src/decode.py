#!/usr/bin/env python
import optparse
import os
import sys
from mafan import simplify
from collections import namedtuple
from math import log

import models


# Parameter constants
alpha = 0.5  # reordering parameter
max_distance = 10  # maximum distance of reordering
unknown_word_logprob = -100.0  # the logprob of unknown single words
# Features: 0 phi(f|e), 1 lex(f|e), 2 phi(e|f), 3 lex(e|f), 4 lm, 5 distortion
number_of_features_PT = 4  # in phrase table
number_of_features = number_of_features_PT + 2
"""
optparser = optparse.OptionParser()
optparser.add_option("-d", "--dataset", dest="dataset", help="Data set to run on (override other paths): toy, dev, test")
optparser.add_option("-w", "--weights", dest="weights", help="File containing weights for log-linear models")
optparser.add_option("-i", "--input", dest="input", default="data/test/all.cn-en.cn", help="File containing sentences to translate")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/large/phrase-table/test-filtered/rules_cnt.final.out", help="File containing phrase table (translation model)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm/en.gigaword.3g.filtered.train_dev_test.arpa.gz", help="File containing ARPA-format language model")
optparser.add_option("--nbest", dest="nbest", default=1, type="int", help="Number of best translation candidates to print; if larger than 1, will print indexes and scores as well (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Verbose mode (default=off)")
optparser.add_option("-s", "--verbose", dest="verbose", action="store_true", default=False, help="Verbose mode (default=off)")
hyperparam_opts = optparse.OptionGroup(optparser, "Hyperparameters")
hyperparam_opts.add_option("-k", "--translations-per-phrase", dest="k", default=3, type="int", help="Limit on number of translations to consider per phrase (default=3)")
hyperparam_opts.add_option("-s", "--stack-size", dest="s", default=100, type="int", help="Maximum stack size (default=100)")
optparser.add_option_group(hyperparam_opts)
"""

def generate_phrase_cache(f, tm):
  cache = []
  for i in xrange(0, len(f)):
    entries = []
    bitstring = 0
    for j in xrange(i+1, len(f)+1):
      bitstring += 1 << (len(f) - j)
      if tuple(f[i:j]) in tm:
        entries.append({'end': j, 'bitstring': bitstring, 'phrase': tm[tuple(f[i:j])]})
    cache.append(entries)
  return cache


def enumerate_phrases(f_cache, coverage):
  for i in xrange(0, len(f_cache)):
    bitstring = 0
    for entry in f_cache[i]:
      if (entry['bitstring'] & coverage) == 0:
        yield ((i, entry['end']), entry['bitstring'], entry['phrase'])


def precalcuate_future_cost(f, tm, PTweights):
  table = {}  # table[i,j] := [i,j)
  neginf = -float('inf')
  for l in xrange(1, len(f)+1):  # 1 .. len(f)
    for i in xrange(0, len(f)-l+1):  # 0 .. len(f)-l (s.t. i+j <= len(f))
      j = i + l  # 1 .. j-1 (f[i:j])
      if tuple(f[i:j]) in tm:
        scores = [calculate_total_score(p.features, PTweights) for p in tm[tuple(f[i:j])]]
        table[i,j] = max(scores)
      else:
        table[i,j] = neginf
      for k in xrange(1, l): # 1 .. l-1 (skipped when l=1)
        table[i,j] = max(table[i,j], table[i,i+k] + table[i+k,j])
  return table


def get_future_list(bitstring, length):
  bitList = bin(bitstring)[2:].rjust(length, '0')
  futureList = []
  start = 0
  while True:
    pos = bitList.find('1', start)
    if pos == -1:
      break
    if pos > start:
      futureList.append((start, pos))
    start = pos + 1
  if start != length:
    futureList.append((start, length))
  return futureList


def get_future_cost(bitList, futureCostTable):
  cost = 0
  for item in bitList:
    cost = cost + futureCostTable[item]
  return cost


def calculate_total_score(features, weights):
  return sum(p*q for p,q in zip(features, weights))


def extract_english(h):
  return "" if h.predecessor is None else \
    "%s%s " % (extract_english(h.predecessor), h.phrase.english)


def get_candidates(inputfile, tm, lm, weights, stack_size=10, nbest=None, verbose=False, simpmode=True):
  if nbest is None:
    nbest = stack_size

  print >> sys.stderr, "Decoding: " + inputfile
  print >> sys.stderr, "Reading input..."
  french = [list(line.strip().split()) for line in open(inputfile).readlines()]
  if simpmode:
    for li, line in enumerate(french):
      for wi, word in enumerate(line):
        french[li][wi] = simplify(word.decode('utf-8')).encode('utf-8')

  # tm should translate unknown words as-is with a small probability
  # (i.e. only fallback to copying unknown words over as the last resort)
  for word in set(sum(french,[])):
    if (word,) not in tm:
      tm[(word,)] = [models.phrase(word, [unknown_word_logprob] * number_of_features_PT)]

  print >> sys.stderr, "Start decoding..."
  for n, f in enumerate(french):
    if verbose:
      print >> sys.stderr, "Input: " + ' '.join(f)
    # Generate cache for phrase segmentations.
    f_cache = generate_phrase_cache(f, tm)
    # Pre-calculate future cost table
    future_cost_table = precalcuate_future_cost(f, tm, weights[:number_of_features_PT])

    # score = dot(features, weights)
    # features = sums of each log feature
    # predecessor = previous hypothesis
    # lm_state = N-gram state (the last one or two words)
    # last_frange = (i, j) the range of last translated phrase in f
    # phrase = the last TM phrase object (correspondence to f[last_frange])
    # coverage = bit string representing the translation coverage on f
    # future_cost = a safe estimation to be added to total_score
    hypothesis = namedtuple("hypothesis", "score, features, lm_state, predecessor, last_frange, phrase, coverage, future_cost")
    initial_hypothesis = hypothesis(0.0, [0.0] * number_of_features, lm.begin(), None, (0, 0), None, 0, 0)

    # stacks[# of covered words in f] (from 0 to |f|)
    stacks = [{} for _ in xrange(len(f) + 1)]
    # stacks[size][(lm_state, last_frange[1], coverage)]:
    # recombination based on (lm_state, last_frange[1], coverage).
    # For different hypotheses with the same tuple, keep the one with the higher score.
    # lm_state affects LM; last_frange affects distortion; coverage affects available choices.
    stacks[0][(lm.begin(), None, 0)] = initial_hypothesis

    for i, stack in enumerate(stacks[:-1]):
      if verbose:
        print >> sys.stderr, "Stack[%d]:" % i

      # Top-k pruning
      s_hypotheses = sorted(
        stack.values(), key=lambda h: h.score + h.future_cost, reverse=True)
      for h in s_hypotheses[:stack_size]:
        if verbose:
          print >> sys.stderr, h.score, h.lm_state, bin(h.coverage), ' '.join(f[h.last_frange[0]:h.last_frange[1]]), h.future_cost

        for (f_range, delta_coverage, tm_phrases) in enumerate_phrases(f_cache, h.coverage):
          # f_range = (i, j) of the enumerated next phrase to be translated
          # delta_coverage = coverage of f_range
          # tm_phrases = TM entries corresponding to fphrase f[f_range]
          length = i + f_range[1] - f_range[0]
          coverage = h.coverage | delta_coverage
          distance = abs(f_range[0] - h.last_frange[1])
          if distance > 10:
            continue

          # TM might give us multiple candidates for a fphrase.
          for phrase in tm_phrases:
            features = h.features[:]  # copy!
            # Features from phrase table
            for fid in range(number_of_features_PT):
              features[fid] += phrase.features[fid]
            # log_lmprob (N-gram)
            lm_state = h.lm_state
            loglm = 0.0
            for word in phrase.english.split():
              (lm_state, word_logprob) = lm.score(lm_state, word)
              loglm += word_logprob
            # Don't forget the STOP N-gram if we just covered the whole sentence.
            loglm += lm.end(lm_state) if length == len(f) else 0.0
            features[4] += loglm
            # log distortion (distance ** alpha)
            features[5] += log(alpha) * distance

            score = calculate_total_score(features, weights)
            future_list = get_future_list(coverage, len(f))
            future_cost = get_future_cost(future_list, future_cost_table)

            new_state = (lm_state, f_range[1], coverage)
            new_hypothesis = hypothesis(score, features, lm_state, h, f_range, phrase, coverage, future_cost)
            # Recombination
            if new_state not in stacks[length] or \
                score + future_cost > stacks[length][new_state].score + stacks[length][new_state].future_cost:
              stacks[length][new_state] = new_hypothesis

    winners = sorted(stacks[len(f)].values(), key=lambda h: h.score, reverse=True)
    if nbest == 1:
      yield extract_english(winners[0])
    else:
      for s in winners[:nbest]:
        yield ("%d ||| %s |||" + " %f" * number_of_features) % \
          ((n, extract_english(s)) + tuple(s.features))
  print >> sys.stderr, "Decoding completed"


if __name__ == "__main__":
  opts = optparser.parse_args()[0]
  if opts.dataset == "toy":
    opts.input = "data/toy/train.cn"
    opts.lm = "data/lm/en.tiny.3g.arpa"
    opts.tm = "data/toy/phrase-table/phrase_table.out"
  elif opts.dataset == "dev":
    opts.input = "data/dev/all.cn-en.cn"
    opts.lm = "data/lm/en.gigaword.3g.filtered.train_dev_test.arpa.gz"
    opts.tm = "data/large/phrase-table/dev-filtered/rules_cnt.final.out"
  elif opts.dataset == "test":
    opts.input = "data/test/all.cn-en.cn"
    opts.lm = "data/lm/en.gigaword.3g.filtered.train_dev_test.arpa.gz"
    opts.tm = "data/large/phrase-table/test-filtered/rules_cnt.final.out"
  if opts.weights is None:
    weights = [1. / number_of_features] * number_of_features
  else:
    with open(opts.weights) as weights_file:
      weights = [float(line.strip()) for line in weights_file]
      # weights = map(lambda x: 1.0 if math.isnan(x) or x == float("-inf") or x == float("inf") or x == 0.0 else x, w)
      assert len(weights) == number_of_features

  tm = models.TM(opts.tm, opts.k, weights)
  lm = models.LM(opts.lm)

  candidates = get_candidates(opts.input, tm, lm, weights, stack_size=opts.s, nbest=opts.nbest, verbose=opts.verbose)
  for i in candidates:
    print i
