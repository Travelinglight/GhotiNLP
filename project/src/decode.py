#!/usr/bin/env python
import optparse
import os
import sys
from collections import namedtuple
from math import log, sqrt

import models


# Parameter constants
alpha = 0.5  # reordering parameter
unknown_word_logprob = -100.0  # the logprob of unknown single words

optparser = optparse.OptionParser()
optparser.add_option("-d", "--dataset", dest="dataset", help="Data set to run on (override other paths): toy, dev, test")
optparser.add_option("-i", "--input", dest="input", default="data/test/all.cn-en.cn", help="File containing sentences to translate")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/large/phrase-table/test-filtered/rules_cnt.final.out", help="File containing phrase table (translation model)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm/en.gigaword.3g.filtered.train_dev_test.arpa.gz", help="File containing ARPA-format language model")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("--nbest", dest="nbest", default=1, type="int", help="Number of best translation candidates to print; if larger than 1, will print indexes and scores as well (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Verbose mode (default=off)")
hyperparam_opts = optparse.OptionGroup(optparser, "Hyperparameters")
hyperparam_opts.add_option("-k", "--translations-per-phrase", dest="k", default=3, type="int", help="Limit on number of translations to consider per phrase (default=3)")
hyperparam_opts.add_option("-s", "--stack-size", dest="s", default=100, type="int", help="Maximum stack size (default=100)")
optparser.add_option_group(hyperparam_opts)


def generate_phrase_cache(f):
  cache = []
  for i in xrange(0, len(f)):
    entries = []
    bitstring = 0
    for j in xrange(i+1, len(f)+1):
      bitstring += 1 << (len(f) - j)
      if f[i:j] in tm:
        entries.append({'end': j, 'bitstring': bitstring, 'phrase': tm[f[i:j]]})
    cache.append(entries)
  return cache


def enumerate_phrases(f_cache, coverage):
  for i in xrange(0, len(f_cache)):
    bitstring = 0
    for entry in f_cache[i]:
      if (entry['bitstring'] & coverage) == 0:
        yield ((i, entry['end']), entry['bitstring'], entry['phrase'])


def precalcuate_future_cost(f):
  table = {}  # table[i,j] := [i,j)
  neginf = -float('inf')
  for l in xrange(1, len(f)+1):  # 1 .. len(f)
    for i in xrange(0, len(f)-l+1):  # 0 .. len(f)-l (s.t. i+j <= len(f))
      j = i + l  # 1 .. j-1 (f[i:j])
      if f[i:j] in tm:
        maxph = max(tm[f[i:j]], key=lambda x: x.logprob)
        table[i,j] = maxph.logprob
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
  return futureList


def get_future_cost(bitList, futureCostTable):
  cost = 0
  for item in bitList:
    cost = cost + futureCostTable[item]
  return cost


def get_candidates(french, tm, lm, weights, stack_size=1, nbest=None, verbose=False):
  if nbest is None:
    nbest = stack_size
  sys.stderr.write("Decoding %s...\n" % (opts.input,))
  for n, f in enumerate(french):
    if opts.verbose:
      print >> sys.stderr, "Input: " + f
    # Generate cache for phrase segmentations.
    f_cache = generate_phrase_cache(f)
    # Pre-calculate future cost table
    future_cost_table = precalcuate_future_cost(f)

    # logprob = log_lmprob + log_tmprob + distortion_penalty
    # predecessor = previous hypothesis
    # lm_state = N-gram state (the last one or two words)
    # last_frange = (i, j) the range of last translated phrase in f
    # phrase = the last TM phrase object (correspondence to f[last_frange])
    # coverage = bit string representing the translation coverage on f
    # future_cost
    hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, last_frange, phrase, coverage, future_cost")
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, (0, 0), None, 0, 0)
    # stacks[# of covered words in f] (from 0 to |f|)
    stacks = [{} for _ in xrange(len(f) + 1)]
    # stacks[size][(lm_state, last_frange, coverage)]:
    # recombination based on (lm_state, last_frange, coverage).
    # For different hypotheses with the same tuple, keep the one with the higher logprob.
    # lm_state affects LM; last_frange affects distortion; coverage affects available choices.
    stacks[0][(lm.begin(), None, 0)] = initial_hypothesis
    for i, stack in enumerate(stacks[:-1]):
      if opts.verbose:
        print >> sys.stderr, "Stack[%d]:" % i

      # Top-k pruning
      s_hypotheses = sorted(
        stack.values(), key=lambda h: h.logprob + h.future_cost, reverse=True)
      for h in s_hypotheses[:opts.s]:
        if verbose:
          print >> sys.stderr, h.logprob, h.lm_state, bin(h.coverage), unicode(' '.join(f[h.last_frange[0]:h.last_frange[1]]), 'utf8'), h.future_cost

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
            logprob = h.logprob + phrase.logprob + log(alpha)*sqrt(abs(distance))
            # log_lmprob (N-gram)
            lm_state = h.lm_state
            for word in phrase.english.split():
              (lm_state, word_logprob) = lm.score(lm_state, word)
              logprob += word_logprob
            # Don't forget the STOP N-gram if we just covered the whole sentence.
            logprob += lm.end(lm_state) if length == len(f) else 0.0

            # Future cost.
            future_list = get_future_list(coverage, len(f))
            future_cost = get_future_cost(future_list, future_cost_table)

            new_state = (lm_state, f_range, coverage)
            new_hypothesis = hypothesis(logprob, lm_state, h, f_range, phrase, coverage, future_cost)
            if new_state not in stacks[length] or \
                logprob + future_cost > stacks[length][new_state].logprob + stacks[length][new_state].future_cost:  # recombination
              stacks[length][new_state] = new_hypothesis

    winners = sorted(stacks[len(f)].values(), key=lambda h: h.logprob, reverse=True)

    def extract_english(h):
      return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

    if nbest == 1:
      yield extract_english(winners[0])
    else:
      for s in winners[:nbest]:
        yield "%d ||| %s ||| %f" % \
          (n, extract_english(s), s.logprob)


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

  tm = models.TM(opts.tm, opts.k)
  lm = models.LM(opts.lm)
  french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

  # tm should translate unknown words as-is with a small probability
  # (i.e. only fallback to copying unknown words over as the last resort)
  for word in set(sum(french,())):
    if (word,) not in tm:
      tm[(word,)] = [models.phrase(word, unknown_word_logprob)]

  candidates = get_candidates(french, tm, lm, 1, stack_size=opts.s, nbest=opts.nbest, verbose=opts.verbose)
  for i in candidates:
    print i
