#!/usr/bin/env python
# Simple translation model and language model data structures
import sys
import gzip
import math
from collections import namedtuple

# A translation model is a dictionary where keys are tuples of French words
# and values are lists of (english, features) named tuples. For instance,
# the French phrase "que se est" has two translations, represented like so:
# tm[('que', 'se', 'est')] = [
#   phrase(english='what has', features=[-0.301030009985, ...]), 
#   phrase(english='what has been', features=[-0.301030009985, ...])]
# k is a pruning parameter: only the top k translations are kept for each f.
phrase = namedtuple("phrase", "english, features")
def TM(filename, k, weights, simpmode=False):
  sys.stderr.write("Reading translation model from %s...\n" % (filename,))
  tm = {}
  for line in open(filename).readlines():
    (f, e, features) = line.strip().split(" ||| ")
    tm.setdefault(tuple(f.split()), []).append(
      phrase(e, [float(i) for i in features.strip().split()]))

  tmptm = {}
  for f in tm: # prune all but top k translations
    tm[f].sort(key=lambda x: sum(p*q for p,q in zip(x.features, weights)), reverse=True)
    del tm[f][k:]
    if simpmode:
      from mafan import simplify
      sf = tuple(simplify(f[i].decode('utf-8')) for i in range(len(f)))
      if sf != tuple(f[i].decode('utf-8') for i in range(len(f))):
        if sf in tm:
          for p in tm[f]:
            found = False
            for pi, sp in enumerate(tm[sf]):
              if sp.english == p.english:
                found = True
                tm[sf][pi].features[0] = (tm[sf][pi].features[0] + p.features[0]) / 2
                tm[sf][pi].features[1] = math.log(math.exp(tm[sf][pi].features[1]) + math.exp(p.features[1]))
                tm[sf][pi].features[2] = max([tm[sf][pi].features[2], p.features[2]])
                tm[sf][pi].features[3] = max([tm[sf][pi].features[3], p.features[3]])
            if not found:
              tm[sf].append(phrase(p.english, [
                p.features[0] / 2,
                math.log(math.exp(p.features[1]) + 1),
                p.features[2],
                p.features[3]
              ]))
        else:
          for p in tm[f]:
            tmptm.setdefault(sf, []).append(p)

  for f in tmptm:
    tm[f] = tmptm[f]

  return tm

# # A language model scores sequences of English words, and must account
# # for both beginning and end of each sequence. Example API usage:
# lm = models.LM(filename)
# sentence = "This is a test ."
# lm_state = lm.begin() # initial state is always <s>
# logprob = 0.0
# for word in sentence.split():
#   (lm_state, word_logprob) = lm.score(lm_state, word)
#   logprob += word_logprob
# logprob += lm.end(lm_state) # transition to </s>, can also use lm.score(lm_state, "</s>")[1]
ngram_stats = namedtuple("ngram_stats", "logprob, backoff")
class LM:
  def __init__(self, filename):
    sys.stderr.write("Reading language model from %s...\n" % (filename,))
    self.table = {}
    if filename.endswith('.gz'):
      f = gzip.open(filename)
    else:
      f = open(filename)
    for line in f:
      entry = line.strip().split("\t")
      if len(entry) > 1 and entry[0] != "ngram":
        (logprob, ngram, backoff) = (float(entry[0]), tuple(entry[1].split()), float(entry[2] if len(entry)==3 else 0.0))

        self.table[ngram] = ngram_stats(logprob, backoff)

  def begin(self):
    return ("<s>",)

  def score(self, state, word):
    ngram = state + (word,)
    score = 0.0
    while len(ngram)> 0:
      if ngram in self.table:
        return (ngram[-2:], score + self.table[ngram].logprob)
      else: #backoff
        score += self.table[ngram[:-1]].backoff if len(ngram) > 1 else 0.0
        ngram = ngram[1:]
    # return ((), score + self.table[("<unk>",)].logprob)
    return ((), score)

  def end(self, state):
    return self.score(state, "</s>")[1]
