#!/usr/bin/env python
import sys, optparse

import models
import decode
import rerank
import trainreranker
import scorereranker

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/dev/all.cn-en.cn", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-e", "--eval", dest="eval", default="data/test/all.cn-en.cn", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-r", "--reference", dest="reference", default="data/dev/all.cn-en.en0 data/dev/all.cn-en.en1 data/dev/all.cn-en.en2 data/dev/all.cn-en.en3", help="English reference sentences")
optparser.add_option("--reference-test", dest="referencetest", default="data/test/all.cn-en.en0 data/test/all.cn-en.en1 data/test/all.cn-en.en2 data/test/all.cn-en.en3", help="English reference sentences")
optparser.add_option("-t", "--translation-model-train", dest="tmdev", default="data/large/phrase-table/dev-filtered/rules_cnt.final.out", help="File containing translation model (default=data/tm)")
optparser.add_option("-u", "--translation-model-test", dest="tmtest", default="data/large/phrase-table/test-filtered/rules_cnt.final.out", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm/en.gigaword.3g.filtered.train_dev_test.arpa.gz", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=3, type="int", help="Limit on number of translations to consider per phrase (default=3)")
optparser.add_option("-s", "--stack-size", dest="s", default=100, type="int", help="Maximum stack size (default=100)")
optparser.add_option("-f", "--feedback-loop", dest="loop", default=5, type="int", help="The number of times the weight vector loops between decoder and reranker (default=5)")
optparser.add_option("--simplify", dest="simplify", action="store_true", default=False, help="Simplified mode (default=off)")
optparser.add_option("--resegment-unknown", dest="reseg_unknown", action="store_true", default=False, help="Try to resegment unknown words into two known words (default=off)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Verbose mode (default=off)")

opts = optparser.parse_args()[0]

lm = models.LM(opts.lm)

# Training
weights = [1.0 / 7] * 7
for i in range(opts.loop):
    tm = models.TM(opts.tmdev, opts.k, weights[:4], simpmode=opts.simplify)
    nbest_list = list(decode.get_candidates(opts.input, tm, lm, weights, stack_size=opts.s, verbose=opts.verbose, simpmode=opts.simplify, separate_unknown_words=opts.reseg_unknown))
    weights = trainreranker.train(nbest_list, opts.reference.split(), weights)
    print weights
    results = rerank.rerank(weights, nbest_list)
    print >> sys.stderr, "TRAINING LOOP %d BLEU SCORE: %f:" % \
        (i, scorereranker.score(results, opts.reference.split()))

# Testing
tm = models.TM(opts.tmtest, opts.k, weights[:4], simpmode=opts.simplify)
nbest_list = list(decode.get_candidates(opts.eval, tm, lm, weights, nbest=1, stack_size=100, verbose=opts.verbose, simpmode=opts.simplify, separate_unknown_words=opts.reseg_unknown))
print >> sys.stderr, "TEST BLEU SCORE: %f:" % scorereranker.score(results, opts.referencetest.split())
with open("output1", "w") as f:
    f.write("\n".join(results))
print >> sys.stderr, "Results written to output1"
