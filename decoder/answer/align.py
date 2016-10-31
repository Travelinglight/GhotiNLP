#!/usr/bin/env python
import optparse, sys, os, logging, math
from collections import defaultdict
from operator import itemgetter

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-i", "--iterations", dest="iterations", default=5, type="int", help="number of iterations")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

sys.stderr.write("Training IBM Model 1 (no nulls) with Expectation Maximization...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

prob = defaultdict(float)
for t in xrange(opts.iterations):
  sys.stderr.write("\nIteration %d..." % t)
  fe_count = defaultdict(float)
  likelihood = 0.0
  for n, (f, e) in enumerate(bitext):
    for f_i in f:
      sum_i = sum(prob[f_i,e_j] for e_j in e)
      likelihood += sum_i
      for e_j in e:
        fe_count[f_i,e_j] += 1.0/len(e) if t==0 else prob[f_i,e_j]/sum_i
    if n % 500 == 0:
      sys.stderr.write(".")
  z = defaultdict(float)
  for f_i, e_j in fe_count:
    z[e_j] += fe_count[f_i, e_j]
  for f_i, e_j in fe_count:
    prob[f_i, e_j] = fe_count[f_i, e_j] / z[e_j]
  if opts.logfile:
    logging.info("Iteration: %d" % t)
    for (f,e) in sorted(prob.keys(), key=itemgetter(1)):
        logging.info("t(%s | %s) = %f" % (f, e, prob[f,e]))
    logging.info("L = %f" % math.log10(likelihood) if likelihood > 0.0 else float("-inf"))

sys.stderr.write("\nAligning...")
for f, e in bitext:
  for i, f_i in enumerate(f): 
    j = max(((prob[f_i,e_j],j) for j, e_j in enumerate(e)), key=lambda x: x[0])[1]
    sys.stdout.write("%i-%i " % (i,j))
  sys.stdout.write("\n")
sys.stderr.write("\n")
