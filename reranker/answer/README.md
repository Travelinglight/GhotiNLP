# Ghoti Translation Reranker

## Run

Please follow the instructions carefully to run the program as we added custom features:

```bash
# in the reranker/ directory:
python answer/learn.py
python score-reranker.py < output
```

`learn.py` will not only learn the weights from the training set (adding some custom
features) but also do the reranking on the test set and save the output to `output`. The
provided `rerank.py` won't work with our weights because the custom features are not in
`*.nbest` files.

One of the custom features is the IBM Model 1 score (over all alignments). The best result
(the one we submitted to the leaderboard) was achieved by using the alignment model
trained from the data in HW3 (alignment). If the pretrained model (a pickle dump) is not
found, the script will still try to train a model from the training set in this
assignment, which is very likely to yield significantly worse results.

## New features

There are three new features we added:

* IBM Model 1 score (summed over all alignments)
* Number of translated words (a word is estimated to be untranlsated if it has very low
  translation probability or has a direct match in the source French sentence)
* Length of the English sentence

## Algorithm

In addition to the baseline, we averaged the perceptrons across epochs. It doesn't give
much improvement though.

## Note

`test.nbest` and `test.fr` are read by `learn.py`, but ONLY for the purpose of printing
out the reranked output. We NEVER use them to train our model.

## Acknowledgement

* Anoop Sarkar, the course web page
