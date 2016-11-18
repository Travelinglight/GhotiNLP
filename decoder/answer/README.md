#Ghoti Phrase Based Decoder
## Algorithm outline
The final version (in decode.py) is based on the baseline beam-search algorithm (Collins), which combined top-k pruning and Viterbi DP. On top of that we added three improvements:

1. We implemented the recombination of indistinguishable hypotheses (in our model, those with the same coverage, same ending words and same last source phrase).

2. We cached the bitstrings and candidate translations for each phrase in the source language, as well as their indexes, so that they can be enumerated efficiently and don't have to be calculated repeatedly each time the phrase is used to update a hypothesis. This improved the performance a lot and enabled us to set the beam to 5000.

3.  We added `future cost` to the criteria in top-k pruning instead of simply using `log probability`. This eliminates the preference on the hypotheses with good log probability but may have very bad future in top-k pruning. <br> <br>
  To calculate the future cost, we first pre-compute a futureCostTable `FCT[i, j]`. `FCT[i, j]` is the highest probability among all the possible translations for words from index i to j in the source language. The words from index i to j could be regarded as a single phrase or multiple phrases. Then for each hypothesis we sum up all `FCT[i, j]` where words from index i to j are not translated.

4.  We added distortion penalty to prevent ridiculous rearrangements in the target language. We used square root of distance to calculate distortion penalty:
$$\alpha^{\sqrt{| Ph_i.s - Ph_{i-1}.t |}}$$
Where $Ph_i.s$ is the start index of the current phrase and  $Ph_{i-1}.t$ is the end index of the previous phrase. We chose $\alpha = 0.95$ at last.


## Other experimented approaches
We first implemented the standard baseline method, and then we tried two more solutions:

1. __Strict aligning method__: only two neighbouring phrases are allowed to be reordered. This algirithm runs fast and produces a slightly better result than the standard baseline beam-search.
2. __A* search__: The future cost function used in our final version doesn't work for A* search because it's not capable of reducing search space enough. We used a simple future cost function:
$$FC = 10 \times |words\ untranslated|$$
In Standard A* search, the search stops once the target is reached. However, in decoding problem there're multiple targets (final hypotheses). So we stopped the search once $2 \times |source\ sentence|$ targets are reached. With the above settings, we got a bad Total corpus log probability: -1644.188934

## Acknowledgement
* Anoop Sarkar, the course web page
* Michael Collins, _Phrase-Based Translation Models_. April 10, 2013
