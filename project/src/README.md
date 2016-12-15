# Ghoti Chinese-English Translator
## Requirements
1. You need python 2.7.
2. You need the library Mafan. Install with pip: `pip install --user mafan` (or you may
   use virtualenv with `requirements.txt`)
3. Enter the home directory of this project (in the same directory as `src`).
4. Put (link) all the data in the `data` directory with the same structure.

## Running direct-decoder translator
```
python src/decode.py -w [weights_file]
```
* w: the file that contains weights. Format: one weight per line, 7 lines in total
* i: the input data

## Running feedback-loop translator
```
python src/feedback-loop -s 20 -k 3 -f 3
```
* s: the stack size
* k: the limit on number of translations to consider per phrase
* f: the number of feedback training loops
* i: the input data
* r: a list of the reference translations
* h: to see the help

The output of test is in the file `output1` by default
