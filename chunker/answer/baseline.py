"""

You have to write the perc_train function that trains the feature weights using the perceptron algorithm for the CoNLL 2000 chunking task.

Each element of train_data is a (labeled_list, feat_list) pair. 

Inside the perceptron training loop:

    - Call perc_test to get the tagging based on the current feat_vec and compare it with the true output from the labeled_list

    - If the output is incorrect then we have to update feat_vec (the weight vector)

    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)

    - Instead of indexing each feature with an integer we index each feature using a string we called feature_id

    - The feature_id is constructed using the elements of feat_list (which correspond to x above) combined with the output tag (which correspond to y above)

    - The function perc_test shows how the feature_id is constructed for each word in the input, including the bigram feature "B:" which is a special case

    - feat_vec[feature_id] is the weight associated with feature_id

    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id not used in a particular example does not participate in the dot product

    - To save space and time make sure you do not store zero values in the feat_vec dictionary which can happen if \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value

    - If you are going word by word to check if the predicted tag is equal to the true tag, there is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.

"""

import sys, optparse, os

# Add the parent directory into search paths so that we can import perc
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import perc

if os.path.abspath(".") not in sys.path:
    sys.path.insert(0, os.path.abspath("."))
if os.path.abspath("..") not in sys.path:
    sys.path.insert(0, os.path.abspath(".."))

import perc
from collections import defaultdict

def get_global_vector(labels, feat_list):
    """
    :param labels: a list of chuncking label, e.g. ['B-NP', ...]
    :param feat_list: a list of features on words and POS tags, e.g. ['U12:VBDq', ...]
    :return: a dict of features with chuncking labels, e.g. {('U12:VBDq'): 1, ...}
    """
    global_vector = defaultdict(int)

    index = -1
    prev_tag = 'B_-1'
    for i in range(len(feat_list)):
        feat_value = feat_list[i]
        # check if we reached the feats for the next word
        if feat_value[:4] == 'U00:':
            index += 1
            if (index > 0):
                prev_tag = labels[index - 1]

        if feat_value[0] == 'B':
            feat_value = 'B:' + prev_tag

        global_vector[(feat_value, labels[index])] += 1

    return global_vector


def add_vector(a, b, k):
    for key in b:
        a[key] += b[key] * k
        if a[key] == 0:
            del a[key]
    return a


def get_labels(labeled_list):
    """
    get chuncking labels from a labeled list
    """
    labels = []
    for i in range(len(labeled_list)):
        labeled_list_value = labeled_list[i]
        labels.append(labeled_list_value.split()[2])

    return labels


def perc_train(train_data, tagset, numepochs):
    """
    :current_global_vector: a dict of features for the predicted labels
    :gold_global_vector: a dict of features for the standard
    """
    feat_vec = defaultdict(int)
    #for t in range(numepochs):
    default_tag = tagset[0]
    for t in range(numepochs):
        for (labeled_list, feat_list) in train_data:
            std_labels = get_labels(labeled_list)
            output = perc.perc_test(feat_vec, labeled_list, feat_list, tagset, default_tag)
            gold_global_vector = get_global_vector(std_labels, feat_list)
            current_global_vector = get_global_vector(output, feat_list)
            add_vector(feat_vec, gold_global_vector, 1)
            add_vector(feat_vec, current_global_vector, -1)

        perc.perc_write_to_file(feat_vec, opts.modelfile + str(t))

    return feat_vec


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(4), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    tagset = perc.read_tagset(opts.tagsetfile)
    print >>sys.stderr, "reading data ..."
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile)
    print >>sys.stderr, "done."
    feat_vec = perc_train(train_data, tagset, int(opts.numepochs))
    perc.perc_write_to_file(feat_vec, opts.modelfile)

