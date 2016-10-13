import sys, optparse, os
from collections import defaultdict

# Add the parent directory into search paths so that we can import perc
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import perc

def global_feature_vector(feat_list, tag_list):
    vec = {}
    feat_index = 0

    for i in range(0, len(tag_list)):
        (feat_index, feats) = perc.feats_for_word(feat_index, feat_list)
        for feat in feats:
            if (feat, tag_list[i]) in vec:
                vec[(feat, tag_list[i])] += 1;
            else:
                vec[(feat, tag_list[i])] = 1;

    return vec

def update_weight_vector(feat_vec, incre_vec, flag):
    for vec in incre_vec:
        feat_vec[vec] += incre_vec[vec] * flag

def perc_train(train_data, tagset, numepochs):
    feat_vec = defaultdict(int)
    default_tag = tagset[0]

    for t in range(numepochs):
        error_num = 0
        for (labeled_list, feat_list) in train_data:
            output = perc.perc_test(feat_vec, labeled_list, feat_list, tagset, default_tag)
            expected = [i.split()[2] for i in labeled_list]
            if output != expected:
                vec_output = global_feature_vector(feat_list, output)
                vec_expected = global_feature_vector(feat_list, expected)
                update_weight_vector(feat_vec, vec_output, -1)
                update_weight_vector(feat_vec, vec_expected, 1)
                error_num += 1
        print "Number of mistakes: ", error_num

    return feat_vec

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(10), help="number of epochs of training; in each epoch we iterate over over all the training examples")
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
