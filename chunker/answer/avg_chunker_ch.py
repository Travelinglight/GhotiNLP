import perc
import sys, optparse, os
from collections import defaultdict

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

def perc_train(train_data, tagset, numepochs):
    feat_vec = defaultdict(int)
    sigma = defaultdict(float)

    default_tag = tagset[0]
    tau = {}

    for t in range(numepochs):
        error_num = 0
        T = numepochs - 1
        for i in range(len(train_data)):
            m = len(train_data) - 1
            (labeled_list, feat_list) = train_data[i]
            output = perc.perc_test(feat_vec, labeled_list, feat_list, tagset, default_tag)
            expected = [exp.split()[2] for exp in labeled_list]
            if ((t != T) or (i != m)):     
                if output != expected:
                    vec_output = global_feature_vector(feat_list, output)
                    vec_expected = global_feature_vector(feat_list, expected)
                    for vec in vec_output:
                        vec_expected[vec] = vec_expected.get(vec, 0) - vec_output[vec]
                    for vec in vec_expected:
                        if vec in tau:
                            sigma[vec] = sigma.get(vec, 0) + feat_vec[vec] * (t * m + i - tau[vec][1] * m - tau[vec][0])
                        feat_vec[vec] += vec_expected[vec]
                        sigma[vec] += vec_expected[vec] 
                        tau[vec] = (i, t)
                    error_num += 1
            else:    
                for vec in tau:
                    sigma[vec] = sigma.get(vec, 0) + feat_vec[vec] * (T * m + m - tau[vec][1] * m - tau[vec][0])
                if output != expected:
                    vec_output = global_feature_vector(feat_list, output)
                    vec_expected = global_feature_vector(feat_list, expected)
                    for vec in vec_output:
                        vec_expected[vec] = vec_expected.get(vec, 0) - vec_output[vec]
                    for vec in vec_expected:
                        if vec in tau:
                            sigma[vec] = sigma.get(vec, 0) + feat_vec[vec] * (t * m + i - tau[vec][1] * m - tau[vec][0])
                        feat_vec[vec] += vec_expected[vec]
                        sigma[vec] += vec_expected[vec] 
                    error_num += 1
            print error_num
        print "Number of mistakes: ", error_num
    sigma = {k: v / (numepochs * len(train_data)) for (k, v) in sigma.items()}

    return sigma

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(5), help="number of epochs of training; in each epoch we iterate over over all the training examples")
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