import sys, optparse, os
from collections import defaultdict

# Add the parent directory into search paths so that we can import perc
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import perc

def global_feature_vector(feat_list, tag_list):
    global_vector = defaultdict(int)

    index = -1
    prev_tag = 'B_-1'
    for i in range(len(feat_list)):
        feat_value = feat_list[i]
        # check if we reached the feats for the next word
        if feat_value[:4] == 'U00:':
            index += 1
            if (index > 0):
                prev_tag = tag_list[index - 1]

        if feat_value[0] == 'B':
            feat_value = 'B:' + prev_tag

        global_vector[(feat_value, tag_list[index])] += 1

    return global_vector

def add_vector(a, b, k):
    for key in b:
        a[key] += b[key] * k
        if a[key] == 0:
            del a[key]
    return a

def perc_train(train_data, tagset, numepochs):
    feat_vec = defaultdict(int)
    sigma = defaultdict(float)
    default_tag = tagset[0]
    tau = {}
    T = numepochs - 1
    m = len(train_data) - 1
    
    for t in range(numepochs):
        error_num = 0
        for i in range(len(train_data)):         
            (labeled_list, feat_list) = train_data[i]
            output = perc.perc_test(feat_vec, labeled_list, feat_list, tagset, default_tag)
            expected = [exp.split()[2] for exp in labeled_list]

            # If not the last sentence or not in the last iteration
            if ((t != T) or (i != m)):     
                if output != expected:
                    vec_output = global_feature_vector(feat_list, output)
                    vec_expected = global_feature_vector(feat_list, expected)
                    
                    # Calculate difference between output and expcted result    
                    add_vector(vec_expected, vec_output, -1)

                    # include the weight calculated from comparing output and expcted result
                    for feat in vec_expected:
                        
                        # Include the total weight during the time
                        if feat in tau:
                            sigma[feat] += feat_vec[feat] * \
                                (t * m + i - tau[feat][1] * m - tau[feat][0])
                        feat_vec[feat] += vec_expected[feat]
                        sigma[feat] += vec_expected[feat]

                        # Record the location where the dimension s is updated
                        tau[feat] = (i, t)

                    error_num += 1

            # To deal with the last sentence in the last iteration
            else:    
                # Include the total weight during the time
                for feat in tau:
                    sigma[feat] += feat_vec[feat] * \
                        (T * m + m - tau[feat][1] * m - tau[feat][0])

                if output != expected:
                    vec_output = global_feature_vector(feat_list, output)
                    vec_expected = global_feature_vector(feat_list, expected)

                    # Calculate difference between output and expcted result    
                    add_vector(vec_expected, vec_output, -1)

                    # include the weight calculated from comparing output and expcted result
                    add_vector(feat_vec, vec_expected, 1)
                    add_vector(sigma, vec_expected, 1)

                    error_num += 1

        print >>sys.stderr, "Epoch", t + 1, "done. # of incorrect sentences: ", error_num

    return {k: float(sigma[k]) / (numepochs * len(train_data)) for k in sigma}

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(6), help="number of epochs of training; in each epoch we iterate over over all the training examples")
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
