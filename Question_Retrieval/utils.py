import gzip
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.autograd import Variable
import torch.nn.functional as f

def normalize_3d(x, eps=1e-8):
    #TODO
    # x is len*batch*d
    # l2 is len*batch*1
    #l2 = x.norm(2,axis=2).dimshuffle((0,1,'x'))
    l2 = torch.norm(x, 2).dimshuffle((0,1,'x'))#??
    return x/(l2+eps)

def average_without_padding(x, ids, padding_id, eps=1e-8):
    x = f.normalize(x, p=2, dim=2)
    mask = np.not_equal(ids, padding_id).astype('float')
    # len*batch*1
    mask = Variable(torch.from_numpy(mask)).float().view(ids.shape[0],ids.shape[1],1)
    result = torch.sum(x*mask,dim=0)/ (torch.sum(mask,dim=0)+eps)
    #print 'result shape', result.data.shape
    return result 

def create_idf_weights(corpus_path, embedding_layer):
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,1), binary=False)

    lst = [ ]
    fopen = gzip.open if corpus_path.endswith(".gz") else open
    with fopen(corpus_path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            lst.append(title)
            lst.append(body)
    vectorizer.fit_transform(lst)

    idfs = vectorizer.idf_
    avg_idf = sum(idfs)/(len(idfs)+0.0)/4.0
    weights = np.array([ avg_idf for i in xrange(embedding_layer.n_V) ])
    vocab_map = embedding_layer.vocab_map
    for word, idf_value in zip(vectorizer.get_feature_names(), idfs):
        id = vocab_map.get(word, -1)
        if id != -1:
            weights[id] = idf_value
    return weights