import copy
import numpy as np 
import torch
from torch.autograd import Variable
import torch.nn.functional as f
from sklearn.feature_extraction.text import TfidfVectorizer

def average_without_padding(x, mask, eps=1e-8):

    x = f.normalize(x, p=2, dim=2)
    # move mask out.
#     mask = np.not_equal(ids, padding_id).astype('float')
#     mask = Variable(torch.from_numpy(mask)).float().view(ids.shape[0],ids.shape[1],1)

    result = torch.sum(x*mask,dim=0)/ (torch.sum(mask,dim=0)+eps)
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

def customized_loss(args, h_final, idps, model):

    h_final = torch.squeeze(h_final)
    xp = h_final[idps.view(idps.size()[0]*idps.size()[1])]
    xp = xp.view((idps.size()[0], idps.size()[1], args.hidden_dim))
    # num query * n_d
    query_vecs = xp[:,0,:]
     # num query
    pos_scores = torch.sum(query_vecs*xp[:,1,:], dim=1)
    # num query * candidate size
    neg_scores = torch.sum(torch.unsqueeze(query_vecs, dim=1)*xp[:,2:,:], dim=2)
    # num query
    neg_scores = torch.max(neg_scores, dim=1)[0]
    # print pos_scores, neg_scores
    diff = neg_scores - pos_scores + args.margin
    loss = torch.mean((diff>0).float()*diff)
    prct = torch.mean((diff>0).float())

    # add regularization
    l2_reg = None
    for layer in model.parameters():
        if l2_reg is None:
            l2_reg = torch.norm(layer.data, 2)
        else:
            l2_reg = l2_reg + torch.norm(layer.data, 2)

    l2_reg = l2_reg * args.l2_reg
    cost  = loss + l2_reg 
    return loss, cost, prct

def update_dict(raw_corpus_s, raw_corpus_t):
    total_corpus = copy.deepcopy(raw_corpus_s)
    for key, pair in raw_corpus_t.iteritems():
        if key not in total_corpus:
            total_corpus[key] = pair
        else:
            title = list(set(total_corpus[key][0])|set(pair[0]))
            body = list(set(total_corpus[key][1])|set(pair[1]))
            total_corpus[key] = (title, body)
    return total_corpus
