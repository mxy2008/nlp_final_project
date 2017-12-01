import sys
import gzip
import time
import random
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from numpy import linalg as LA

import torch
from torch import nn
from torch import autograd
from torch import optim
from torch.autograd import Variable

from embedding import *
from evaluation import *
import utils


# # load raw corpus
corpus = 'askubuntu/text_tokenized.txt.gz'
raw_corpus = read_corpus(corpus)
print 'type of raw_corpus', type(raw_corpus), '\n', raw_corpus.keys()[1]#, '\n', raw_corpus.values()[1]


# # create embedding layer
embeddings = 'askubuntu/vector/vectors_pruned.200.txt.gz'
embedding_layer = create_embedding_layer(
            raw_corpus,
            n_d = 200,
            cut_off = 2,
            embs = load_embedding_iterator(embeddings) if embeddings else None
        )

ids_corpus = map_corpus(raw_corpus, embedding_layer, max_len = 100)
ids_corpus['378466']

say("vocab size={}, corpus size={}\n".format(embedding_layer.n_V,len(raw_corpus)))
# because there are some out-of-vocabulary words.

padding_id = embedding_layer.vocab_map["<padding>"]
print 'padding_id', padding_id


# # Create training batches
train = read_annotations('askubuntu/train_random.txt')

batch_size = 25#40
average = False
train_batches = create_batches(ids_corpus, train, batch_size, padding_id, pad_left = not average)

print 'train_batch[0][0]', len(train_batches[0][0]), len(train_batches[0][1]), len(train_batches[0][2])
say("create batches\n")
say("{} batches, {} tokens in total, {} triples in total\n".format(
        len(train_batches),
        sum(len(x[0].ravel())+len(x[1].ravel()) for x in train_batches),
        sum(len(x[2].ravel()) for x in train_batches)
    ))
train_batches = None


# # Model
args = {'dropout':0.1, 'hidden_dim':200, 'depth':1, 'average':True, 'dev': 'askubuntu/dev.txt', \
        'test': 'askubuntu/test.txt', 'normalize':0, 'l2_reg': 1e-5}


class LSTM(nn.Module):
    def __init__(self, args, embedding_layer, weights=None):
        super(LSTM, self).__init__()

        self.args = args
        self.embedding_layer = embedding_layer
        self.weights = weights
        
        n_d = self.n_d = args['hidden_dim'] # hidden dimension
        n_e = self.n_e = embedding_layer.n_d # embedding dimension
        depth = self.depth = args['depth'] # layer depth
        dropout = self.dropout = args['dropout']

        self.lstm = nn.LSTM(     
            input_size = n_e,  
            hidden_size = n_d,     
            num_layers = depth,      
            dropout = dropout, 
        )
        
        

    def forward(self, idts, idbs):
        # embedding layer
        xt = embedding_layer.forward(idts.ravel()) # flatten
        if self.weights is not None:
            xt_w = self.weights[idts.ravel()]
            xt = xt * xt_w.reshape(xt_w.shape[0],1)
        xt = xt.reshape((idts.shape[0], idts.shape[1], self.n_e))
        xt = Variable(torch.from_numpy(xt)).float()

        xb = embedding_layer.forward(idbs.ravel())
        # TODO: add weight as data is unbalanced
        if self.weights is not None:
            xb_w = self.weights[idbs.ravel()]
            xb = xb * xb_w.reshape(xb_w.shape[0],1)
        xb = xb.reshape((idbs.shape[0], idbs.shape[1], self.n_e))
        xb = Variable(torch.from_numpy(xb)).float()

        # lstm
        output_t, (ht, ct) = self.lstm(xt, None)
        output_b, (hb, cb) = self.lstm(xb, None)
        
        # self.ht = output_t#ht
        # self.hb = output_b#hb
        # TODO: add pooling with no padding.
        if args['average']:
            ht = utils.average_without_padding(output_t, idts, padding_id)
            hb = utils.average_without_padding(output_b, idbs, padding_id)
        else:
            ht = output_t[-1]
            bb = output_b[-1]
        #say("h_avg_title dtype: {}\n".format(type(output_t.data)))
        
        # ht = output_t[-1]
        # hb = output_b[-1]
        
        # get final vector encoding 
        h_final = (ht+hb)*0.5
        self.h_final = h_final
        
        return h_final


# initialize the model
lstm = LSTM(args, embedding_layer)
optimizer = optim.Adagrad(lstm.parameters(), lr=0.001)
# get the number of parameters of the model
num_params = 0
for layer in lstm.parameters():
    print type(layer), layer.data.shape, len(layer.data.numpy().ravel())
    num_params += len(layer.data.numpy().ravel())
say("num of parameters: {}\n".format(num_params))


def customized_loss(h_final, idps):

    h_final = torch.squeeze(h_final)
    xp = h_final[torch.from_numpy(idps.ravel()).long()]
    xp = xp.view((idps.shape[0], idps.shape[1], args['hidden_dim']))
    # num query * n_d
    query_vecs = xp[:,0,:]
     # num query
    pos_scores = torch.sum(query_vecs*xp[:,1,:], dim=1)
    # num query * candidate size
    neg_scores = torch.sum(torch.unsqueeze(query_vecs, dim=1)*xp[:,2:,:], dim=2)
    # num query
    neg_scores = torch.max(neg_scores, dim=1)[0]
    diff = neg_scores - pos_scores + 1.0
    loss = torch.mean((diff>0).float()*diff)

    # TODO: add regularization
    l2_reg = None
    for layer in lstm.parameters():
        if l2_reg is None:
            l2_reg = LA.norm(layer.data.numpy(), 2)
        else:
            l2_reg = l2_reg + LA.norm(layer.data.numpy(), 2)

    l2_reg = l2_reg * args['l2_reg']
#   self.cost = self.loss + l2_reg
    loss  += l2_reg
    return loss

def evaluate(data):
    lstm.eval()
    res = [ ]
    for idts, idbs, labels in data:
        h_final = lstm(idts, idbs)
        h_final = torch.squeeze(h_final)
        scores = torch.mm(h_final[1:], torch.unsqueeze(h_final[0],1))
        scores = torch.squeeze(scores).data.numpy()
        assert len(scores) == len(labels)
        ranks = (-scores).argsort()
        ranked_labels = labels[ranks]
        res.append(ranked_labels)
    e = Evaluation(res)
    MAP = e.MAP()*100
    MRR = e.MRR()*100
    P1 = e.Precision(1)*100
    P5 = e.Precision(5)*100
    return MAP, MRR, P1, P5


# train
train = read_annotations('askubuntu/train_random.txt')
dev = read_annotations(args['dev'], K_neg=-1, prune_pos_cnt=-1)
test = read_annotations(args['test'], K_neg=-1, prune_pos_cnt=-1)

for i in range(20):
    start_time = time.time()
    train = read_annotations('askubuntu/train_random.txt')
    dev = read_annotations(args['dev'], K_neg=-1, prune_pos_cnt=-1)
    test = read_annotations(args['test'], K_neg=-1, prune_pos_cnt=-1)
    train_batches = create_batches(ids_corpus, train, batch_size,
                                padding_id, pad_left = not average)
    
    dev = create_eval_batches(ids_corpus, dev, padding_id, pad_left = not average)
    test = create_eval_batches(ids_corpus, test, padding_id, pad_left = not average)

    N =len(train_batches)
    avrg_loss = 0
    for j in xrange(N):
        # get current batch
        idts, idbs, idps = train_batches[i]
        optimizer.zero_grad()
        
        h_final = lstm(idts, idbs)      # lstm output
        loss = customized_loss(h_final, idps)     
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        avrg_loss += loss.data.numpy()[0]
        if j == N-1:
            dev_MAP, dev_MRR, dev_P1, dev_P5 = evaluate(dev)
            test_MAP, test_MRR, test_P1, test_P5 = evaluate(test)
            print "running time:", time.time() - start_time
            print "epoch", i
            print "train loss", avrg_loss*1.0/N
            print "dev", dev_MAP, dev_MRR, dev_P1, dev_P5
            print "test", test_MAP, test_MRR, test_P1, test_P5