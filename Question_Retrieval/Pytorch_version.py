import sys
import gzip
import time
import random
import argparse
import numpy as np

import torch
from torch import nn
from torch import autograd
from torch import optim
from torch.autograd import Variable

from embedding import *
from evaluation import *
from myio import *
from utils import *


# Model
#args = {'dropout':0.3, 'hidden_dim':240, 'depth':1, 'average':True, 'dev': 'askubuntu/dev.txt', \
#        'test': 'askubuntu/test.txt', 'normalize':0, 'l2_reg': 1e-5}

class Model(nn.Module):
    def __init__(self, args, embedding_layer, weights=None):
        super(Model, self).__init__()

        self.args = args
        self.embedding_layer = embedding_layer
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        
        n_d = self.n_d = args.hidden_dim # hidden dimension
        n_e = self.n_e = embedding_layer.n_d # embedding dimension
        depth = self.depth = args.depth # layer depth
        dropout = self.dropout = args.dropout
        self.tanh = nn.Tanh()
        self.Relu = nn.ReLU()

        if args.layer == 'lstm':
            self.model = nn.LSTM(
                input_size = n_e,
                hidden_size = n_d,
                num_layers = depth,
                #dropout = dropout,
            )

            self.hidden_t = None
            self.hidden_b = None

        if args.layer == 'cnn':
            print "CNN"
            self.cnn = nn.Conv1d(
                in_channels = n_e,
                out_channels = n_d, #670 in the paper
                kernel_size = args.kernel_size,
                padding = (args.kernel_size-1)/2
            #dropout = dropout,
            )

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, batch_size, self.n_d)),
                autograd.Variable(torch.zeros(1, batch_size, self.n_d)))

    def forward(self, idts, idbs):
        # embedding layer
        xt = self.embedding_layer.forward(idts.ravel()) # flatten
        if self.weights is not None:
            xt_w = self.weights[idts.ravel()]
            xt = xt * xt_w.reshape(xt_w.shape[0],1)

        xb = self.embedding_layer.forward(idbs.ravel())
        if self.weights is not None:
            xb_w = self.weights[idbs.ravel()]
            xb = xb * xb_w.reshape(xb_w.shape[0],1)

        if args.layer == 'lstm':
            xt = xt.reshape((idts.shape[0], idts.shape[1], self.n_e))
            xb = xb.reshape((idbs.shape[0], idbs.shape[1], self.n_e))
            xt = Variable(torch.from_numpy(xt)).float()
            xb = Variable(torch.from_numpy(xb)).float()
            output_t, self.hidden_t = self.model(xt, (self.tanh(self.hidden_t[0]), 
                                                      self.tanh(self.hidden_t[1])))
            output_b, self.hidden_b = self.model(xb, (self.tanh(self.hidden_b[0]), 
                                                      self.tanh(self.hidden_b[1])))
        if args.layer == 'cnn':
            #xt = xt.reshape((idts.shape[1], self.n_e, idts.shape[0]))
            #xb = xb.reshape((idbs.shape[1], self.n_e, idbs.shape[0]))
            xt = xt.reshape((idts.shape[0], idts.shape[1], self.n_e))
            xt = Variable(torch.from_numpy(xt)).float()
            xt = xt.permute(1,2,0)

            xb = xb.reshape((idbs.shape[0], idbs.shape[1], self.n_e))
            xb = Variable(torch.from_numpy(xb)).float()
            xb = xb.permute(1,2,0)

            output_t = self.cnn(self.Relu(xt))
            output_t = output_t.permute(2,0,1)
            #output_t = output_t.view(output_t.data.shape[2],output_t.data.shape[0],
            #                         output_t.data.shape[1])
            output_b = self.cnn(self.Relu(xb))
            output_b = output_b.permute(2,0,1)
            #output_b = output_b.view(output_b.data.shape[2],output_b.data.shape[0],
            #                         output_b.data.shape[1])

        if args.average:
            #print "mean pooling"
            ht = average_without_padding(output_t, idts, self.padding_id)
            hb = average_without_padding(output_b, idbs, self.padding_id)
        else:
            ht = output_t[-1]
            hb = output_b[-1]
        #say("h_avg_title dtype: {}\n".format(type(output_t.data)))
        
        # get final vector encoding 
        h_final = (ht+hb)*0.5
        self.h_final = h_final
        
        return h_final

    #def train_model(self, ids_corpus, train, dev_batches=None, test_batches=None):
        

def customized_loss(h_final, idps, model):

    h_final = torch.squeeze(h_final)
    
    xp = h_final[torch.from_numpy(idps.ravel()).long()]
    xp = xp.view((idps.shape[0], idps.shape[1], args.hidden_dim))
    # num query * n_d
    query_vecs = xp[:,0,:]
     # num query
    
    q_len = torch.sqrt(torch.sum(query_vecs*query_vecs, dim=1))
    p_len = torch.sqrt(torch.sum(xp[:,1,:]*xp[:,1,:], dim=1))
    pos_scores = torch.sum(query_vecs*xp[:,1,:], dim=1)/(q_len*p_len)

    x_len = torch.sqrt(torch.sum(torch.unsqueeze(query_vecs, dim=1)*torch.unsqueeze(query_vecs, dim=1), dim=2))
    y_len = torch.sqrt(torch.sum(xp[:,2:,:]*xp[:,2:,:], dim=2))
    # num query * candidate size
    neg_scores = torch.sum(torch.unsqueeze(query_vecs, dim=1)*xp[:,2:,:], dim=2)/(x_len*y_len)
    # num query
    neg_scores = torch.max(neg_scores, dim=1)[0]
    diff = neg_scores - pos_scores + 1.0
    loss = torch.mean((diff>0).float()*diff)
    
    # TODO: add regularization
    l2_reg = None
    for layer in model.parameters():
        if l2_reg is None:
            l2_reg = torch.norm(layer.data, 2)**2
        else:
            l2_reg = l2_reg + torch.norm(layer.data, 2)**2

    l2_reg = (l2_reg**0.5) * args.l2_reg
#   self.cost = self.loss + l2_reg
    cost  = loss + l2_reg
    return loss, cost

def evaluate(args, data, model):
    model.eval()
    res = [ ]
    for idts, idbs, labels in data:
        if args.layer == 'lstm':
            model.hidden_t = model.init_hidden(idts.shape[1])
            model.hidden_b = model.init_hidden(idbs.shape[1])
        h_final = model(idts, idbs)
        h_final = torch.squeeze(h_final)
        scores = torch.mm(h_final[1:], torch.unsqueeze(h_final[0],1))
        scores = torch.squeeze(scores).data.numpy()
        #print len(scores)
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


def main(args):
    ## load raw corpus
    raw_corpus = read_corpus(args.corpus)
    #print 'type of raw_corpus', type(raw_corpus), '\n', raw_corpus.keys()[1]#, '\n', raw_corpus.values()[1]

    ## create embedding layer
    #embeddings = 'askubuntu/vector/vectors_pruned.200.txt.gz'
    embedding_layer = create_embedding_layer(
                raw_corpus,
                n_d = args.hidden_dim,#200,
                cut_off = args.cut_off,#2,
                embs = load_embedding_iterator(args.embeddings) if args.embeddings else None
            )

    ids_corpus = map_corpus(raw_corpus, embedding_layer, max_len = args.max_seq_len)

    # because there are some out-of-vocabulary words.
    say("vocab size={}, corpus size={}\n".format(embedding_layer.n_V,len(raw_corpus)))
    padding_id = embedding_layer.vocab_map["<padding>"]
    #print 'padding_id', padding_id

    if args.reweight:
        print 'add weights to the model'
        weights = create_idf_weights(args.corpus, embedding_layer)

    if args.dev:
        dev = read_annotations(args.dev, K_neg=-1, prune_pos_cnt=-1)
        dev_batches = create_eval_batches(ids_corpus, dev, padding_id, pad_left = not args.average)
    if args.test:
        test = read_annotations(args.test, K_neg=-1, prune_pos_cnt=-1)
        test_batches = create_eval_batches(ids_corpus, test, padding_id, pad_left = not args.average)

    if args.train:
        ## Create training batches
        train = read_annotations(args.train)
        train_batches = create_batches(ids_corpus, train, args.batch_size, padding_id, 
                                       pad_left = not args.average)
        #print 'train_batch[0][0]', len(train_batches[0][0]), len(train_batches[0][1]), len(train_batches[0][2])
        say("create batches\n")
        say("{} batches, {} tokens in total, {} triples in total\n".format(
                len(train_batches),
                sum(len(x[0].ravel())+len(x[1].ravel()) for x in train_batches),
                sum(len(x[2].ravel()) for x in train_batches)
            ))
        train_batches = None

        # initialize the model
        model = Model(args, embedding_layer, 
                      weights=weights if args.reweight else None)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) # changed from 0.0001 ro 0.01
        # get the number of parameters of the model
        num_params = 0
        for layer in model.parameters():
            print type(layer), layer.data.shape, len(layer.data.numpy().ravel())
            num_params += len(layer.data.numpy().ravel())
        say("num of parameters: {}\n".format(num_params))

        #model.train(model, ids_corpus, train, dev_batches=None, test_batches=None)
        ## training process
        batch_size = args.batch_size
        #padding_id = self.padding_id
        for i in range(20):
            start_time = time.time()

            train_batches = create_batches(ids_corpus, train, batch_size,
                                           padding_id, pad_left = not args.average)
            #dev_batches = create_eval_batches(ids_corpus, dev, padding_id, 
            #                                  pad_left = not args.average)
            #test_batches = create_eval_batches(ids_corpus, test, padding_id, 
            #                                   pad_left = not args.average)

            N =len(train_batches)
            avrg_loss, avrg_cost = 0, 0

            for j in xrange(N):
                # get current batch
                idts, idbs, idps = train_batches[j]
                optimizer.zero_grad()

                if args.layer == 'lstm':
                    model.hidden_t = model.init_hidden(idts.shape[1])
                    model.hidden_b = model.init_hidden(idbs.shape[1])
                
                h_final = model(idts, idbs)      # model output
                loss, cost = customized_loss(h_final, idps, model)
                cost.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients
                avrg_loss += loss.data.numpy()[0]
                avrg_cost += cost.data.numpy()[0]
                print i, j, loss.data.numpy()[0], cost.data.numpy()[0]
                if j == N-1:
                    dev_MAP, dev_MRR, dev_P1, dev_P5 = evaluate(args, dev_batches, model)
                    test_MAP, test_MRR, test_P1, test_P5 = evaluate(args, test_batches, model)
                    print "running time:", time.time() - start_time
                    print "epoch", i
                    print "train loss", avrg_loss*1.0/N, avrg_cost*1.0/N
                    print "dev", dev_MAP, dev_MRR, dev_P1, dev_P5
                    print "test", test_MAP, test_MRR, test_P1, test_P5




if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--average", type = int, default = 1)
    argparser.add_argument("--batch_size", type = int, default = 40)
    argparser.add_argument("--embeddings", type = str, default = "")
    argparser.add_argument("--lstm", type = int, default = 1)
    argparser.add_argument("--corpus", type = str)
    argparser.add_argument("--train", type = str, default = "")
    argparser.add_argument("--test", type = str, default = "")
    argparser.add_argument("--dev", type = str, default = "")
    argparser.add_argument("--hidden_dim", "-d", type = int, default = 200)
    argparser.add_argument("--optimizer", type = str, default = "adam")
    argparser.add_argument("--learning_rate", "-lr", type = float, default = 0.001)
    argparser.add_argument("--l2_reg", type = float, default = 1e-5)
    argparser.add_argument("--activation", "-act", type = str, default = "tanh")
    argparser.add_argument("--depth", type = int, default = 1)
    argparser.add_argument("--dropout", type = float, default = 0.0)
    argparser.add_argument("--max_epoch", type = int, default = 50)
    argparser.add_argument("--max_seq_len", type = int, default = 100)
    argparser.add_argument("--normalize", type = int, default = 1)
    argparser.add_argument("--reweight", type = int, default = 0)
    argparser.add_argument("--verbose", type = int, default = 0)
    argparser.add_argument("--cut_off", type = int, default = 2) # original code default 1
    argparser.add_argument("--layer", type = str, default = "lstm")
    argparser.add_argument("--kernel_size", type = int, default = 3)
    #argparser.add_argument("--order", type = int, default = 2)
    #argparser.add_argument("--mode", type = int, default = 1)
    #argparser.add_argument("--outgate", type = int, default = 0)
    #argparser.add_argument("--load_pretrain", type = str, default = "")
    argparser.add_argument("--save_model", type = str, default = "")
    args = argparser.parse_args()
    print args
    print ""
    main(args)