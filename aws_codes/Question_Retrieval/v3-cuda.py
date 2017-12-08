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
import torch.nn.functional as f

from embedding import *
from evaluation import *
import utils

# # Model
args = {'dropout':0, 'hidden_dim':150, 'depth':2, 'average':True, 'dev': 'askubuntu/dev.txt', \
        'test': 'askubuntu/test.txt', 'normalize':0, 'l2_reg': 0, 'reweight': False, 'embed_dim': 200}

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

batch_size = 16

train_batches = create_batches(ids_corpus, train, batch_size, padding_id, pad_left = not args['average'])

print 'train_batch[0][0]', len(train_batches[0][0]), len(train_batches[0][1]), len(train_batches[0][2])
say("create batches\n")
say("{} batches, {} tokens in total, {} triples in total\n".format(
        len(train_batches),
        sum(len(x[0].ravel())+len(x[1].ravel()) for x in train_batches),
        sum(len(x[2].ravel()) for x in train_batches)
    ))

train_batches = None

def average_without_padding(x, mask, eps=1e-8):

    x = f.normalize(x, p=2, dim=2)
    # move mask out.
#     mask = np.not_equal(ids, padding_id).astype('float')
#     mask = Variable(torch.from_numpy(mask)).float().view(ids.shape[0],ids.shape[1],1)

    result = torch.sum(x*mask,dim=0)/ (torch.sum(mask,dim=0)+eps)
    return result 

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()

        self.args = args
        
        n_d = self.n_d = args['hidden_dim'] # hidden dimension
        n_e = self.n_e = args['embed_dim'] # embedding dimension
        depth = self.depth = args['depth'] # layer depth
        dropout = self.dropout = args['dropout']

        self.lstm = nn.LSTM(
            input_size = n_e,
            hidden_size = n_d,
            num_layers = depth,
            # bidirectional = True,
            # dropout = dropout,
        )

        self.hidden_1 = None
        self.hidden_2 = None

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(self.depth, batch_size, self.n_d)).cuda(),
                autograd.Variable(torch.zeros(self.depth, batch_size, self.n_d)).cuda())


    def forward(self, xt, xb, mask_t, mask_b):
        # lstm
        output_t, _ = self.lstm(xt, self.hidden_1)
        output_b, _ = self.lstm(xb, self.hidden_2)

        if args['average']:
            ht = average_without_padding(output_t, mask_t)
            hb = average_without_padding(output_b, mask_b)
        else:
            ht = output_t[-1]
            hb = output_b[-1]

        # get final vector encoding 
        h_final = (ht+hb)*0.5

        h_final = f.normalize(h_final, p=2, dim=1)
        
        return h_final

def customized_loss(h_final, idps):

    h_final = torch.squeeze(h_final)
    xp = h_final[idps.view(idps.size()[0]*idps.size()[1])]
    xp = xp.view((idps.size()[0], idps.size()[1], args['hidden_dim']))
    # num query * n_d
    query_vecs = xp[:,0,:]
     # num query
    pos_scores = torch.sum(query_vecs*xp[:,1,:], dim=1)
    # num query * candidate size
    neg_scores = torch.sum(torch.unsqueeze(query_vecs, dim=1)*xp[:,2:,:], dim=2)
    # num query
    neg_scores = torch.max(neg_scores, dim=1)[0]
    # print pos_scores, neg_scores
    diff = neg_scores - pos_scores + 0.3
    loss = torch.mean((diff>0).float()*diff)
    prct = torch.mean((diff>0).float())

    # add regularization
    l2_reg = None
    for layer in lstm.parameters():
        if l2_reg is None:
            l2_reg = torch.norm(layer.data, 2)
        else:
            l2_reg = l2_reg + torch.norm(layer.data, 2)

    l2_reg = l2_reg * args['l2_reg']
    cost  = loss + l2_reg 
    return loss, cost, prct

def evaluate(data):
    lstm.eval()
    res = [ ]
    for idts, idbs, labels in data:
        lstm.hidden_1 = lstm.init_hidden(idts.shape[1])
        lstm.hidden_2 = lstm.init_hidden(idbs.shape[1])
        
	#lstm.init_hidden(idts.shape[1], idbs.shape[1])

        # embedding layer
        xt = embedding_layer.forward(idts.ravel()) # flatten
        xt = xt.reshape((idts.shape[0], idts.shape[1], args['embed_dim']))
        xt = Variable(torch.from_numpy(xt).float())

        xb = embedding_layer.forward(idbs.ravel())
        xb = xb.reshape((idbs.shape[0], idbs.shape[1], args['embed_dim']))
        xb = Variable(torch.from_numpy(xb).float())
        
        # build mask
        mt = np.not_equal(idts, padding_id).astype('float')
        mt = Variable(torch.from_numpy(mt).float().view(idts.shape[0], idts.shape[1], 1))
        
        mb = np.not_equal(idbs, padding_id).astype('float')
        mb = Variable(torch.from_numpy(mb).float().view(idbs.shape[0], idbs.shape[1], 1))
	
	h_final = lstm(xt.cuda(), xb.cuda(), mt.cuda(), mb.cuda())
	h_final = torch.squeeze(h_final)
        
        scores = torch.mm(h_final[1:], torch.unsqueeze(h_final[0],1))
        scores = torch.squeeze(scores).data.cpu().numpy()
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

# initialize the model
lstm = LSTM(args)

optimizer = optim.Adam(lstm.parameters(), lr=0.001)

# get the number of parameters of the model
num_params = 0
for layer in lstm.parameters():
    print type(layer), layer.data.shape, len(layer.data.numpy().ravel())
    num_params += len(layer.data.numpy().ravel())
say("num of parameters: {}\n".format(num_params))

lstm.cuda()

train_eval = read_annotations('askubuntu/train_random.txt', K_neg=-1, prune_pos_cnt=-1)
dev = read_annotations(args['dev'], K_neg=-1, prune_pos_cnt=-1)
test = read_annotations(args['test'], K_neg=-1, prune_pos_cnt=-1)

train_eval = create_eval_batches(ids_corpus, train_eval, padding_id, pad_left = not args['average'])
dev = create_eval_batches(ids_corpus, dev, padding_id, pad_left = not args['average'])
test = create_eval_batches(ids_corpus, test, padding_id, pad_left = not args['average'])

for i in range(20):
    start_time = time.time()
    
    train = read_annotations('askubuntu/train_random.txt')
    train_batches = create_batches(ids_corpus, train, batch_size,
                                padding_id, pad_left = not args['average'])


    N =len(train_batches)
    #avrg_loss = 0
    #avrg_cost = 0
    for j in xrange(N):
        # get current batch
        idts, idbs, idps = train_batches[j]
        optimizer.zero_grad()

        lstm.hidden_1 = lstm.init_hidden(idts.shape[1])
        lstm.hidden_2 = lstm.init_hidden(idbs.shape[1])


        # embedding layer
        xt = embedding_layer.forward(idts.ravel()) # flatten
        xt = xt.reshape((idts.shape[0], idts.shape[1], args['embed_dim']))
        xt = Variable(torch.from_numpy(xt).float())

        xb = embedding_layer.forward(idbs.ravel())
        xb = xb.reshape((idbs.shape[0], idbs.shape[1], args['embed_dim']))
        xb = Variable(torch.from_numpy(xb).float())
        
        # build mask
        mt = np.not_equal(idts, padding_id).astype('float')
        mt = Variable(torch.from_numpy(mt).float().view(idts.shape[0], idts.shape[1], 1))
        
        mb = np.not_equal(idbs, padding_id).astype('float')
        mb = Variable(torch.from_numpy(mb).float().view(idbs.shape[0], idbs.shape[1], 1))
        
        # back prop
        h_final = lstm(xt.cuda(), xb.cuda(), mt.cuda(), mb.cuda())       # lstm output
	idps = Variable(torch.from_numpy(idps).long())
	
        loss, cost, prct = customized_loss(h_final, idps.cuda())     
        cost.backward()                              # backpropagation, compute gradients
        optimizer.step()                             # apply gradients
        #avrg_loss += loss.data.cpu().numpy()[0]
        #avrg_cost += cost.data.cpu().numpy()[0]
        
	if j%200==0:
            train_MAP, train_MRR, train_P1, train_P5 = evaluate(train_eval)
            dev_MAP, dev_MRR, dev_P1, dev_P5 = evaluate(dev)
            test_MAP, test_MRR, test_P1, test_P5 = evaluate(test)
            print "running time:", time.time() - start_time
            print "epoch", i, "batch", j, "loss:", loss.data.cpu().numpy()[0], "cost:", cost.data.cpu().numpy()[0], "percent:", prct.data.cpu().numpy()[0]
            #print "train loss", avrg_loss*1.0/N, avrg_cost*1.0/N
            print "train", train_MAP, train_MRR, train_P1, train_P5
            print "dev", dev_MAP, dev_MRR, dev_P1, dev_P5
	    print "test", test_MAP, test_MRR, test_P1, test_P5
	
	model_name = "model-" + str(i) + ".pt"
	torch.save(lstm.state_dict(), model_name)

