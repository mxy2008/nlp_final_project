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
import torch.nn.functional as F
from torch.autograd import Function

from embedding import *
from evaluation import *
import utils
from meter import *

args = {'embeddings':'../data/askubuntu/vector/vectors_pruned.200.txt.gz',\
        's_corpus':'../data/askubuntu/text_tokenized.txt.gz', \
        't_corpus':'../data/Android/corpus.tsv.gz',\
        'train':'../data/askubuntu/train_random.txt',\
        'test':'../data/askubuntu/test.txt', 'dev':'../../data/askubuntu/dev.txt',\
        'dropout': 0, 'hidden_dim': 100, 'batch_size':16, 'lr':0.001, 'l2_reg':0,\
        'layer':'lstm','average':1, 'reweight':0, 'learning_rate':0.001,\
        'depth': 1, 'embedding_dim': 300, 'cuda': True, 'batch_size_t': 20, 'lambda': 1e-4}


# load raw corpus and combine
raw_corpus_s = read_corpus(args['s_corpus'])
raw_corpus_t = read_corpus(args['t_corpus'])

def update_dict(raw_corpus_s, raw_corpus_t):
    total_corpus = raw_corpus_s
    for key, pair in raw_corpus_t.iteritems():
        if key not in total_corpus:
            total_corpus[key] = pair
        else:
            title = list(set(total_corpus[key][0])|set(pair[0]))
            body = list(set(total_corpus[key][1])|set(pair[1]))
            total_corpus[key] = (title, body)
    return total_corpus

# get a new dictionary with combined corpus
total_corpus = update_dict(raw_corpus_s, raw_corpus_t)
#len(total_corpus)

# build embedding layer.
#embeddings = 'glove.840B.300d.txt.gz'
embeddings = '../data/askubuntu/vector/vectors_pruned.200.txt.gz'
embedding_layer = create_embedding_layer(
            total_corpus,
            n_d = args['embedding_dim'],
            cut_off = 2,
            embs = load_embedding_iterator(embeddings) if embeddings else None
        )

# convert the word corpus to ids corpus
ids_corpus_target = map_corpus(raw_corpus_t, embedding_layer, max_len = 100)
ids_corpus_source = map_corpus(raw_corpus_s, embedding_layer, max_len = 100)

print "samples from source and target", len(ids_corpus_source), len(ids_corpus_target)

say("vocab size={}, corpus size={}\n".format(embedding_layer.n_V,len(total_corpus)))
padding_id = embedding_layer.vocab_map["<padding>"]

#train = read_annotations(args['train'])
#train_batches = create_source_target_batches(ids_corpus_source, train, 
#                             ids_corpus_target, args['batch_size'],
#                             padding_id, pad_left = not args['average'])
#say("create batches\n")
#say("{} batches, {} tokens in total, {} triples in total\n".format(
#        len(train_batches),
#        sum(len(x[0].ravel())+len(x[1].ravel()) for x in train_batches),
#        sum(len(x[2].ravel()) for x in train_batches)
#    ))

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()

        self.args = args

        n_d = self.n_d = args['hidden_dim'] # hidden dimension
        n_e = self.n_e = args['embedding_dim'] # embedding dimension
        depth = self.depth = args['depth'] # layer depth
        dropout = self.dropout = args['dropout']

        self.lstm = nn.LSTM(
            input_size = n_e,
            hidden_size = n_d,
            num_layers = depth,
            bidirectional = True,
            # dropout = dropout,
        )

        self.hidden_1 = None
        self.hidden_2 = None

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if args['cuda']:
            return (autograd.Variable(torch.zeros(self.depth*2, batch_size, self.n_d)).cuda(),
                    autograd.Variable(torch.zeros(self.depth*2, batch_size, self.n_d)).cuda())
        else:
            return (autograd.Variable(torch.zeros(self.depth*2, batch_size, self.n_d)),
                autograd.Variable(torch.zeros(self.depth*2, batch_size, self.n_d)))


    def forward(self, xt, xb, mask_t, mask_b):
        # lstm
        output_t, self.hidden_1 = self.lstm(xt, self.hidden_1)
        output_b, self.hidden_1 = self.lstm(xb, self.hidden_2)

        if args['average']:
            ht = average_without_padding(output_t, mask_t)
            hb = average_without_padding(output_b, mask_b)
        else:
            ht = output_t[-1]
            hb = output_b[-1]

        # get final vector encoding 
        h_final = (ht+hb)*0.5

        h_final = F.normalize(h_final, p=2, dim=1)

        return h_final

def average_without_padding(x, mask, eps=1e-8):

    x = F.normalize(x, p=2, dim=2)
    result = torch.sum(x*mask,dim=0)/ (torch.sum(mask,dim=0)+eps)
    return result


class GradReverse(Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -args['lambda'])

def grad_reverse(x):
    return GradReverse()(x)

class domain_classifier(nn.Module):
    def __init__(self):
        super(domain_classifier, self).__init__()
        self.fc1 = nn.Linear(args['hidden_dim']*2, 300) 
        self.fc2 = nn.Linear(300, 150)
        self.fc3 = nn.Linear(150, 1)
        #self.drop = nn.Dropout2d(0.25)

    def forward(self, x):
        #x = grad_reverse(x)
	x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return F.sigmoid(x)


def customized_loss(h_final, idps):

    h_final = torch.squeeze(h_final)
    xp = h_final[idps.view(idps.size()[0]*idps.size()[1])]
    xp = xp.view((idps.size()[0], idps.size()[1], args['hidden_dim']*2))
    # num query * n_d
    query_vecs = xp[:,0,:]
     # num query
    pos_scores = torch.sum(query_vecs*xp[:,1,:], dim=1)
    # num query * candidate size
    neg_scores = torch.sum(torch.unsqueeze(query_vecs, dim=1)*xp[:,2:,:], dim=2)
    # num query
    neg_scores = torch.max(neg_scores, dim=1)[0]
    # print pos_scores, neg_scores
    diff = neg_scores - pos_scores + 0.4
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

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(20,1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if args['cuda'] else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if args['cuda']:
        interpolates = interpolates.cuda()

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    # TODO: Make ConvBackward diffentiable
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if args['cuda'] else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty



def evaluate(data):
    lstm.eval()
    res = [ ]
    m = AUCMeter()
    for idts, idbs, labels in data:
        lstm.hidden_1 = lstm.init_hidden(idts.shape[1])
        lstm.hidden_2 = lstm.init_hidden(idbs.shape[1])

        # embedding layer
        xt = embedding_layer.forward(idts.ravel()) # flatten
        xt = xt.reshape((idts.shape[0], idts.shape[1], args['embedding_dim']))
        xt = Variable(torch.from_numpy(xt).float())

        xb = embedding_layer.forward(idbs.ravel())
        xb = xb.reshape((idbs.shape[0], idbs.shape[1], args['embedding_dim']))
        xb = Variable(torch.from_numpy(xb).float())

        # build mask
        mt = np.not_equal(idts, padding_id).astype('float')
        mt = Variable(torch.from_numpy(mt).float().view(idts.shape[0], idts.shape[1], 1))

        mb = np.not_equal(idbs, padding_id).astype('float')
        mb = Variable(torch.from_numpy(mb).float().view(idbs.shape[0], idbs.shape[1], 1))

	#if args['cuda']:
	#    xt.cuda()
	#    xb.cuda()
	#    mt.cuda()
        #    mb.cuda()

        h_final = lstm(xt.cuda(), xb.cuda(), mt.cuda(), mb.cuda())
        h_final = torch.squeeze(h_final)

        scores = torch.mm(h_final[1:], torch.unsqueeze(h_final[0],1))
        scores = torch.squeeze(scores).data.cpu().numpy()
        assert len(scores) == len(labels)
        m.add(scores, labels)
        ranks = (-scores).argsort()
        ranked_labels = labels[ranks]
        res.append(ranked_labels)
    e = Evaluation(res)
    MAP = e.MAP()*100
    MRR = e.MRR()*100
    P1 = e.Precision(1)*100
    P5 = e.Precision(5)*100
    return m.value(0.05), MAP, MRR, P1, P5


# initialize the model
lstm = LSTM(args)
dcls = domain_classifier()

# get the number of parameters of the model
num_params = 0
for layer in lstm.parameters():
    print type(layer), layer.data.shape, len(layer.data.numpy().ravel())
    num_params += len(layer.data.numpy().ravel())
say("num of parameters: {}\n".format(num_params))

if args['cuda']:
    lstm.cuda()
    dcls.cuda()
    
# optimizer
f_optimizer = optim.Adam(lstm.parameters(), lr=0.0001, betas=(0.5, 0.9))
d_optimizer = optim.Adam(dcls.parameters(), lr=0.0001, betas=(0.5, 0.9))
one = torch.FloatTensor([1])
mone = one * -1
if args['cuda']:
    one = one.cuda()
    mone = mone.cuda()


train = read_annotations('askubuntu/train_random.txt')
test_dic_pos = read_target_file('Android/test.pos.txt', prune_pos_cnt=10, is_neg=False)
test_dic_neg = read_target_file('Android/test.neg.txt', prune_pos_cnt=10, is_neg=True)
dev_dic_pos = read_target_file('Android/dev.pos.txt', prune_pos_cnt=10, is_neg=False)
dev_dic_neg = read_target_file('Android/dev.neg.txt', prune_pos_cnt=10, is_neg=True)

if args['dev']:
    dev = read_target_annotations(dev_dic_pos, dev_dic_neg)
    dev_batches_target = create_eval_batches(ids_corpus_target, dev, padding_id, pad_left = not args['average'])
if args['test']:
    test = read_target_annotations(test_dic_pos, test_dic_neg)
    test_batches_target = create_eval_batches(ids_corpus_target, test, padding_id, pad_left = not args['average'])

# evaluate
train_eval = read_annotations('askubuntu/train_random.txt', K_neg=-1, prune_pos_cnt=-1)
dev = read_annotations(args['dev'], K_neg=-1, prune_pos_cnt=-1)
test = read_annotations(args['test'], K_neg=-1, prune_pos_cnt=-1)

train_eval = create_eval_batches(ids_corpus_source, train_eval, padding_id, pad_left = not args['average'])
dev = create_eval_batches(ids_corpus_source, dev, padding_id, pad_left = not args['average'])
test = create_eval_batches(ids_corpus_source, test, padding_id, pad_left = not args['average'])


for i in range(20):
    start_time = time.time()

    train_batches = create_batches(ids_corpus_source, train, args['batch_size'],
                                padding_id, pad_left = not args['average'])

    N =len(train_batches)
    for j in xrange(N):
        for p in dcls.parameters():
            p.requires_grad = False  # to avoid computation

        ####### update feature extraction network ########
        # get current batch
        idts, idbs, idps = train_batches[j]

        lstm.hidden_1 = lstm.init_hidden(idts.shape[1])
        lstm.hidden_2 = lstm.init_hidden(idbs.shape[1])

        # embedding layer
        xt = embedding_layer.forward(idts.ravel()) # flatten
        xt = xt.reshape((idts.shape[0], idts.shape[1], args['embedding_dim']))
        xt = Variable(torch.from_numpy(xt).float())

        xb = embedding_layer.forward(idbs.ravel())
        xb = xb.reshape((idbs.shape[0], idbs.shape[1], args['embedding_dim']))
        xb = Variable(torch.from_numpy(xb).float())

        # build mask
        mt = np.not_equal(idts, padding_id).astype('float')
        mt = Variable(torch.from_numpy(mt).float().view(idts.shape[0], idts.shape[1], 1))

        mb = np.not_equal(idbs, padding_id).astype('float')
        mb = Variable(torch.from_numpy(mb).float().view(idbs.shape[0], idbs.shape[1], 1))

        idps = Variable(torch.from_numpy(idps).long())
        
        #if args['cuda']:
        #    xt.cuda()
        #    xb.cuda()
        #    mt.cuda()
        #    mb.cuda()
        #    idps.cuda()
        
	#print xt, xb, mt, mb, lstm.hidden_1, lstm.hidden_2

        # back prop
        lstm.zero_grad()
        h_final = lstm(xt.cuda(), xb.cuda(), mt.cuda(), mb.cuda())      
        f_loss, f_cost, f_prct = customized_loss(h_final, idps.cuda())
        f_cost.backward()
        f_optimizer.step()
        
        ##### update discriminator network #####################
        for p in dcls.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

	for iter in range(10):
       	    # random select 20 items each from source and target corpus
            idts, idbs, labels = create_target_batches(ids_corpus_source, ids_corpus_target, args['batch_size_t'], padding_id, pad_left = not args['average'])
        
        
            lstm.hidden_1 = lstm.init_hidden(idts.shape[1])
            lstm.hidden_2 = lstm.init_hidden(idbs.shape[1])

            # embedding layer
            xt = embedding_layer.forward(idts.ravel()) # flatten
            xt = xt.reshape((idts.shape[0], idts.shape[1], args['embedding_dim']))
            xt = Variable(torch.from_numpy(xt).float())

            xb = embedding_layer.forward(idbs.ravel())
            xb = xb.reshape((idbs.shape[0], idbs.shape[1], args['embedding_dim']))
            xb = Variable(torch.from_numpy(xb).float())

            # build mask
            mt = np.not_equal(idts, padding_id).astype('float')
            mt = Variable(torch.from_numpy(mt).float().view(idts.shape[0], idts.shape[1], 1))

            mb = np.not_equal(idbs, padding_id).astype('float')
            mb = Variable(torch.from_numpy(mb).float().view(idbs.shape[0], idbs.shape[1], 1))
        
            labels = Variable(torch.from_numpy(labels).float())
        
            #if args['cuda']:
            #    xt.cuda()
            #    xb.cuda()
            #    mt.cuda()
            #    mb.cuda()
            #    labels.cuda()

            # back prop
            dcls.zero_grad()
            h_final = lstm(xt.cuda(), xb.cuda(), mt.cuda(), mb.cuda())      
            s_loss = dcls(h_final[0:20])
            t_loss = dcls(h_final[20:])
            s_loss = s_loss.mean()
            t_loss = t_loss.mean()
            s_loss.backward(mone, retain_graph=True)
	    t_loss.backward(one, retain_graph=True)
	    W_loss = s_loss - t_loss
	    gradient_penalty = calc_gradient_penalty(dcls, h_final[:20].data, h_final[20:].data)
	    gradient_penalty.backward()
	    D_cost = s_loss - t_loss + gradient_penalty
            d_optimizer.step()	
        
	print "epoch", i, "batch", j, "G_loss:", f_loss.data.cpu().numpy()[0], "W_loss:", W_loss.data.cpu().numpy()[0], "D_cost", D_cost.data.cpu().numpy()[0]
        
        if j%200==0:
            train_auc, train_MAP, train_MRR, train_P1, train_P5 = evaluate(train_eval)
            dev_auc, dev_MAP, dev_MRR, dev_P1, dev_P5 = evaluate(dev)
            test_auc, test_MAP, test_MRR, test_P1, test_P5 = evaluate(test)
            print "epoch", i, "batch", j, "G_loss:", f_loss.data.cpu().numpy()[0], "W_loss:", W_loss.data.cpu().numpy()[0], "D_cost", D_cost.data.cpu().numpy()[0]
            #print "train loss", avrg_loss*1.0/N, avrg_cost*1.0/N
            print "train", train_auc, train_MAP, train_MRR, train_P1, train_P5
            print "dev", dev_auc, dev_MAP, dev_MRR, dev_P1, dev_P5
            print "test", test_auc, test_MAP, test_MRR, test_P1, test_P5
            print "training running time:", time.time() - start_time

            print('--------------------')
            start_time_target = time.time()
            print('evaluating target...')
            auc_dev, MAP_dev_t, MRR_dev_t, P1_dev_t, P5_dev_t = evaluate(dev_batches_target)
            auc_test, MAP_test_t, MRR_test_t, P1_test_t, P5_test_t = evaluate(test_batches_target)
            #print "train", auc_train, MAP_train, MRR_train, P1_train, P5_train
            print "dev", auc_dev, MAP_dev_t, MRR_dev_t, P1_dev_t, P5_dev_t
            print "test", auc_test, MAP_test_t, MRR_test_t, P1_test_t, P5_test_t
            print "target evaluation running time:", time.time() - start_time_target
            print "***********************************"

