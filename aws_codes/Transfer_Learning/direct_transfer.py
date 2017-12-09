import sys
import gzip
import time
import random
from collections import Counter
import numpy as np
from collections import defaultdict

import torch
from torch import nn
from torch import autograd
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as f

sys.path.append('..')
from embedding import *
from evaluation import *
from utils import *
from myio import *
from meter import *
from qr_model import *


# def average_without_padding(x, mask, eps=1e-8):

#     x = f.normalize(x, p=2, dim=2)
#     # move mask out.
# #     mask = np.not_equal(ids, padding_id).astype('float')
# #     mask = Variable(torch.from_numpy(mask)).float().view(ids.shape[0],ids.shape[1],1)

#     result = torch.sum(x*mask,dim=0)/ (torch.sum(mask,dim=0)+eps)
#     return result 

# class Model(nn.Module):
#     def __init__(self, args, weights=None):
#         super(Model, self).__init__()

#         self.args = args
        
#         n_d = self.n_d = args.hidden_dim # hidden dimension
#         n_e = self.n_e = args.embed_dim # embedding dimension
#         depth = self.depth = args.depth # layer depth
#         dropout = self.dropout = args.dropout
#         self.Relu = nn.ReLU()

#         if args.layer == 'lstm':
#             self.lstm = nn.LSTM(
#                 input_size = n_e,
#                 hidden_size = n_d,
#                 num_layers = depth,
#                 # bidirectional = True,
#                 # dropout = dropout,
#             )

#             self.hidden_1 = None
#             self.hidden_2 = None

#         if args.layer == 'cnn':
#             print "current model is CNN"
#             self.cnn = nn.Conv1d(
#                 in_channels = n_e,
#                 out_channels = n_d, #670 in the paper
#                 kernel_size = args.kernel_size,
#                 padding = (args.kernel_size-1)/2
#             #dropout = dropout,
#             )

#     def init_hidden(self, batch_size):
#         # The axes semantics are (num_layers, minibatch_size, hidden_dim)
#         return (autograd.Variable(torch.zeros(self.depth, batch_size, self.n_d)).cuda(),
#                 autograd.Variable(torch.zeros(self.depth, batch_size, self.n_d)).cuda())


#     def forward(self, xt, xb, mask_t, mask_b):
#         # lstm
#         if args.layer == 'lstm':
#             output_t, _ = self.lstm(xt, self.hidden_1)
#             output_b, _ = self.lstm(xb, self.hidden_2)

#         if args.layer == 'cnn':
#             xt = xt.permute(1,2,0)
#             xb = xb.permute(1,2,0)
#             output_t = self.cnn(self.Relu(xt))
#             output_t = output_t.permute(2,0,1)
#             output_b = self.cnn(self.Relu(xb))
#             output_b = output_b.permute(2,0,1)

#         if args['average']:
#             ht = average_without_padding(output_t, mask_t)
#             hb = average_without_padding(output_b, mask_b)
#         else:
#             ht = output_t[-1]
#             hb = output_b[-1]

#         # get final vector encoding 
#         h_final = (ht+hb)*0.5

#         h_final = f.normalize(h_final, p=2, dim=1)
        
#         return h_final

# def customized_loss(h_final, idps):

#     h_final = torch.squeeze(h_final)
#     xp = h_final[idps.view(idps.size()[0]*idps.size()[1])]
#     xp = xp.view((idps.size()[0], idps.size()[1], args['hidden_dim']))
#     # num query * n_d
#     query_vecs = xp[:,0,:]
#      # num query
#     pos_scores = torch.sum(query_vecs*xp[:,1,:], dim=1)
#     # num query * candidate size
#     neg_scores = torch.sum(torch.unsqueeze(query_vecs, dim=1)*xp[:,2:,:], dim=2)
#     # num query
#     neg_scores = torch.max(neg_scores, dim=1)[0]
#     # print pos_scores, neg_scores
#     diff = neg_scores - pos_scores + 0.3
#     loss = torch.mean((diff>0).float()*diff)
#     prct = torch.mean((diff>0).float())

#     # add regularization
#     l2_reg = None
#     for layer in lstm.parameters():
#         if l2_reg is None:
#             l2_reg = torch.norm(layer.data, 2)
#         else:
#             l2_reg = l2_reg + torch.norm(layer.data, 2)

#     l2_reg = l2_reg * args['l2_reg']
#     cost  = loss + l2_reg 
#     return loss, cost, prct

# def evaluate(data, target=False):
#     lstm.eval()
#     res = [ ]
#     m = AUCMeter()
#     for idts, idbs, labels in data:
#         if args.layer == 'lstm':
#             lstm.hidden_1 = lstm.init_hidden(idts.shape[1])
#             lstm.hidden_2 = lstm.init_hidden(idbs.shape[1])
            
#         # embedding layer
#         xt = embedding_layer.forward(idts.ravel()) # flatten
#         xt = xt.reshape((idts.shape[0], idts.shape[1], args['embed_dim']))
#         xt = Variable(torch.from_numpy(xt).float())

#         xb = embedding_layer.forward(idbs.ravel())
#         xb = xb.reshape((idbs.shape[0], idbs.shape[1], args['embed_dim']))
#         xb = Variable(torch.from_numpy(xb).float())
        
#         # build mask
#         mt = np.not_equal(idts, padding_id).astype('float')
#         mt = Variable(torch.from_numpy(mt).float().view(idts.shape[0], idts.shape[1], 1))
        
#         mb = np.not_equal(idbs, padding_id).astype('float')
#         mb = Variable(torch.from_numpy(mb).float().view(idbs.shape[0], idbs.shape[1], 1))
	
#         h_final = lstm(xt.cuda(), xb.cuda(), mt.cuda(), mb.cuda())
#         h_final = torch.squeeze(h_final)
        
#         scores = torch.mm(h_final[1:], torch.unsqueeze(h_final[0],1))
#         scores = torch.squeeze(scores).data.cpu().numpy()
#         assert len(scores) == len(labels)
#     	if target:
#     	    m.add(scores, labels)
#         ranks = (-scores).argsort()
#         ranked_labels = labels[ranks]
#         res.append(ranked_labels)
#     e = Evaluation(res)
#     MAP = e.MAP()*100
#     MRR = e.MRR()*100
#     P1 = e.Precision(1)*100
#     P5 = e.Precision(5)*100
#     if target:
#     	return m.value(0.05), MAP, MRR, P1, P5
#     else:
#         return MAP, MRR, P1, P5
	
# def read_target_file(path, prune_pos_cnt=10, is_neg=False):
#     """
#         read the target pos and neg files and save them to dictionaries
#     """
#     dic = defaultdict(list)
#     count = 0
#     with open(path) as fin:
#         for line in fin: 
#             count += 1
#             pid, qid = line.split()
#             dic[pid].append(qid)
#             if not is_neg:
#                 if len(dic[pid]) == 0 or (len(dic[pid]) > prune_pos_cnt and prune_pos_cnt != -1): 
#                     continue
            
#     # if is_neg:
#     #     for key in dic:
#     #         if K_neg != -1 and len(dic[key]) > K_neg:
#     #             neg = dic[key]
#     #             random.shuffle(neg)
#     #             dic[key] = neg[:K_neg]
#     print 'target file total lines', count
#     return dic

# def read_target_annotations(dic_pos, dic_neg):
#     lst = [ ]
#     for pid in dic_pos.keys():
#         if pid not in dic_neg:
#             continue
#         pos = dic_pos[pid]
#         neg = dic_neg[pid]
#         s = set()
#         qids = [ ]
#         qlabels = [ ]
#         for q in neg:
#             if q not in s:
#                 qids.append(q)
#                 qlabels.append(0 if q not in pos else 1)
#                 s.add(q)
#         for q in pos:
#             if q not in s:
#                 qids.append(q)
#                 qlabels.append(1)
#                 s.add(q)
#         lst.append((pid, qids, qlabels))
#     return lst

def create_target_batches(ids_corpus, length, padding_id, pad_left=True):
    #maybe not necessay since the dict returns keys in random order
    perm = ids_corpus.keys()
    random.shuffle(perm)

    titles = [ ]
    bodies = [ ]
    for i in perm[:length]:
        titles.append(ids_corpus[i][0])
        bodies.append(ids_corpus[i][1])
    #titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
    return titles, bodies, np.array([1]*length)

def main(args):
    # # Model
    # args = {'dropout':0, 'hidden_dim':100, 'depth':1, 'average':True, 'dev': 'askubuntu/dev.txt', \
    #         'test': 'askubuntu/test.txt', 'normalize':0, 'l2_reg': 0, 'reweight': False, 'embed_dim': 300}

    # # load raw corpus
    #corpus = 'askubuntu/text_tokenized.txt.gz'
    raw_corpus = read_corpus(args.corpus)
    raw_corpus_t = read_corpus(args.corpus_t)
    print 'raw_corpus_t', len(raw_corpus_t)
    #print 'type of raw_corpus', type(raw_corpus), '\n', raw_corpus.keys()[1]#, '\n', raw_corpus.values()[1]


    # # create embedding layer
    #embeddings = 'askubuntu/vector/vectors_pruned.200.txt.gz'
    #embeddings = 'glove.840B.300d.txt.gz'
    embedding_layer = create_embedding_layer(
                raw_corpus,
                n_d = args.embed_dim,
                cut_off = 2,
                embs = load_embedding_iterator(args.embeddings) if args.embeddings else None
            )

    ids_corpus = map_corpus(raw_corpus, embedding_layer, max_len = args.max_seq_len)
    ids_corpus_target = map_corpus(raw_corpus_t, embedding_layer, max_len = args.max_seq_len)
    #ids_corpus['378466']

    say("vocab size={}, corpus size={}\n".format(embedding_layer.n_V,len(raw_corpus)))
    # because there are some out-of-vocabulary words.

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

    if args.dev_pos and args.dev_neg:
        dev_dic_pos = read_target_file(args.dev_pos, prune_pos_cnt=10, is_neg=False)
        dev_dic_neg = read_target_file(args.dev_neg, prune_pos_cnt=10, is_neg=True)
        dev_target = read_target_annotations(dev_dic_pos, dev_dic_neg)
        dev_batches_target = create_eval_batches(ids_corpus_target, dev_target, padding_id, pad_left = not args.average)
    if args.test_pos and args.test_pos:
        test_dic_pos = read_target_file(args.test_pos, prune_pos_cnt=10, is_neg=False)
        test_dic_neg = read_target_file(args.test_neg, prune_pos_cnt=10, is_neg=True)
        test_target = read_target_annotations(test_dic_pos, test_dic_neg)
        test_batches_target = create_eval_batches(ids_corpus_target, test_target, padding_id, pad_left = not args.average)
    print('finish creating target dev/test batches')

    if args.train:
        # # Create training batches
        train = read_annotations(args.train)
        #batch_size = 16
        train_batches = create_batches(ids_corpus, train, args.batch_size, padding_id, pad_left = not args.average)

        print 'train_batch[0][0]', len(train_batches[0][0]), len(train_batches[0][1]), len(train_batches[0][2])
        say("create batches\n")
        say("{} batches, {} tokens in total, {} triples in total\n".format(
                len(train_batches),
                sum(len(x[0].ravel())+len(x[1].ravel()) for x in train_batches),
                sum(len(x[2].ravel()) for x in train_batches)
            ))
        #train_batches = None

        # initialize the model
        lstm = Model(args, weights=weights if args.reweight else None)
        optimizer = optim.Adam(lstm.parameters(), lr=args.learning_rate)

        # get the number of parameters of the model
        num_params = 0
        for layer in model.parameters():
            print type(layer), layer.data.shape, len(layer.data.numpy().ravel())
            num_params += len(layer.data.numpy().ravel())
        say("num of parameters: {}\n".format(num_params))

        model.cuda()

        train_eval = read_annotations(args.train, K_neg=-1, prune_pos_cnt=-1)
        train_eval = create_eval_batches(ids_corpus, train_eval, padding_id, pad_left = not args.average)

        #train = read_annotations('askubuntu/train_random.txt')
        for i in range(args.max_epoch):
            start_time = time.time()
            
            train_batches = create_batches(ids_corpus, train, args.batch_size,
                                        padding_id, pad_left = not args.average)

            N =len(train_batches)
            #avrg_loss = 0
            #avrg_cost = 0
            for j in xrange(N):
                # get current batch
                idts, idbs, idps = train_batches[j]
                optimizer.zero_grad()

                if args.layer == 'lstm':
                    model.hidden_1 = model.init_hidden(idts.shape[1])
                    model.hidden_2 = model.init_hidden(idbs.shape[1])

                # embedding layer
                xt = embedding_layer.forward(idts.ravel()) # flatten
                xt = xt.reshape((idts.shape[0], idts.shape[1], args.embed_dim))
                xt = Variable(torch.from_numpy(xt).float())

                xb = embedding_layer.forward(idbs.ravel())
                xb = xb.reshape((idbs.shape[0], idbs.shape[1], args.embed_dim))
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
                    train_auc, train_MAP, train_MRR, train_P1, train_P5 = evaluate(train_eval, True)
                    dev_auc, dev_MAP, dev_MRR, dev_P1, dev_P5 = evaluate(dev_batches, True)
                    test_auc, test_MAP, test_MRR, test_P1, test_P5 = evaluate(test_batches, True)
                    print "epoch", i, "batch", j, "loss:", loss.data.cpu().numpy()[0], "cost:", cost.data.cpu().numpy()[0], "percent:", prct.data.cpu().numpy()[0]
                    #print "train loss", avrg_loss*1.0/N, avrg_cost*1.0/N
                    print "train", train_auc, train_MAP, train_MRR, train_P1, train_P5
                    print "dev", dev_auc, dev_MAP, dev_MRR, dev_P1, dev_P5
                    print "test", test_auc, test_MAP, test_MRR, test_P1, test_P5
                    print "training running time:", time.time() - start_time
                
                    print('--------------------')
                    start_time_target = time.time()
                    print('evaluating target...')
                    auc_dev, MAP_dev_t, MRR_dev_t, P1_dev_t, P5_dev_t = evaluate(dev_batches_target, True)
                    auc_test, MAP_test_t, MRR_test_t, P1_test_t, P5_test_t = evaluate(test_batches_target, True)
                    #print "train", auc_train, MAP_train, MRR_train, P1_train, P5_train
                    print "dev", auc_dev, MAP_dev_t, MRR_dev_t, P1_dev_t, P5_dev_t
                    print "test", auc_test, MAP_test_t, MRR_test_t, P1_test_t, P5_test_t
                    print "target evaluation running time:", time.time() - start_time_target
                    print "***********************************"

            if args.save_model:
                model_name = "model-" + str(i) + ".pt"
                torch.save(lstm.state_dict(), model_name)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--average", type = int, default = 1)
    argparser.add_argument("--batch_size", type = int, default = 40)
    argparser.add_argument("--embeddings", type = str, default = "")
    argparser.add_argument("--corpus", type = str)
    argparser.add_argument("--corpus_t", type = str)
    argparser.add_argument("--train", type = str, default = "")
    argparser.add_argument("--test", type = str, default = "")
    argparser.add_argument("--dev", type = str, default = "")
    argparser.add_argument("--test_pos", type = str, default = "")
    argparser.add_argument("--dev_pos", type = str, default = "")
    argparser.add_argument("--test_neg", type = str, default = "")
    argparser.add_argument("--dev_neg", type = str, default = "")
    argparser.add_argument("--hidden_dim", "-d", type = int, default = 200)
    argparser.add_argument("--embed_dim", "-d", type = int, default = 200)
    argparser.add_argument("--optimizer", type = str, default = "adam")
    argparser.add_argument("--learning_rate", "-lr", type = float, default = 0.001)
    argparser.add_argument("--l2_reg", type = float, default = 1e-5)
    argparser.add_argument("--activation", "-act", type = str, default = "tanh")
    argparser.add_argument("--depth", type = int, default = 1)
    argparser.add_argument("--dropout", type = float, default = 0.0)
    argparser.add_argument("--max_epoch", type = int, default = 20)
    argparser.add_argument("--max_seq_len", type = int, default = 100)
    argparser.add_argument("--normalize", type = int, default = 1)
    argparser.add_argument("--margin", type = float, default = 0.3)
    argparser.add_argument("--reweight", type = int, default = 0)
    argparser.add_argument("--verbose", type = int, default = 0)
    argparser.add_argument("--cut_off", type = int, default = 2) # original code default 1
    argparser.add_argument("--layer", type = str, default = "lstm")
    argparser.add_argument("--kernel_size", type = int, default = 3)
    argparser.add_argument("--save_model", type = int, default = 0)
    args = argparser.parse_args()
    print args
    print ""
    main(args)

