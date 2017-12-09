import sys
import time
import random
import argparse
import numpy as np

import torch
from torch import nn
from torch import autograd
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as f

sys.path.append('..')
from misc import *
from model import *

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
    # # load raw corpus
    raw_corpus = read_corpus(args.corpus)
    raw_corpus_t = read_corpus(args.corpus_t)
    print 'raw_corpus_t', len(raw_corpus_t)
    
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

    say("vocab size={}, corpus size={}\n".format(embedding_layer.n_V,len(raw_corpus)))
    # because there are some out-of-vocabulary words.

    padding_id = embedding_layer.vocab_map["<padding>"]

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
        train_batches = create_batches(ids_corpus, train, args.batch_size, padding_id, pad_left = not args.average)
        print 'train_batch[0][0]', len(train_batches[0][0]), len(train_batches[0][1]), len(train_batches[0][2])
        say("create batches\n")
        say("{} batches, {} tokens in total, {} triples in total\n".format(
                len(train_batches),
                sum(len(x[0].ravel())+len(x[1].ravel()) for x in train_batches),
                sum(len(x[2].ravel()) for x in train_batches)
            ))
        train_batches = None

        # initialize the model
        model = Model(args, weights=weights if args.reweight else None)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        # get the number of parameters of the model
        num_params = 0
        for layer in model.parameters():
            print type(layer), layer.data.shape, len(layer.data.numpy().ravel())
            num_params += len(layer.data.numpy().ravel())
        say("num of parameters: {}\n".format(num_params))
        
        if args.cuda:
            model.cuda()

        #train_eval = read_annotations(args.train)
        train_eval = create_eval_batches(ids_corpus, train[:200], padding_id, pad_left = not args.average)
        evaluation = Evaluation(args, embedding_layer, padding_id)
        print "evaluation class created"
        #train = read_annotations('askubuntu/train_random.txt')
        for i in range(args.max_epoch):
            start_time = time.time()
            
            train_batches = create_batches(ids_corpus, train[:50], args.batch_size,
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
                
                idps = Variable(torch.from_numpy(idps).long())
                # back prop
                if args.cuda:
                    h_final = model(xt.cuda(), xb.cuda(), mt.cuda(), mb.cuda())       # lstm output
                    loss, cost, prct = customized_loss(args, h_final, idps.cuda(), model)
                else:
                    h_final = model(xt, xb, mt, mb)       # lstm output
                    loss, cost, prct = customized_loss(args, h_final, idps, model)

                cost.backward()                              # backpropagation, compute gradients
                optimizer.step()                             # apply gradients
                #avrg_loss += loss.data.cpu().numpy()[0]
                #avrg_cost += cost.data.cpu().numpy()[0]
                
                if j%200==0:
                    train_auc, train_MAP, train_MRR, train_P1, train_P5 = evaluation.evaluate(train_eval, model, True)
                    dev_auc, dev_MAP, dev_MRR, dev_P1, dev_P5 = evaluation.evaluate(dev_batches, model, True)
                    test_auc, test_MAP, test_MRR, test_P1, test_P5 = evaluation.evaluate(test_batches, model, True)
                    print "epoch", i, "batch", j, "loss:", loss.data.cpu().numpy()[0], "cost:", cost.data.cpu().numpy()[0], "percent:", prct.data.cpu().numpy()[0]
                    #print "train loss", avrg_loss*1.0/N, avrg_cost*1.0/N
                    print "train", train_auc, train_MAP, train_MRR, train_P1, train_P5
                    print "dev", dev_auc, dev_MAP, dev_MRR, dev_P1, dev_P5
                    print "test", test_auc, test_MAP, test_MRR, test_P1, test_P5
                    print "training running time:", time.time() - start_time

                    print('--------------------')
                    start_time_target = time.time()
                    print('evaluating target...')
                    auc_dev, MAP_dev_t, MRR_dev_t, P1_dev_t, P5_dev_t = evaluation.evaluate(dev_batches_target, model, True)
                    auc_test, MAP_test_t, MRR_test_t, P1_test_t, P5_test_t = evaluation.evaluate(test_batches_target, model, True)
                    print "dev", auc_dev, MAP_dev_t, MRR_dev_t, P1_dev_t, P5_dev_t
                    print "test", auc_test, MAP_test_t, MRR_test_t, P1_test_t, P5_test_t
                    print "target evaluation running time:", time.time() - start_time_target
                    print "***********************************"

            if args.save_model:
                model_name = "model-" + str(i) + ".pt"
                torch.save(model.state_dict(), model_name)

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
    argparser.add_argument("--embed_dim", "-e", type = int, default = 200)
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
    argparser.add_argument("--cuda", type = int, default = 0)
    args = argparser.parse_args()
    print args
    print ""
    main(args)

