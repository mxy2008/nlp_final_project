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
import torch.nn.functional as F
from torch.autograd import Function

sys.path.append('..')
from misc import *
from model import *

def main(args):
    # load raw corpus and combine
    raw_corpus_s = read_corpus(args.corpus)
    raw_corpus_t = read_corpus(args.corpus_t)

    # get a new dictionary with combined corpus
    total_corpus = update_dict(raw_corpus_s, raw_corpus_t)
    #len(total_corpus)

    # build embedding layer.
    #embeddings = 'glove.840B.300d.txt.gz'
    #embeddings = 'askubuntu/vector/vectors_pruned.200.txt.gz'
    embedding_layer = create_embedding_layer(
                total_corpus,
                n_d = args.embed_dim,
                cut_off = 2,
                embs = load_embedding_iterator(args.embeddings) if args.embeddings else None
            )

    # convert the word corpus to ids corpus
    ids_corpus_target = map_corpus(raw_corpus_t, embedding_layer, max_len = args.max_seq_len)
    ids_corpus_source = map_corpus(raw_corpus_s, embedding_layer, max_len = args.max_seq_len)
    print "samples from source and target", len(ids_corpus_source), len(ids_corpus_target)

    say("vocab size={}, corpus size={}\n".format(embedding_layer.n_V,len(total_corpus)))
    padding_id = embedding_layer.vocab_map["<padding>"]

    if args.reweight:
        print 'add weights to the model'
        weights = create_idf_weights(args.corpus, embedding_layer)

    if args.dev:
        dev = read_annotations(args.dev, K_neg=-1, prune_pos_cnt=-1)
        dev_batches = create_eval_batches(ids_corpus_source, dev, padding_id, pad_left = not args.average)
    if args.test:
        test = read_annotations(args.test, K_neg=-1, prune_pos_cnt=-1)
        test_batches =  create_eval_batches(ids_corpus_source, test, padding_id, pad_left = not args.average)

    if args.dev_pos and args.dev_neg:
        dev_dic_pos = read_target_file(args.dev_pos, prune_pos_cnt=10, is_neg=False)
        dev_dic_neg = read_target_file(args.dev_neg, prune_pos_cnt=10, is_neg=True)
        dev_t = read_target_annotations(dev_dic_pos, dev_dic_neg)
        dev_batches_target = create_eval_batches(ids_corpus_target, dev_t, padding_id, pad_left = not args.average)
    if args.test_pos and args.test_pos:
        test_dic_pos = read_target_file(args.test_pos, prune_pos_cnt=10, is_neg=False)
        test_dic_neg = read_target_file(args.test_neg, prune_pos_cnt=10, is_neg=True)
        test_t = read_target_annotations(test_dic_pos, test_dic_neg)
        test_batches_target = create_eval_batches(ids_corpus_target, test_t, padding_id, pad_left = not args.average)
    print('finish creating target dev/test batches')

    if args.train:
        # initialize the model
        model = Model(args, weights=weights if args.reweight else None)
        dcls = domain_classifier(args)

        # get the number of parameters of the model
        num_params = 0
        for layer in model.parameters():
            print type(layer), layer.data.shape, len(layer.data.numpy().ravel())
            num_params += len(layer.data.numpy().ravel())
        say("num of parameters: {}\n".format(num_params))

        if args.cuda:
            model.cuda()
            dcls.cuda()

        # optimizer
        f_optimizer = optim.Adam(model.parameters(), lr=0.001)
        d_optimizer = optim.Adam(list(model.parameters()) + list(dcls.parameters()), lr=0.0001)

        train = read_annotations(args.train)
        train_batches = create_batches(ids_corpus_source, train[:50], 
                                       args.batch_size,
                                       padding_id, pad_left = not args.average)
        say("create batches\n")
        say("{} batches, {} tokens in total, {} triples in total\n".format(
               len(train_batches),
               sum(len(x[0].ravel())+len(x[1].ravel()) for x in train_batches),
               sum(len(x[2].ravel()) for x in train_batches)
           ))
        train_batches = None
        # evaluate
        #train_eval = read_annotations(args.train)
        train_eval = create_eval_batches(ids_corpus_source, train[:200], padding_id, pad_left = not args.average)
        evaluation = Evaluation(args, embedding_layer, padding_id)
        print "evaluation class created"

        for i in range(args.max_epoch):
            start_time = time.time()

            train_batches = create_batches(ids_corpus_source, train[:50], args.batch_size,
                                        padding_id, pad_left = not args.average)

            N =len(train_batches)
            for j in xrange(N):
                
                ####### update feature extraction network ########
                # get current batch
                idts, idbs, idps = train_batches[j]
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
                
                #if args['cuda']:
                #    xt.cuda()
                #    xb.cuda()
                #    mt.cuda()
                #    mb.cuda()
                #    idps.cuda()
                
        	    #print xt, xb, mt, mb, lstm.hidden_1, lstm.hidden_2

                # back prop
                f_optimizer.zero_grad()
                if args.cuda:
                    h_final = model(xt.cuda(), xb.cuda(), mt.cuda(), mb.cuda())      
                    f_loss, f_cost, f_prct = customized_loss(args, h_final, idps.cuda(), model)
                else:
                    h_final = model(xt, xb, mt, mb)
                    f_loss, f_cost, f_prct = customized_loss(args, h_final, idps, model)
                f_cost.backward(retain_graph=True)
                f_optimizer.step()
                
                ##### update discriminator network #####################3
                # random select 20 items each from source and target corpus
                idts, idbs, labels = create_target_batches(ids_corpus_source, ids_corpus_target, \
                                                           args.batch_size_t, padding_id, \
                                                           pad_left = not args.average)
                
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
                
                labels = Variable(torch.from_numpy(labels).float())
                
                #if args['cuda']:
                #    xt.cuda()
                #    xb.cuda()
                #    mt.cuda()
                #    mb.cuda()
                #    labels.cuda()

                # back prop
                d_optimizer.zero_grad()
                if args.cuda:
                    h_final = model(xt.cuda(), xb.cuda(), mt.cuda(), mb.cuda())  
                else:
                    h_final = model(xt, xb, mt, mb)
                estimated = dcls(h_final)
                if args.cuda:
                    d_loss = F.binary_cross_entropy(torch.squeeze(estimated), labels.cuda())
                else:
                    d_loss = F.binary_cross_entropy(torch.squeeze(estimated), labels)
                d_loss.backward(retain_graph=True)
                d_optimizer.step()
                
                
                if j%200==0:
                    train_auc, train_MAP, train_MRR, train_P1, train_P5 = evaluation.evaluate(train_eval, model, True)
                    dev_auc, dev_MAP, dev_MRR, dev_P1, dev_P5 = evaluation.evaluate(dev_batches, model, True)
                    test_auc, test_MAP, test_MRR, test_P1, test_P5 = evaluation.evaluate(test_batches, model, True)
                    print "epoch", i, "batch", j, "loss:", f_loss.data.cpu().numpy()[0], "cost:", d_loss.data.cpu().numpy()[0]
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
    argparser.add_argument("--batch_size_t", type = int, default = 20)
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
    argparser.add_argument("--lam", type = float, default = 1e-4)
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
