import sys
import time
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

def main(args):
    # # load raw corpus
    raw_corpus = read_corpus(args.corpus)

    # # create embedding layer
    embedding_layer = create_embedding_layer(
                raw_corpus,
                n_d = args.embed_dim,#200,
                cut_off = 2,
                embs = load_embedding_iterator(args.embeddings) if args.embeddings else None
            )

    ids_corpus = map_corpus(raw_corpus, embedding_layer, max_len = args.max_seq_len)
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

    if args.train:
        # # Create training batches
        train = read_annotations(args.train)
        train_batches = create_batches(ids_corpus, train, args.batch_size, padding_id, pad_left = not args.average)
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

        for i in range(args.max_epoch):
            start_time = time.time()
            train_batches = create_batches(ids_corpus, train, args.batch_size,
                                        padding_id, pad_left = not args.average)

            N =len(train_batches)
            #print "num of batches", N
            #avrg_loss = 0
            #avrg_cost = 0
            for j in xrange(N):
                # get current batch
                idts, idbs, idps = train_batches[j]
                optimizer.zero_grad()

                if args.layer == 'lstm':
                    model.hidden_1 = model.init_hidden(idts.shape[1])
                    model.hidden_2 = model.init_hidden(idbs.shape[1])
                    #print "hidden initialization 0", model.hidden_t[0].data.shape,  model.hidden_b[0].data.shape

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
                    h_final = model(xt.cuda(), xb.cuda(), mt.cuda(), mb.cuda())# lstm output
                    loss, cost, prct = customized_loss(args, h_final, idps.cuda(), model)
                else:
                    #print xt.data.shape, xb.data.shape, mt.data.shape, mb.data.shape
                    h_final = model(xt, xb, mt, mb) # lstm output
                    loss, cost, prct = customized_loss(args, h_final, idps, model)
                
                cost.backward()                              # backpropagation, compute gradients
                optimizer.step()
                if j%200==0:
                    train_MAP, train_MRR, train_P1, train_P5 = evaluation.evaluate(train_eval, model)
                    dev_MAP, dev_MRR, dev_P1, dev_P5 = evaluation.evaluate(dev_batches, model)
                    test_MAP, test_MRR, test_P1, test_P5 = evaluation.evaluate(test_batches, model)
                    print "running time:", time.time() - start_time
                    print "epoch", i, "batch", j, "loss:", loss.data.cpu().numpy()[0], "cost:", cost.data.cpu().numpy()[0], "percent:", prct.data.cpu().numpy()[0]
                    #print "train loss", avrg_loss*1.0/N, avrg_cost*1.0/N
                    print "train", train_MAP, train_MRR, train_P1, train_P5
                    print "dev", dev_MAP, dev_MRR, dev_P1, dev_P5
                    print "test", test_MAP, test_MRR, test_P1, test_P5

            if args.save_model:
                model_name = "model-" + str(i) + ".pt"
                torch.save(lstm.state_dict(), model_name)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--average", type = int, default = 1)
    argparser.add_argument("--batch_size", type = int, default = 40)
    argparser.add_argument("--embeddings", type = str, default = "")
    argparser.add_argument("--corpus", type = str)
    argparser.add_argument("--train", type = str, default = "")
    argparser.add_argument("--test", type = str, default = "")
    argparser.add_argument("--dev", type = str, default = "")
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
    argparser.add_argument("--cuda", type = int, default = 1)
    args = argparser.parse_args()
    print args
    print ""
    main(args)

