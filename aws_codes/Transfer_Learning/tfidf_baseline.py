import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

from embedding import *
from evaluation import *
from myio import *
import utils
from meter import *

def read_corpus(path):
    total_corpus = []
    id_to_idx = {}
    empty_cnt = 0
    raw_corpus = {}
    fopen = gzip.open if path.endswith(".gz") else open
    idx = 0
    with fopen(path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            if len(title) == 0:
                #print id
                empty_cnt += 1
                continue
            total_corpus.append(title+body)
            id_to_idx[id] = idx
            idx += 1
            title = title.strip().split()
            body = body.strip().split()
            raw_corpus[id] = (title, body)
    say("{} empty titles ignored.\n".format(empty_cnt))
    return raw_corpus, total_corpus, id_to_idx

def evaluate(data, tfidf, id_to_idx):
    N = len(data)
    res = []
    m = AUCMeter()
    for i in range(N):
        pid, qids, labels = data[i]
        q_idxs = []
        p_idx = id_to_idx[pid]
        for id in qids:
            q_idxs.append(id_to_idx[id])
#         if len(q_idxs) == 202:
#             print pid, len(qids)
        q_idxs = np.array(q_idxs)
        scores = np.dot(tfidf[q_idxs,:], tfidf[p_idx, :].T).toarray()
        scores, labels = scores.ravel(), np.array(labels)
        assert len(scores) == len(labels)
        #print scores, labels
        m.add(scores, labels)
        ranks = (-scores).argsort()
        ranked_labels = labels[ranks]
        res.append(ranked_labels)
    e = Evaluation(res)
    MAP = e.MAP()*100
    MRR = e.MRR()*100
    P1 = e.Precision(1)*100
    P5 = e.Precision(5)*100
    #return MAP, MRR, P1, P5
    return m.value(0.05), MAP, MRR, P1, P5

def main(args):

    s_corpus = args.s_corpus#'../Question_Retrieval/askubuntu/text_tokenized.txt'
    t_corpus = args.t_corpus#'Android-master/corpus.tsv.gz'

    raw_corpus_s, total_corpus_s, _ = read_corpus(s_corpus)
    raw_corpus_t, total_corpus_t, id_to_idx = read_corpus(t_corpus)
    print('finish reading corpus.')
    vectorizer = TfidfVectorizer()
    vectorizer.fit(total_corpus_s)
    tfidf = vectorizer.transform(total_corpus_t)
    print('finish create tfidf scores.')

    if args.test_pos and args.test_neg:
        test_dic_pos = read_target_file(args.test_pos, prune_pos_cnt=10, is_neg=False)
        test_dic_neg = read_target_file(args.test_neg, prune_pos_cnt=10, is_neg=True)
        test = read_target_annotations(test_dic_pos, test_dic_neg)
    if args.dev_pos and args.dev_neg:
        dev_dic_pos = read_target_file(args.dev_pos, prune_pos_cnt=10, is_neg=False)
        dev_dic_neg = read_target_file(args.dev_neg, prune_pos_cnt=10, is_neg=True)
        dev = read_target_annotations(dev_dic_pos, dev_dic_neg)
        
    print('evaluating...')
    auc_dev, _, _, _, _ = evaluate(dev, tfidf, id_to_idx)
    auc_test, _, _, _, _ = evaluate(test, tfidf, id_to_idx)

    print "dev AUC:", auc_dev#dev_MAP, dev_MRR, dev_P1, dev_P5
    print "test AUC:", auc_test#test_MAP, test_MRR, test_P1, test_P5

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--s_corpus", type = str)
    argparser.add_argument("--t_corpus", type = str)
    argparser.add_argument("--test_pos", type = str, default = "")
    argparser.add_argument("--test_neg", type = str, default = "")
    argparser.add_argument("--dev_pos", type = str, default = "")
    argparser.add_argument("--dev_neg", type = str, default = "")
    args = argparser.parse_args()
    print args
    print ""
    main(args)


