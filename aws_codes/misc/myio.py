import sys
import gzip
import random
import numpy as np
from collections import defaultdict

def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()

def read_corpus(path):
    empty_cnt = 0
    raw_corpus = {}
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            if len(title) == 0:
                print id
                empty_cnt += 1
                continue
            title = title.strip().split()
            body = body.strip().split()
            raw_corpus[id] = (title, body)
    say("{} empty titles ignored.\n".format(empty_cnt))
    return raw_corpus

def map_corpus(raw_corpus, embedding_layer, max_len=100):
    ids_corpus = { }
    for id, pair in raw_corpus.iteritems():
        item = (embedding_layer.map_to_ids(pair[0], filter_oov=True),
                          embedding_layer.map_to_ids(pair[1], filter_oov=True)[:max_len])
        ids_corpus[id] = item   
    return ids_corpus

def read_target_file(path, prune_pos_cnt=10, is_neg=False):
    """
        read the target pos and neg files and save them to dictionaries
    """
    dic = defaultdict(list)
    count = 0
    with open(path) as fin:
        for line in fin:
            count += 1
            pid, qid = line.split()
            dic[pid].append(qid)
            if not is_neg:
                if len(dic[pid]) == 0 or (len(dic[pid]) > prune_pos_cnt and prune_pos_cnt != -1):
                    continue

    # if is_neg:
    #     for key in dic:
    #         if K_neg != -1 and len(dic[key]) > K_neg:
    #             neg = dic[key]
    #             random.shuffle(neg)
    #             dic[key] = neg[:K_neg]
    print 'target file total lines', count
    return dic

def read_target_annotations(dic_pos, dic_neg):
    lst = [ ]
    for pid in dic_pos.keys():
        if pid not in dic_neg:
            continue
        pos = dic_pos[pid]
        neg = dic_neg[pid]
        s = set()
        qids = [ ]
        qlabels = [ ]
        for q in neg:
            if q not in s:
                qids.append(q)
                qlabels.append(0 if q not in pos else 1)
                s.add(q)
        for q in pos:
            if q not in s:
                qids.append(q)
                qlabels.append(1)
                s.add(q)
        lst.append((pid, qids, qlabels))
    return lst

def read_annotations(path, K_neg=20, prune_pos_cnt=10):
    lst = [ ]
    #count = 0
    with open(path) as fin:
        for line in fin:
            parts = line.split("\t")
            pid, pos, neg = parts[:3]
            pos = pos.split()
            neg = neg.split()
            if len(pos) == 0 or (len(pos) > prune_pos_cnt and prune_pos_cnt != -1): continue
            if K_neg != -1:
                random.shuffle(neg)
                neg = neg[:K_neg]
            s = set()
            qids = [ ]
            qlabels = [ ]
            for q in neg:
                if q not in s:
                    qids.append(q)
                    qlabels.append(0 if q not in pos else 1)
                    s.add(q)
            for q in pos:
                if q not in s:
                    qids.append(q)
                    qlabels.append(1)
                    s.add(q)
            lst.append((pid, qids, qlabels))
            #count += 1
            #if count >= 50:
            #    break
    #print "count
    return lst

def create_batches(ids_corpus, data, batch_size, padding_id, perm=None, pad_left=True):
    if perm is None:
        perm = range(len(data))
        random.shuffle(perm)

    N = len(data)
    cnt = 0
    pid2id = {}
    titles = [ ]
    bodies = [ ]
    triples = [ ]
    batches = [ ]
    for u in xrange(N):
        i = perm[u]
        pid, qids, qlabels = data[i]
        if pid not in ids_corpus: continue
        cnt += 1
        for id in [pid] + qids:
            if id not in pid2id:
                if id not in ids_corpus: continue
                pid2id[id] = len(titles)
                t, b = ids_corpus[id]
                titles.append(t)
                bodies.append(b)
        pid = pid2id[pid]
        pos = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 1 and q in pid2id ]
        neg = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 0 and q in pid2id ]
        triples += [ [pid,x]+neg for x in pos ]

        if cnt == batch_size or u == N-1:
            titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
            triples = create_hinge_batch(triples)
            batches.append((titles, bodies, triples))
            titles = [ ]
            bodies = [ ]
            triples = [ ]
            pid2id = {}
            cnt = 0
    return batches

def create_target_batches(ids_corpus_source, ids_corpus_target, length, padding_id, pad_left):

    source = ids_corpus_source.keys()
    random.shuffle(source)
    
    target = ids_corpus_target.keys()
    random.shuffle(target)

    titles = [ ]
    bodies = [ ]
    labels = [ ]
    for i in source[:length]:
        titles.append(ids_corpus_source[i][0])
        bodies.append(ids_corpus_source[i][1])
        labels.append(0)
    
    for i in target[:length]:
        titles.append(ids_corpus_target[i][0])
        bodies.append(ids_corpus_target[i][1])
        labels.append(1)
        
    titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
    return titles, bodies, np.array(labels)

def create_eval_batches(ids_corpus, data, padding_id, pad_left):
    lst = [ ]
    count = 0
    for pid, qids, qlabels in data:
        titles = [ ]
        bodies = [ ]
        for id in [pid]+qids:
            t, b = ids_corpus[id]
            titles.append(t)
            bodies.append(b)
        titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
        lst.append((titles, bodies, np.array(qlabels, dtype="int32")))
        count += 1
        if count > 1500:
            print "create_eval_batches if length greater than a threshold", count
            break
    return lst

def create_one_batch(titles, bodies, padding_id, pad_left):
    max_title_len = max(1, max(len(x) for x in titles))
    max_body_len = max(1, max(len(x) for x in bodies))
    if pad_left:
        titles = np.column_stack([ np.pad(x,(max_title_len-len(x),0),'constant',
                                constant_values=padding_id) for x in titles])
        bodies = np.column_stack([ np.pad(x,(max_body_len-len(x),0),'constant',
                                constant_values=padding_id) for x in bodies])
    else:
	#print 'pad_left', pad_left
        titles = np.column_stack([ np.pad(x,(0,max_title_len-len(x)),'constant',
                                constant_values=padding_id) for x in titles])
        bodies = np.column_stack([ np.pad(x,(0,max_body_len-len(x)),'constant',
                                constant_values=padding_id) for x in bodies])
    return titles, bodies

def create_hinge_batch(triples):
    max_len = max(len(x) for x in triples)
    triples = np.vstack([ np.pad(x,(0,max_len-len(x)),'edge')
                        for x in triples ]).astype('int32')
    return triples
