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
        #if len(item[0]) == 0:
        #    say("empty title after mapping to IDs. Doc No.{}\n".format(id))
        #    continue
        ids_corpus[id] = item   
    return ids_corpus

def read_target_file(path, K_neg=20, prune_pos_cnt=10, is_neg=False):
    """
        used by read_target_annotation
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
            
    if is_neg:
        for key in dic:
            if K_neg != -1 and len(dic[key]) > K_neg:
                neg = dic[key]
                random.shuffle(neg)
                dic[key] = neg[:K_neg]
    print count
    return dic

def read_target_annotations(dic_pos, dic_neg, max_len=20):
    lst = [ ]
    count = 0
    for pid in dic_pos.keys():
        if pid not in dic_neg:
            continue
        pos = dic_pos[pid]
        neg = dic_neg[pid]
        s = set()
        qids = [ ]
        qlabels = [ ]
        for q in pos:
            if q not in s:
                qids.append(q)
                qlabels.append(1 if q not in neg else 0)
                s.add(q)
        count = max_len-len(qids)
        for q in neg:
            if q not in s and count > 0:
                qids.append(q)
                qlabels.append(0)
                s.add(q)
                count -= 1
        lst.append((pid, qids, qlabels))
        count += 1
        if count >= 500 :
            break
    return lst

def read_annotations(path, K_neg=20, prune_pos_cnt=10):
    lst = [ ]
    count = 0
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
            count += 1
            if count >= 500 :
                break
    return lst

def read_train_annotations(path, K_neg=20, prune_pos_cnt=10):
    """
    For each pid, there may be multiple postive id.
    Random sample K_neg number of negative sample for each positive id. not the current method
    Then for each sample, the qids size will always be 21: 
    the first one is positive id, the next 20s are negative ids
    """
    lst = [ ]
    count = 0
    with open(path) as fin:
        for line in fin:
            parts = line.split("\t")
            pid, pos, neg = parts[:3]
            pos = pos.split()
            neg = list(set(neg.split())) ### there are duplicate ids in neg
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
            # different from read_annotation
            if len(qids) > K_neg + 1:
               #print 'length greater than 21', pid
               qids = qids[:21-len(qids[21:])]+qids[21:]
               qlabels = qlabels[:21-len(qlabels[21:])]+qlabels[21:]
            lst.append((pid, qids, qlabels))
            count += 1
            if count >= 200 :
                break
    return lst

def create_source_target_batches(ids_corpus_source, source_data, 
                                 ids_corpus_target, batch_size,
                                 padding_id, perm=None, pad_left=True):
    # shuffle the data
    if perm is None:
        perm = range(len(source_data))
        random.shuffle(perm)

    N = len(source_data)
    cnt = 0
    pid2id = {}
    titles = [ ]
    bodies = [ ]
    triples = [ ]
    batches = [ ]
    for u in xrange(N):
        i = perm[u] #shuffle the data
        pid, qids, qlabels = source_data[i]
        if pid not in ids_corpus_source: 
            continue
        cnt += 1

        for id in [pid] + qids:
            if id not in pid2id:
                if id not in ids_corpus_source: 
                    continue
                pid2id[id] = len(titles)
                t, b = ids_corpus_source[id]
                titles.append(t)
                bodies.append(b)
        
        pid = pid2id[pid]
        pos = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 1 and q in pid2id ]
        neg = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 0 and q in pid2id ]
        triples += [ [pid,x]+neg for x in pos ]

        if cnt == batch_size or u == N-1:
            titles_tar, bodies_tar, d_labels_tar = \
            create_target_batches(ids_corpus_target, len(titles), padding_id, pad_left)

            titles_total, bodies_total = titles+titles_tar, bodies+bodies_tar
            labels_total = np.concatenate((np.array([0]*len(titles)), d_labels_tar))
            #all_data = list(zip(titles_total, bodies_total,labels_total))
            #random.shuffle(all_data)
            #titles_total, bodies_total, labels_total = zip(*all_data)

            titles, bodies = create_one_batch(titles_total, bodies_total, padding_id, pad_left)
            triples = create_hinge_batch(triples)

            batches.append((titles, bodies, triples, labels_total))
            titles = [ ]
            bodies = [ ]
            triples = [ ]
            pid2id = {}
            cnt = 0
    return batches

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

def create_batches(ids_corpus, data, batch_size, padding_id, perm=None, pad_left=True):
    # shuffle the data
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
        #print cnt
        i = perm[u] #shuffle the data
        pid, qids, qlabels = data[i]
        if pid not in ids_corpus: 
            continue
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

def create_train_batches(ids_corpus, data, batch_size, padding_id, pad_left):
    perm = range(len(data))
    random.shuffle(perm)

    N = len(data)
    cnt = 0
    batches = [ ]
    titles = [ ]
    bodies = [ ]
    for u in xrange(N):
        i = perm[u] #shuffle the data
        pid, qids, qlabels = data[i]
        cnt += 1
        if len(qids) > 21:
            print "train batches"
            for i in qids[21:]:
                for id in [pid]+qids[:20]+[i]:
                    t, b = ids_corpus[id]
                    titles.append(t)
                    bodies.append(b)
        else:
            if len([pid]+qids) != 22:
                print [pid]+qids
            for id in [pid]+qids:
                t, b = ids_corpus[id]
                titles.append(t)
                bodies.append(b)

        #print len(titles), len(bodies)
        if cnt == batch_size or u == N-1:
            titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
            batches.append((titles, bodies, np.array(qlabels, dtype="int32")))
            cnt = 0
            titles = [ ]
            bodies = [ ]
    return batches

def create_eval_batches(ids_corpus, data, padding_id, pad_left):
    lst = [ ]
    for pid, qids, qlabels in data:
        titles = [ ]
        bodies = [ ]
        for id in [pid]+qids:
            #print id
            t, b = ids_corpus[id]
            #print t
            titles.append(t)
            bodies.append(b)
        #print len(titles), len(bodies)
        titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
        lst.append((titles, bodies, np.array(qlabels, dtype="int32")))
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