import sys
import gzip
import random
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# def say(s, stream=sys.stdout):
#     stream.write(s)
#     stream.flush()

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

def create_idf_weights(corpus_path, embedding_layer):
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,1), binary=False)

    lst = [ ]
    fopen = gzip.open if corpus_path.endswith(".gz") else open
    with fopen(corpus_path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            lst.append(title)
            lst.append(body)
    vectorizer.fit_transform(lst)

    idfs = vectorizer.idf_
    avg_idf = sum(idfs)/(len(idfs)+0.0)/4.0
    weights = np.array([ avg_idf for i in xrange(embedding_layer.n_V) ])
    vocab_map = embedding_layer.vocab_map
    for word, idf_value in zip(vectorizer.get_feature_names(), idfs):
        id = vocab_map.get(word, -1)
        if id != -1:
            weights[id] = idf_value
    return weights#, name="word_weights")

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
            if count >= 500:
                break
    return lst


#default_rng = np.random.RandomState(random.randint(0,9999))
def random_init(size, rng=None, rng_type=None):
    '''
    Return initial parameter values of the specified size

    Inputs
    ------

        size            : size of the parameter, e.g. (100, 200) and (100,)
        rng             : random generator; the default is used if None
        rng_type        : the way to initialize the values
                            None    -- (default) uniform [-0.05, 0.05]
                            normal  -- Normal distribution with unit variance and zero mean
                            uniform -- uniform distribution with unit variance and zero mean
'''
    if rng is None: rng = np.random.RandomState(random.randint(0,9999))
    if rng_type is None:
        #vals = rng.standard_normal(size)
        vals = rng.uniform(low=-0.05, high=0.05, size=size)

    elif rng_type == "normal":
        vals = rng.standard_normal(size)

    elif rng_type == "uniform":
        vals = rng.uniform(low=-3.0**0.5, high=3.0**0.5, size=size)

    else:
        raise Exception(
            "unknown random inittype: {}".format(rng_type)
          )

    return vals#.astype(theano.config.floatX)

def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                vals = np.array([ float(x) for x in parts[1:] ])
                yield word, vals

def create_embedding_layer(raw_corpus, n_d, embs=None, \
        cut_off=2, unk="<unk>", padding="<padding>", fix_init_embs=True):
    
    #count the occurrence of each word in the title and body
    cnt = Counter(w for id, pair in raw_corpus.iteritems() \
                        for x in pair for w in x)
    cnt[unk] = cut_off + 1
    cnt[padding] = cut_off + 1
    embedding_layer = EmbeddingLayer(
            n_d = n_d,
            #vocab = (w for w,c in cnt.iteritems() if c > cut_off),
            vocab = [ unk, padding ],
            embs = embs,
            fix_init_embs = fix_init_embs
        )
    return embedding_layer
    
class EmbeddingLayer(object):
    '''
        Embedding layer that
                (1) maps string tokens into integer IDs
                (2) maps integer IDs into embedding vectors (as matrix)

        Inputs
        ------

        n_d             : dimension of word embeddings; may be over-written if embs
                            is specified
        vocab           : an iterator of string tokens; the layer will allocate an ID
                            and a vector for each token in it
        oov             : out-of-vocabulary token
        embs            : an iterator of (word, vector) pairs; these will be added to
                            the layer
        fix_init_embs   : whether to fix the initial word vectors loaded from embs

    '''
    def __init__(self, n_d, vocab, oov="<unk>", embs=None, fix_init_embs=True):

        #read in pre-trained word vectors
        if embs is not None:
            # all unique word
            lst_words = [ ]
            # map each unique word to an index
            vocab_map = {}
            # all word vectors
            emb_vals = [ ]
            for word, vector in embs:
                assert word not in vocab_map, "Duplicate words in initial embeddings"
                vocab_map[word] = len(vocab_map)
                emb_vals.append(vector)
                lst_words.append(word)

            #?? truncate the word vector starting from init_end
            self.init_end = len(emb_vals) if fix_init_embs else -1
            #check whether the n_d parameter is the same as 
            #the size of pre-trained word vectors
            if n_d != len(emb_vals[0]):
                say("WARNING: n_d ({}) != init word vector size ({}). Use {} instead.\n".format(
                        n_d, len(emb_vals[0]), len(emb_vals[0])
                    ))
                n_d = len(emb_vals[0])

            say("{} pre-trained embeddings loaded.\n".format(len(emb_vals)))

            # random initial word vectors for words not in pre-trained word_vectors
            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    emb_vals.append(random_init((n_d,))*(0.001 if word != oov else 0.0))
                    lst_words.append(word)

            emb_vals = np.vstack(emb_vals)
            self.vocab_map = vocab_map
            self.lst_words = lst_words
            self.emb_vals = emb_vals
        else:
            lst_words = [ ]
            vocab_map = {}
            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    lst_words.append(word)

            self.lst_words = lst_words
            self.vocab_map = vocab_map
            #random initialize word vectors
            emb_vals = random_init((len(self.vocab_map), n_d))
            self.init_end = -1
            self.emb_vals = emb_vals

        if oov is not None and oov is not False:
            assert oov in self.vocab_map, "oov {} not in vocab".format(oov)
            self.oov_tok = oov
            self.oov_id = self.vocab_map[oov]
        else:
            self.oov_tok = None
            self.oov_id = -1

        self.embeddings = emb_vals#create_shared(emb_vals)
        if self.init_end > -1:
            self.embeddings_trainable = self.embeddings[self.init_end:]
        else:
            self.embeddings_trainable = self.embeddings

        # the number of unique words
        self.n_V = len(self.vocab_map)
        self.n_d = n_d

    def map_to_words(self, ids):
        n_V, lst_words = self.n_V, self.lst_words
        return [ lst_words[i] if i < n_V else "<err>" for i in ids ]

    def map_to_ids(self, words, filter_oov=False):
        '''
            map the list of string tokens into a numpy array of integer IDs

            Inputs
            ------

            words           : the list of string tokens
            filter_oov      : whether to remove oov tokens in the returned array


            Outputs
            -------

            return the numpy array of word IDs

        '''
        vocab_map = self.vocab_map
        oov_id = self.oov_id
        if filter_oov:
            not_oov = lambda x: x!=oov_id
            return np.array(
                    filter(not_oov, [ vocab_map.get(x, oov_id) for x in words ]),
                    dtype="int32"
                )
        else:
            return np.array(
                    [ vocab_map.get(x, oov_id) for x in words ],
                    dtype="int32"
                )

    def forward(self, x):
        '''
            Fetch and return the word embeddings given word IDs x

            Inputs
            ------

            x           : an array of integer IDs


            Outputs
            -------

            a matrix of word embeddings
        '''
        return self.embeddings[x]

    @property
    def params(self):
        return [ self.embeddings_trainable ]

    @params.setter
    def params(self, param_list):
        self.embeddings.set_value(param_list[0].get_value())

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

def create_eval_batches(ids_corpus, data, padding_id, pad_left):
    lst = [ ]
    for pid, qids, qlabels in data:
        titles = [ ]
        bodies = [ ]
        for id in [pid]+qids:
            t, b = ids_corpus[id]
            titles.append(t)
            bodies.append(b)
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
