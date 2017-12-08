import sys
import gzip
import random
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()

#default_rng = np.random.RandomState(random.randint(0,9999))
def random_init(size, rng=None, rng_type=None):
    '''
    Return initial parameter values of the specified size

    Inputs:
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

    return vals

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

        Inputs:
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
                if word in vocab_map:
                    print word, vocab_map[word]
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

        self.embeddings = emb_vals
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

            Inputs:
            words           : the list of string tokens
            filter_oov      : whether to remove oov tokens in the returned array

            Outputs:
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

            Inputs:
            x           : an array of integer IDs

            Outputs:
            a matrix of word embeddings
        '''
        return self.embeddings[x]

    @property
    def params(self):
        return [ self.embeddings_trainable ]

    @params.setter
    def params(self, param_list):
        self.embeddings.set_value(param_list[0].get_value())