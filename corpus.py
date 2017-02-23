from gensim import corpora,models
import cPickle as pickle
import string

table=string.maketrans("","")

from collections import Counter

from skl_groups.features import Features
from skl_groups.summaries import BagOfWords
from sklearn.cluster import KMeans


#dicts=corpora.dictionary.Dictionary()


      
def build_vocab(fname):
    vocab=Counter()
    data=pickle.load(open(fname,'rb'))
    for line in data:
        line=str(line)
        #line=line.translate(table,string.punctuation).strip()
        for word in line.lower().split():
            vocab[word]+=1
    return {word:(i,freq) for i,(word,freq) in enumerate(vocab.iteritems())}


def iterate_corpus(vocab,fname):
    #corpus is the corpus iterator
    
    vocab_size=len(vocab)
    word2id=dict((word,i) for word,(i,_) in vocab.iteritems())
    data=pickle.load(open(fname,'rb'))
    corpus=data
    token_list=[]
    for _,line in enumerate(corpus):
        tokens=line.lower().split()
        token_ids=[vocab[word][0] for word in tokens]
        token_list.append(token_ids)
    return token_list


def get_ratings(fname):
    data=pickle.load(open(fname,'rb'))
    ratings=data
    return ratings


def train_embeddings(vocab_embeddings,fname):
    corpus=pickle.load(open(fname,'r'))
    train_corpus=corpus
    train_list=[]
    for _,line in enumerate(train_corpus):
        tokens=line.lower().split()
        token_ids=[vocab_embeddings[word] for word in tokens] 
        train_list.append(token_ids)
    return train_list


def kernel_embedding():



 
vocab=build_vocab('cell_phones_and_accessories_v1.p')
#pickle.dump(vocab,open('vocab.p','wb'))
token_list=iterate_corpus(vocab,'cell_phones_and_accessories_v1.p')

ratings=get_ratings('cell_phones_and_accessories_ratings_v1.p')
vocab_embeddings=pickle.load(open('vocab_embeddings.p','rb'))

train_embedding=train_embeddings(vocab_embeddings,'cell_phones_and_accessories_v1.p')


