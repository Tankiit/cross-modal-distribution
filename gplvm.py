import theano
from theano import tensor as T

from collections import Counter
import numpy as np
import cPickle as pickle








def build_vocab(fname):
    vocab=Counter()
    data=pickle.load(open(fname,'rb'))
    for line in data:
        line=str(line)
        #line=line.translate(table,string.punctuation).strip()
        for word in line.lower().split():
            vocab[word]+=1
    return {word:(i,freq) for i,(word,freq) in enumerate(vocab.iteritems())}


def read_dataset(fname):
    data=pickle.load(open(fname,'r'))
    return data

def create_dictionary(sentences):
    """For getting word2id"""
    word2id={}
    counter=Counter()
    for sentence in sentences:
        sentence=str(sentence)
        for word in sentence.lower().split():
            counter[word]+=1
    
    for ids,k in enumerate(counter.iterkeys()):
        word2id[k]=ids
    return word2id 


def document2ids(sentences,word2id):
    ids=[]
    """Given a document return the tokenid"""
    corpus=sentences
    for line in corpus:
        tokens=line.lower().split()
        token_ids=[word2id[word] for word in tokens]
        ids.append(token_ids)
    return ids
        



if __name__== "__main__":
   
   #training parameters. Need to fill this
   
   #Read dataset
   fname='cell_phones_and_accessories_v1.p'
   sentences=read_dataset(fname)
   sentences_train=sentences[0:300]
   sentences_test=sentences[301:]   
   word2id=create_dictionary(sentences)
   sentences_train_id=document2ids(sentences_train,word2id)
   sentences_train_id
   
