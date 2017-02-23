from __future__ import division
import numpy as np
import scipy as sp
import cPickle as pickle

from collections import Counter

import pdb


def build_vocab(fname):
    vocab=Counter()
    data=pickle.load(open(fname,'rb'))
    for line in data:
        line=str(line)
        #line=line.translate(table,string.punctuation).strip()
        for word in line.lower().split():
            vocab[word]+=1
    return {word:(i,freq) for i,(word,freq) in enumerate(vocab.iteritems())}

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


def build_kernel_similarity(train_embedding,alpha=None,beta=None):
    m=len(train_embedding)
    K=np.zeros((m,m))
    alpha=1.0
    beta=1.0
    sigma=1
    for i in range(0,m):
        for j in range(i,m):
            X=np.array(train_embedding[i])
            Y=np.array(train_embedding[j])
            K_temp=[]
            for k in range(0,X.shape[0]):
                for l in range(0,Y.shape[0]):
                    xny=X[k,:]-Y[l,:]
                    normxny=xny.transpose()*xny
                    sim=np.exp(-normxny/(2*np.power(sigma,2)))  
                    K_temp.append(sim)
            K_temp=np.array(K_temp)
            K_temp=np.sum(K_temp)/(X.shape[0]*Y.shape[0])
            K[i,j]=K_temp
            K[j,i]=K[i,j]
    return K        
            #K[i,j]=sum(K)/(X.shape[1]*Y.shape[1])
    #return K      

vocab=build_vocab('cell_phones_and_accessories_v1.p')
ratings=get_ratings('cell_phones_and_accessories_ratings_v1.p')
vocab_embeddings=pickle.load(open('vocab_embeddings.p','rb'))
train_embedding=train_embeddings(vocab_embeddings,'cell_phones_and_accessories_v1.p')

Xtrain=train_embedding[0:300]

kernels=build_kernel_similarity(Xtrain)

pickle.dump(kernels,open('kernels_train.p','wb'))
pickle.dump(train_embedding,open('data.p','wb'))









