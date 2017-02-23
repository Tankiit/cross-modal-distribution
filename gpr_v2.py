from __future__ import division
import autograd.numpy as np
from autograd import grad
import scipy as sp
import cPickle as pickle

from sklearn.metrics.pairwise import rbf_kernel

from scipy.spatial.distance import pdist,squareform
from scipy.spatial import distance
from scipy.optimize import minimize

from grad_desc import *

from collections import Counter
#import pdb




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


def update_train_embeddings(vocab_embeddings,data):
    train_corpus=data
    train_list=[]
    for _,line in enumerate(train_corpus):
        tokens=line.lower().split()
        token_ids=[vocab_embeddings[word] for word in tokens]
        train_list.append(token_ids)
    return train_list



def convert_np_format(train_embeddings,ratings):
    #X=np.array(train_embeddings).reshape(len(train_embeddings),1)
    #Y=np.array(ratings).reshape(len(ratings),1)
    X_new=[]
    Y_new=[]
    for i in range(0,len(train_embeddings)):
        X_new.append(np.array(train_embeddings[i]))
    for i in range(0,len(ratings)):
        Y_new.append(np.array(ratings[i]))
    X_new=np.array(X_new)
    Y_new=np.array(Y_new)
    return X_new,Y_new


   
#Will add more and make it better as we move on

def grad_desc(K,data,vocab,Ytrain,vocab_embeddings,Xtrain,train_tokens,max_iter=10):
    converged=False
    iter=0
    #initial error
    ll=training_loss(K,Ytrain,vocab_embeddings)
    print 'Initial loss :', ll , '!!!'
    #pdb.set_trace()
    while not converged:
          for v in vocab_embeddings.iterkeys():
              #pdb.set_trace()
              print 'For word:'+v
              grad_v=grad_L_x_v(v,K,Xtrain,Ytrain,vocab,vocab_embeddings,train_tokens)
              #temp0=grad_v-alpha*grad_v  
              vocab_embeddings[v]=vocab_embeddings[v]-alpha*grad_v
          Xtrain=update_train_embeddings(vocab_embeddings,data)
          K=build_kernel_similarity(Xtrain) 
          new_ll=training_loss(K,Ytrain,vocab_embeddings)
          print 'New loss after training :', new_ll, '!!!'
          #if np.abs(ll-new_ll) <= ep:
          #   print 'Converged,iterations :', iter, '!!!'
          #   converged=True
          ll=new_ll
          iter+=1
          if iter == max_iter:
             print 'Max iterations exceeded'
             converged=True
    return vocab_embeddings,K    


data=pickle.load(open('cell_phones_and_accessories_v1.p','rb'))
vocab=build_vocab('cell_phones_and_accessories_v1.p')
ratings=get_ratings('cell_phones_and_accessories_ratings_v1.p')
vocab_embeddings=pickle.load(open('vocab_embeddings.p','rb'))
train_embedding=train_embeddings(vocab_embeddings,'cell_phones_and_accessories_v1.p')

#The token list contains the id of the tokens
token_list=iterate_corpus(vocab,'cell_phones_and_accessories_v1.p')

X,Y=convert_np_format(train_embedding,ratings)

Xtrain=X[0:300]
Ytrain=Y[0:300]
train_tokens=token_list[0:300]

#Xtrain_new=X[0:3]
#Ytrain_new=Y[0:3]
#train_tokens_new=token_list[0:3]
alpha=0.5
ep=0.01

#print training_loss(K,Ytrain,vocab_embeddings)

K_init=build_kernel_similarity(Xtrain)


vocab_embeddings_new,K=grad_desc(K_init,data,vocab,Ytrain,vocab_embeddings,Xtrain,train_tokens,max_iter=10)





