from __future__ import division
import autograd.numpy as np
from autograd import grad
import scipy as sp
import cPickle as pickle

from sklearn.metrics.pairwise import rbf_kernel

from scipy.spatial.distance import pdist,squareform
from scipy.spatial import distance

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

def build_kernel_similarity(train_embedding,alpha=None,beta=None):
    m=len(train_embedding)
    K=np.zeros((m,m))
    alpha=1.0
    beta=1.0
    sigma=1
    for i in range(0,m):
        for j in range(i,m):
            X=train_embedding[i]
            Y=train_embedding[j]
            A=rbf_kernel(X,Y)
            if i==j:
               sums=sum(sum(A))/(len(X)*len(X))
               K[i,j]=(1/alpha)*sums+(1/beta)
            else:
                K[i,j]=(1/alpha)*sums
    return K                   

def training_loss(K,targets,vocab_embeddings,rho=None):
    embeddings=[]
    for i in vocab_embeddings.iterkeys():
        embeddings.append(np.linalg.norm(vocab_embeddings[i]))
    term_3=sum(embeddings)/len(embeddings)
    term_1=-0.5*targets.T*K*targets
    term_2=-0.5*np.linalg.slogdet(K)[1]
    #return term_1+term_2+term_3
    return term_1





vocab=build_vocab('cell_phones_and_accessories_v1.p')
ratings=get_ratings('cell_phones_and_accessories_ratings_v1.p')



vocab_embeddings=pickle.load(open('vocab_embeddings.p','rb'))
train_embedding=train_embeddings(vocab_embeddings,'cell_phones_and_accessories_v1.p')

token_list=iterate_corpus(vocab,'cell_phones_and_accessories_v1.p')

X,Y=convert_np_format(train_embedding,ratings)


Xtrain=X[0:300]
Ytrain=Y[0:300]

#print Ytrain.shape
K=build_kernel_similarity(Xtrain)
#print training_loss(K,Ytrain,vocab_embeddings)
#print Ytrain.reshape(1,-1).shape

#prod_1=Ytrain.reshape(1,-1).T*np.linalg.inv(K)

#print 0.5*np.linalg.norm(np.dot(np.dot(Ytrain.reshape(1,-1),K),Ytrain.reshape(1,-1).T))
#print prod_1.shape


#*Ytrain.reshape(1,-1).T




