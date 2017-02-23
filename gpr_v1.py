from __future__ import division
import autograd.numpy as np
from autograd import grad
import scipy as sp
import cPickle as pickle

from sklearn.metrics.pairwise import rbf_kernel

from scipy.spatial.distance import pdist,squareform
from scipy.spatial import distance

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




#@profile
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
    term_3=sum(embeddings)
    term_1=-0.5*(np.dot(np.dot(targets.reshape(1,-1),K),targets.reshape(1,-1).T))
    term_2=-0.5*np.linalg.slogdet(K)[1]   
    return term_1+term_2+term_3

def grad_L_theta_K(Ytrain,K):
    """Returns gradient of L w.r.t K"""
    term_1=0.5*np.linalg.inv(K)*Ytrain*Ytrain.T*np.linalg.inv(K)
    term_2=0.5*np.linalg.inv(K)
    return term_1-term_2


def grad_K_x_v(v,Xtrain_1,Xtrain_2,vocab,vocab_embeddings,train_tokens_list_1,train_tokens_list_2,alpha=None,psi=None):
    """This is the eq 15. This is the gradient of the K with the specific word """
    word2id={word:i for i,(word,_) in enumerate(vocab.iteritems())}
    alpha=1
    psi=0.5
    res=[]
    m=len(train_tokens_list_1)
    n=len(train_tokens_list_2)
    for i in range(0,m):
        for j in range(0,n):
            if word2id[v] in train_tokens_list_1:
               A=rbf_kernel(Xtrain_1[i].reshape(1,-1),Xtrain_2[j].reshape(1,-1))       
               diff=psi*Xtrain_1[i]-vocab_embeddings[v]
               ress=alpha/(m*n)*A*diff
               res.append(ress)
            else:
                A=rbf_kernel(Xtrain_1[i].reshape(1,-1),Xtrain_2[j].reshape(1,-1))
                diff=psi*Xtrain_2[j]-vocab_embeddings[v]
                ress=alpha/(m*n)*A*diff
                res.append(ress)
    vals=sum(res)
    vals=np.array(vals)
    return vals
    
def grad_L_x_v(v,K,Xtrain,Ytrain,vocab,vocab_embeddings,token_list,rho=None):
    """Do not bother about \/alpha or other hyperparameters """
    rho=1
    vals=[]
    for i in range(0,len(Xtrain)):
            for j in range(i,len(Xtrain)):
                Xtrain_1=Xtrain[i]
                Xtrain_2=Xtrain[j]
                train_tokens_list_1=token_list[i]
                train_tokens_list_2=token_list[j]
 
                grad_k=grad_L_theta_K(Ytrain,K)
                                            
                res_2=grad_K_x_v(v,Xtrain_1,Xtrain_2,vocab,vocab_embeddings,train_tokens_list_1,train_tokens_list_2)
                res_3=rho*vocab_embeddings[v]
                res=grad_k[i,j]*res_2 - res_3
                vals.append(res)
    return np.array(sum(vals))             


#Will add more and make it better as we move on
def grad_desc(K,data,vocab,Ytrain,vocab_embeddings,Xtrain,train_tokens,max_iter=100000):
    converged=False
    iter=0
    #initial error
    ll=training_loss(K,Ytrain,vocab_embeddings)
    print 'Initial loss :', ll , '!!!'
    #pdb.set_trace()
    while not converged:
          for v in vocab_embeddings.iterkeys():
              #pdb.set_trace()
              grad_v=grad_L_x_v(v,K,Xtrain,Ytrain,vocab,vocab_embeddings,train_tokens)
              #temp0=grad_v-alpha*grad_v  
              vocab_embeddings[v]=vocab_embeddings[v]-alpha*grad_v
          Xtrain=update_train_embeddings(vocab_embeddings,data)
          K=build_kernel_similarity(Xtrain) 
          new_ll=training_loss(K,Ytrain,vocab_embeddings)
          print 'New loss after training :', new_ll, '!!!'
          if np.abs(ll-new_ll) <= ep:
             print 'Converged,iterations :', iter, '!!!'
             converged=True
          ll=new_ll
          iter+=1
          if iter == max_iter:
             print 'Max iterations exceeded'
             converged=True
    return vocab_embeddings,K    



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
#K_init=K_init[0:3,0:3]

#print training_loss(K_init,Ytrain,vocab_embeddings).shape

vocab_embeddings_new,K=grad_desc(K_init,data,vocab,Ytrain,vocab_embeddings,Xtrain,train_tokens,max_iter=1000)


pickle.dump(vocab_embeddings_new,open('vocab_embeddings_new.p','wb'))
pickle.dump(K,open('K_updated.p','wb'))




