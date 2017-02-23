import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from scipy.spatial.distance import pdist,squareform
from scipy.spatial import distance
from scipy.optimize import minimize



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
    #print 'Done : grad_L_theta_K'
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
    vals=sum(res)
    #print 'Done : grad_K_x_v'
    vals=np.array(vals)
    return vals
 

def grad_L_x_v(v,K,Xtrain,Ytrain,vocab,vocab_embeddings,token_list,rho=None):
    """Do not bother about \/alpha or other hyperparameters """
    rho=1
    vals=[]
    for i in range(0,len(Xtrain)):
            for j in range(0,len(Xtrain)):
                Xtrain_1=Xtrain[i]
                Xtrain_2=Xtrain[j]
                train_tokens_list_1=token_list[i]
                train_tokens_list_2=token_list[j]
 
                grad_k=grad_L_theta_K(Ytrain,K)
                                            
                res_2=grad_K_x_v(v,Xtrain_1,Xtrain_2,vocab,vocab_embeddings,train_tokens_list_1,train_tokens_list_2)
                res_3=rho*vocab_embeddings[v]
                res=grad_k[i,j]*res_2 - res_3
                vals.append(res)
    #print 'Done :grad_L_x_v'
    return np.array(sum(vals))


