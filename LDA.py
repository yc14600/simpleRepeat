
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:

import numpy as np
import six
from time import time
from scipy.special import digamma


# In[3]:

import tensorflow as tf


# In[4]:

import edward as ed


# In[5]:

from edward.models import Multinomial, Dirichlet, Categorical,RandomVariable
from edward.util import get_session, get_variables


# In[6]:

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# In[7]:

print ("loading dataset")
t0 = time()
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
#data_samples = dataset.data[:n_samples]
print("done in %0.3fs." % (time() - t0))



# In[68]:

D = 5000 #number of documents
docs = dataset.data[:D]


# In[69]:

V = 1000 #number of words in the vocabulary
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.8, min_df=2,
                                max_features=V,
                                stop_words='english')
t0 = time()
doc_word_mtrx = tf_vectorizer.fit_transform(docs)
print("done in %0.3fs." % (time() - t0))


# In[63]:

K = 10 #number of topics

# In[74]:

ITER = 20
MAXITR = 10
tol = 1e-5
forget_rate = -0.7
delay = 10
eta = 1./K
alpha = 1./K


# In[75]:

lam=np.zeros((K,V)) + np.random.randint(1,6,(K,V))
gamma = np.ones((D,K))
a = [1/K]*K
phi = np.random.dirichlet(a,D)


# In[76]:

for itr in range(ITER):
    if (itr+1) % 10 == 0:
        print ("iter: %d" % (itr+1))
    for d in range(D):
        ids = doc_word_mtrx.indices[doc_word_mtrx.indptr[d]:doc_word_mtrx.indptr[d+1]]
        LSUM=np.repeat(digamma(np.sum(lam,1)).reshape(-1,1),len(ids),axis=1)
        cnts = doc_word_mtrx.data[doc_word_mtrx.indptr[d]:doc_word_mtrx.indptr[d+1]]
        last_result = gamma[d].copy()
        for iiter in range(MAXITR):
            phi_dn = np.exp(digamma(np.repeat(gamma[d].reshape(-1,1),len(ids),axis=1)) +                        digamma(lam[:,ids]) - LSUM)*cnts
            #print (gamma[d])
            #print (lam[:,ids])
            phi_dn /= np.sum(phi_dn,axis=0) #normalize
            phi[d] = np.sum(phi_dn,axis=1)
            
            #print (phi[d])
            gamma[d] = phi[d]+alpha
            #print (gamma[d])
            #print (gamma[d]-last_result)
            if np.abs(np.mean(gamma[d]-last_result)) < tol:
                #print ("params for doc %d have converged at iiter %d" % (d, iiter))
                break
            last_result = gamma[d].copy()
        hlam = np.zeros((lam.shape))
        hlam[:,ids]+=phi_dn
        hlam *= D
        hlam += eta
        
        rao = np.power(itr+delay,forget_rate)
        lam = (1-rao)*lam + rao*hlam


# In[78]:
m = 30
for k in range(K):
    wds=np.argsort(lam[k,:])[-m:]
    print ("topic %d top %d words:" % (k, m))
    print (np.array(tf_vectorizer.get_feature_names())[wds])




