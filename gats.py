import numpy as np
import pandas as pd
import sklearn


def combine_same_labels(data, targets, cat_col, n):
    ind=np.where(targets==1)[0]
    D1=data[ind,:]

    C1=D1[:, ~X_train.columns.isin(cat_var)]
    B1=D1[:, X_train.columns.isin(cat_var)].astype('int32')


    L1=targets[ind]  
    z=0  
    if ind.shape[0]>n:
        z=1

        D_1=[] 
        for i in range(0,len(D1)):

            idx = np.random.randint(len(D1),size=n)
            
            d=C1[idx,:]        
            a=np.random.rand(n,)
            a=np.exp(a)/sum(np.exp(a))
            for j in range(len(d)):
                d[j,:]=d[j,:]*a[j]
            d=np.sum(d,axis=0)
            
            binary=B1[idx,:]
            b=[]
            for k in range(0,B1.shape[1]):
                counts = np.bincount(binary[:,k])
                b.append(np.argmax(counts))
            
            b=np.array(b)
            d=np.concatenate((d,b))
            D_1.append(d)
        D_1=np.vstack(D_1)


    ind=np.where(targets==0)[0]
    D=data[ind,:]

    C=D[:, ~X_train.columns.isin(cat_var)]
    B=D[:, X_train.columns.isin(cat_var)].astype('int32')

    L0=targets[ind]

    D_0=[] 
    for i in range(0,len(D)):
        idx = np.random.randint(len(D),size=n)
        d=C[idx,:]
        a=np.random.rand(n,)
        a=np.exp(a)/sum(np.exp(a))

        for j in range(len(d)):
            d[j,:]=d[j,:]*a[j]


        d=np.sum(d,axis=0)
        
        binary=B[idx,:]
        b=[]
        for k in range(0,B.shape[1]):
            counts = np.bincount(binary[:,k])
            b.append(np.argmax(counts))
            
        b=np.array(b)        
        d=np.concatenate((d,b))
        D_0.append(d)
    D_0=np.vstack(D_0)
    if z==0:
        D_1=D1
    D=np.concatenate((D_0,D_1))
    L=np.concatenate((L0,L1))
    return D,L

def combine_mixed_labels(data, targets, cat_col, n, q):
    # negative samples
    ind=np.where(targets==0)[0]
    D_0=data[ind,:]

    C_0=D_0[:, ~X_train.columns.isin(cat_var)]
    B_0=D_0[:, X_train.columns.isin(cat_var)].astype('int32')

    L0=targets[ind]

    #  positive samples 
    ind=np.where(targets==1)[0]
    D_1=data[ind,:]

    C_1=D_1[:, ~X_train.columns.isin(cat_var)]
    B_1=D_1[:, X_train.columns.isin(cat_var)].astype('int32')

    L1=targets[ind]
    
    # mixed examples pos
    Mixed1=[] 

    for i in range(int(len(D_1))):
        idx = np.random.randint(len(D_1),size=int(n*(1-q)))
        d1=C_1[idx,:]        
        b1=B_1[idx,:]

        idx = np.random.randint(len(D_0),size=int(n*q))
        d0=C_0[idx,:]
        b0=B_0[idx,:] 

        d=np.concatenate((d0,d1))
        a=np.random.rand(len(d),)
        a=np.exp(a)/sum(np.exp(a))


        for j in range(len(d)):
            d[j,:]=d[j,:]*a[j]
        d=np.sum(d,axis=0)

        binary=np.concatenate((b0,b1))

        b=[]
        for k in range(0,B_1.shape[1]):
            counts = np.bincount(binary[:,k])
            b.append(np.argmax(counts))
            
        b=np.array(b)     
        d=np.concatenate((d,b))

        Mixed1.append(d)
    
    Mixed1=np.vstack(Mixed1)
    LM1=np.ones(len(Mixed1,))
    
    # mixed examples neg
    Mixed2=[] 

    for i in range(int(len(D_0))):
        idx = np.random.randint(len(D_1),size=int(n*(q)))
        d1=C_1[idx,:]        
        b1=B_1[idx,:]

        idx = np.random.randint(len(D_0),size=int(n*(1-q)))
        d0=C_0[idx,:]
        b0=B_0[idx,:] 

        d=np.concatenate((d0,d1))
        a=np.random.rand(len(d),)
        a=np.exp(a)/sum(np.exp(a))


        for j in range(len(d)):
            d[j,:]=d[j,:]*a[j]
        d=np.sum(d,axis=0)

        binary=np.concatenate((b0,b1))

        b=[]
        for k in range(0,B_1.shape[1]):
            counts = np.bincount(binary[:,k])
            b.append(np.argmax(counts))
            
        b=np.array(b)     
        d=np.concatenate((d,b))

        Mixed2.append(d)
    
    Mixed2=np.vstack(Mixed2)
    LM2=np.zeros(len(Mixed2,))

    D=np.concatenate((Mixed1, Mixed2))
    
    L=np.concatenate((LM1,LM2))
    
    return D,L