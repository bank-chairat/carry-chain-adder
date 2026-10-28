import numpy as np
n=8
def bits(x,k): return ((x[:,None]>>np.arange(k))&1).astype(np.float32)
A,B=np.meshgrid(np.arange(256),np.arange(256));A=A.ravel();B=B.ravel()
X=np.concatenate([bits(A,n),bits(B,n)],1)
T={"p4":((A^B)>>4)&1,"c4":(((A&15)+(B&15))>>4)&1,"p3":((A^B)>>3)&1}
for s in range(2):
    perm=np.random.default_rng(500+s).permutation(len(A));tr,te=perm[:32768],perm[32768:]
    a=np.c_[X[tr],np.ones(len(tr))];b=np.c_[X[te],np.ones(len(te))]
    G=np.linalg.inv(a.T@a+np.eye(17))
    print(s,{k:(round(float(((b@(G@(a.T@(2.0*y[tr]-1)))>0).astype(int)==y[te]).mean()),3),round(float(y.mean()),3)) for k,y in T.items()})
