# C-99a: exhaustive check (all 65,536 pairs) of identities stated in the third returned note (INTAKE 5). Nothing here comes from the note's code.
import numpy as np, math
n=8; A,B=np.meshgrid(np.arange(256),np.arange(256)); A=A.ravel(); B=B.ravel()
bit=lambda x,i:(x>>i)&1; sp=lambda b:1-2*b.astype(np.int64)
S=A+B; c=lambda i:(((A&((1<<i)-1))+(B&((1<<i)-1)))>>i)&1
F={i:sp(bit(S,i)) for i in range(9)}; H={i:sp(c(i)) for i in range(9)}; Q={i:sp(bit(A^B,i)) for i in range(8)}
E=lambda v:float(np.mean(v)); ok=True
def chk(name,val,exp,tol=1e-12):
    global ok; good=abs(val-exp)<tol; ok&=good; print(("OK  " if good else "FAIL"),name,val,exp)
# table: Q4 and H4 against F0..F8
for j in range(9): chk(f"E[Q4 F{j}]",E(Q[4]*F[j]) if j!=8 else E(Q[4]*F[8]), {4:1/16,8:-1/256}.get(j,0.0))
for j in range(9): chk(f"E[H4 F{j}]",E(H[4]*F[j]), {0:-1/16,1:-1/8,2:-1/4,3:-1/2,8:1/16}.get(j,0.0))
for j in range(8):
    for k in range(j+1,9): chk(f"E[H{k} F{j}]=-2^(j-k)",E(H[k]*F[j]),-2.0**(j-k))
chk("L'(0) r=1 : -E[Q4 F4]/2",-E(Q[4]*F[4])/2,-1/32); chk("L'(0) r=H4",-E(Q[4]*F[4]*H[4])/2,-1/2)
# active chain length law
for i in range(1,9):
    L=np.zeros(len(A),int)
    for k in range(len(A)): pass
    p=bit(A^B,0)  # placeholder to keep arrays
    for ell in range(1,i+1):
        # active carry chain reaching bit i: generate at i-ell (a&b), propagate at i-ell+1..i-1, and then carry out of generating position... c_i=1 with generator at i-ell
        gen=(bit(A,i-ell)&bit(B,i-ell)); prop=np.ones(len(A),int)
        for t in range(i-ell+1,i): prop&=bit(A^B,t)
        chk(f"P(L_{i}={ell})",E(gen&prop),2.0**-(ell+1))
# truncated carry error: k preceding positions only, zero incoming carry beyond
for i in (4,7):
    for k in range(0,i+1):
        low=((1<<i)-1); lo2=(1<<k)-1
        ctrue=c(i); ctr=((((A>>(i-k))&lo2)+((B>>(i-k))&lo2))>>k)&1
        chk(f"trunc err bit{i} k={k}",E(ctrue!=ctr),2.0**-(k+1)-2.0**-(i+1))
# information numbers
h2=lambda p:-p*math.log2(p)-(1-p)*math.log2(1-p)
chk("I(S4;P4) bits",1-h2(15/32),0.00281960,1e-8)
# I(S4;P4) computed directly
s4=bit(S,4); p4=bit(A^B,4); joint=np.zeros((2,2))
for a in (0,1):
    for b in (0,1): joint[a,b]=np.mean((s4==a)&(p4==b))
ms=joint.sum(1); mp=joint.sum(0); I=sum(joint[a,b]*math.log2(joint[a,b]/(ms[a]*mp[b])) for a in (0,1) for b in (0,1) if joint[a,b]>0)
chk("I(S4;P4) direct",I,1-h2(15/32),1e-9)
print("ALL OK" if ok else "SOME FAILED")
