import math
import numpy as np
from numba import njit
@njit(cache=True)
def mwu_sorted_reference(y,ref):
 g,m=y.shape;n=ref.shape[1];N=m+n;out=np.ones(g)
 for gene in range(g):
  a=np.sort(y[gene]);b=ref[gene];i=0;j=0;u=0.;ties=0.
  while i<m or j<n:
   if i==m:v=b[j]
   elif j==n:v=a[i]
   else:v=min(a[i],b[j])
   i0=i;j0=j
   while i<m and a[i]==v:i+=1
   while j<n and b[j]==v:j+=1
   ac=i-i0;bc=j-j0;t=ac+bc;u+=ac*(j0+.5*bc);ties+=t*t*t-t
  var=max(m*n/12.*((N+1.)-ties/(N*(N-1.))),np.finfo(np.float64).tiny)
  z=(max(u,m*n-u)-m*n/2.-.5)/math.sqrt(var);out[gene]=min(1.,math.erfc(z/math.sqrt(2.)))
 return out
