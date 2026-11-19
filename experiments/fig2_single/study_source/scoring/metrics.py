import numpy as np

def auprc(y,s):
 """Davis-Goadrich nonlinear TP/FP interpolation, grouped tied scores."""
 if y.sum()==0 or y.sum()==len(y):return np.nan
 ix=np.argsort(-s,kind='stable');s=s[ix];y=y[ix].astype(float);ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
 tp=np.cumsum(y)[ends];fp=ends+1-tp;a=np.r_[0,tp[:-1]];b=np.r_[0,fp[:-1]];dt=tp-a;df=fp-b;ok=dt>0
 a=a[ok];b=b[ok];dt=dt[ok];df=df[ok];k=1+df/dt;v=dt/k;positive=(a+b)>0
 v[positive]+=(a[positive]-(a[positive]+b[positive])/k[positive])/k[positive]*np.log1p(k[positive]*dt[positive]/(a[positive]+b[positive]))
 return float(v.sum()/y.sum())

def curve(y,s):
 ix=np.argsort(-s,kind='stable');s=s[ix];y=y[ix];ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1];tp=np.cumsum(y)[ends];fp=ends+1-tp
 out=[];a=b=0.
 for t,f in zip(tp,fp):
  if t>a:
   tt=np.linspace(a,t,8);ff=b+(f-b)*(tt-a)/(t-a);pre=np.divide(tt,tt+ff,out=np.ones_like(tt)*(t/(t+f)),where=(tt+ff)>0)
   out.extend(zip(tt/y.sum(),pre))
  elif out:out.append((t/y.sum(),t/(t+f)))
  a,b=t,f
 return np.array(out)