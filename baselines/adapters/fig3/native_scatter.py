"""TxPert sum-only torch_scatter compatibility using native torch.scatter_add_.
Replaces two public-source imports; no network, parameters, data or loss changed.
"""
import torch

def scatter_add(src,index,dim=-1,out=None,dim_size=None):
 dim=dim if dim>=0 else src.dim()+dim
 if index.dim()==1:
  for _ in range(dim):index=index.unsqueeze(0)
 while index.dim()<src.dim():index=index.unsqueeze(-1)
 index=index.expand_as(src)
 if out is None:
  size=list(src.size());size[dim]=dim_size if dim_size is not None else (int(index.max())+1 if index.numel() else 0);out=src.new_zeros(size)
 return out.scatter_add_(dim,index,src)
def scatter(src,index,dim=-1,out=None,dim_size=None,reduce='sum'):
 if reduce not in ['sum','add']:raise NotImplementedError('Audited TxPert requires only sum/add')
 return scatter_add(src,index,dim,out,dim_size)
if __name__=='__main__':
 import json
 torch.manual_seed(17);checks=[]
 for device in ['cpu','cuda']:
  for dim in [0,1,-1]:
   x=torch.randn(7,7,3 if dim!=-1 else 7,device=device,requires_grad=True);idx=torch.tensor([0,2,1,0,2,1,0],device=device);y=scatter_add(x,idx,dim=dim,dim_size=3);axis=dim%x.ndim;ref=torch.stack([x.index_select(axis,torch.nonzero(idx==k).flatten()).sum(axis) for k in range(3)],axis);assert torch.allclose(y,ref,atol=1e-6);y.square().sum().backward();g=x.grad.clone();x.grad=None;ref.square().sum().backward();assert torch.allclose(g,x.grad,atol=1e-5);checks.append({'device':device,'dim':dim,'value_gradient_pass':True})
 print(json.dumps(checks,indent=2))
