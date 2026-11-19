from pathlib import Path
import json, numpy as np, torch
from torch.utils.data import DataLoader
from stack.data.training.datasets import SimplifiedMultiDataset, DatasetConfig as PreConfig
from stack.data.finetuning.datasets import MultiDatasetSplittableDataset, DatasetConfig as PostConfig
B=Path(__file__).resolve().parents[1]
P=B.parent/'pie_four_context_reproduction_20261106'
CONTEXTS=['hepg2','jurkat','k562','rpe1']
GENES=json.loads((P/'downloads/official_data/preprocessed/meta.json').read_text())['genes']
STACK_GENES=[g.upper() for g in GENES]
assert len(set(STACK_GENES))==6642

def split_pairs(fold,split):
    d=json.loads((P/'downloads/official_splits/replogle_wdataset/unseen_ctx'/fold/(split+'.json')).read_text())
    return {(c.split('.',1)[1],p) for c,ps in d.items() for p in ps}

def labels_for(ctx):return np.load(B/'data/statistics'/ctx/'cells_metadata.npz')['labels']
def context_for_file(info):return Path(info['path']).parent.name

class ExactPretrain(SimplifiedMultiDataset):
    def __init__(self,fold,split):
        self.fold=fold;self.pairs=split_pairs(fold,split);self.split=split
        configs=[PreConfig(path=str(B/'data/stack_raw'/c),filter_organism=False,gene_name_col='gene_name') for c in CONTEXTS if c!=fold]
        files=[str(Path(c.path)/'data.h5ad') for c in configs]
        super().__init__(configs,target_genes=STACK_GENES,sample_size=128,mode=split,random_state=42,train_files=files,val_files=files,test_files=[])
    def _generate_locality_samples(self):
        self._initialize_rng();self.samples=[]
        self.audit={'declared_pairs':len(self.pairs),'eligible_cells':0,'sampled_cells':0,'held_out_context':self.fold,'selected_pairs_equal_PIE':True}
        selected_pairs=set()
        for i,info in enumerate(self.metadata_cache.file_info):
            ctx=context_for_file(info)
            assert ctx!=self.fold and info['found_genes']==6642
            assert info['gene_mapping']=={k:k for k in range(6642)}
            labels=labels_for(ctx);allowed={p for c,p in self.pairs if c==ctx}
            if self.split=='train':allowed.add('non-targeting')
            ix=np.flatnonzero(np.isin(labels,list(allowed)))
            selected_pairs|={(ctx,str(p)) for p in labels[ix] if p!='non-targeting'}
            self.audit['eligible_cells']+=len(ix)
            for lo in range(0,len(ix)-127,128):
                self.samples.append((i,ix[lo:lo+128]));self.audit['sampled_cells']+=128
        assert selected_pairs==self.pairs
        (B/'reports'/f'stack_pretrain_split_{self.fold}_{self.split}.json').write_text(json.dumps(self.audit,indent=2))
        self.rng.shuffle(self.samples);assert self.samples

class CompletePosttrain(MultiDatasetSplittableDataset):
    """Retain official samples; pad small same-condition groups without dropping PIE pairs."""
    def __getitem__(self,index):
        gid,pert,dtype,indices=self.samples[index]
        if len(np.unique(indices))==128:
            return super().__getitem__(index)
        if self.rng is None:self._initialize_rng()
        # The official replacement dictionary collapses repeated IDs. Use positions
        # for padded groups so the set remains exactly 32 prompt + 96 query cells.
        left=np.asarray(indices,dtype=np.int64).copy();self.rng.shuffle(left)
        right=left.copy()
        right[32:]=self.rng.choice(self.control_pool[gid],96,replace=True)
        identities=self.metadata_cache.get_cell_identities(left)
        assert len(set(identities))==1
        types=torch.tensor([self.identity_to_id_map[s] for s in identities],dtype=torch.long)
        _,first=np.unique(left,return_index=True)
        mask=np.zeros(128,dtype=bool);mask[first]=True
        unique,inverse=np.unique(np.concatenate([left,right]),return_inverse=True)
        data=self.metadata_cache.load_expression_data(unique)[inverse]
        meta={'group_id':str(gid),'original_group_id':str(pert),'dataset_type':dtype,
              'sampled_cell_count':128,'n_kept':32,'n_replaced':96}
        return torch.from_numpy(data[:128]).float(),torch.from_numpy(data[128:]).float(),types,torch.from_numpy(mask),meta

def post_dataset(fold,split):
    configs=[PostConfig(path=str(B/'data/stack_raw'/c),type='drug',filter_organism=False,
        gene_name_col='gene_name',condition_col='gene',cell_line_col='cell_line',control_condition='non-targeting') for c in CONTEXTS if c!=fold]
    ds=CompletePosttrain(configs,target_genes=STACK_GENES,sample_size=128,replacement_ratio=.75,
        intra_file_replacement_prob=1.,min_cells_per_group=128,train_groups=[],val_groups=[],test_groups=[],mode=split,random_state=42,resample=False)
    pairs=split_pairs(fold,split);active=[];declared=set()
    for gid,info in ds.metadata_cache.group_mapping.items():
        ctx=Path(info['path']).name;pert=info['original_id'];assert ctx!=fold
        if (ctx,pert) in pairs or (split=='train' and pert=='non-targeting'):
            active.append(gid)
            if pert!='non-targeting':declared.add((ctx,pert))
    assert declared==pairs
    ds.active_groups=active;ds._generate_samples_by_locality()
    original_groups={s[0] for s in ds.samples}
    global_groups=np.asarray(ds.metadata_cache.global_group_ids)
    controls={}
    for gid,info in ds.metadata_cache.group_mapping.items():
        if info['original_id']=='non-targeting':controls[Path(info['path']).name]=np.flatnonzero(global_groups==gid)
    ds.control_pool={};padded=[]
    for gid in active:
        info=ds.metadata_cache.group_mapping[gid];ctx=Path(info['path']).name
        ds.control_pool[gid]=controls[ctx]
        if gid in original_groups:continue
        ix=np.flatnonzero(global_groups==gid);assert 0<len(ix)<128
        # Include every available original cell at least once, then pad within group.
        selected=np.concatenate([ix,ds.rng.choice(ix,128-len(ix),replace=True)])
        ds.samples.append((gid,info['original_id'],info['dataset_type'],selected))
        padded.append({'context':ctx,'perturbation':info['original_id'],'unique_cells':len(ix)})
    ds.rng.shuffle(ds.samples);assert len(ds)>0
    effective=set()
    for gid,pert,dtype,indices in ds.samples:
        info=ds.metadata_cache.group_mapping[gid];ctx=Path(info['path']).name
        assert (ctx,pert) in pairs or (split=='train' and pert=='non-targeting')
        assert np.all(global_groups[indices]==gid)
        if pert!='non-targeting':effective.add((ctx,pert))
    assert effective==pairs
    for info in ds.metadata_cache.file_info:
        assert info['found_genes']==6642 and info['gene_mapping']=={k:k for k in range(6642)}
    report={'declared_pairs':len(pairs),'effective_perturbed_pairs':len(effective),
        'all_PIE_pairs_included':True,'same_condition_padded_groups':padded,
        'samples':len(ds),'sample_size':128,'held_out_context_excluded':True,
        'all_sampled_pairs_within_PIE_split':True,
        'sampling':'official complete 128-cell blocks; groups with fewer than 128 cells include all originals and sample extra within the same condition; 32 prompt cells and 96 matched-context control query cells; duplicates masked exactly as in official code'}
    (B/'reports'/f'stack_posttrain_split_{fold}_{split}.json').write_text(json.dumps(report,indent=2))
    return ds

def loaders(fold,stage,batch_size,workers=2):
    train=ExactPretrain(fold,'train') if stage=='pretrain' else post_dataset(fold,'train')
    val=ExactPretrain(fold,'val') if stage=='pretrain' else post_dataset(fold,'val')
    return (DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=workers,pin_memory=True,persistent_workers=workers>0,drop_last=True),
        DataLoader(val,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True,persistent_workers=workers>0))
