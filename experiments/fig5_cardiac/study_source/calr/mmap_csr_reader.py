"""Low-memory reader for the existing CSR .npz cache. No analysis is run here.

Example:
    reader = open_csr_npz_mmap(source_npz, new_experiment / 'csr_unpacked')
    input_X = reader.take_rows(input_row_indices)

The compressed members are streamed to disk. Only the requested output CSR is
allocated in RAM; a full source CSR is never constructed. The unpack directory
must be in the new experiment, not alongside the read-only source cache.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import zipfile
import numpy as np


class CSRMemmap:
    def __init__(self, folder):
        folder = Path(folder)
        self.data = np.load(folder/'data.npy', mmap_mode='r', allow_pickle=False)
        self.indices = np.load(folder/'indices.npy', mmap_mode='r', allow_pickle=False)
        self.indptr = np.load(folder/'indptr.npy', mmap_mode='r', allow_pickle=False)
        self.shape = tuple(int(x) for x in np.load(folder/'shape.npy', allow_pickle=False))
        fmt = np.load(folder/'format.npy', allow_pickle=False).item()
        assert fmt in (b'csr','csr'), ('Expected CSR format',fmt)
        assert len(self.shape)==2 and self.indptr.shape==(self.shape[0]+1,)
        assert self.data.ndim==self.indices.ndim==1 and len(self.data)==len(self.indices)
        assert int(self.indptr[0])==0 and int(self.indptr[-1])==len(self.data)
        assert np.all(self.indptr[1:]>=self.indptr[:-1])

    def take_rows(self, rows):
        """Preserve requested row order, all columns, values and explicit zeros."""
        from scipy.sparse import csr_matrix
        rows=np.asarray(rows,dtype=np.int64)
        assert rows.ndim==1 and np.all((rows>=0)&(rows<self.shape[0]))
        lengths=self.indptr[rows+1]-self.indptr[rows]
        nnz=int(lengths.sum())
        index_type=np.int32 if max(nnz,self.shape[1])<np.iinfo(np.int32).max else np.int64
        indptr=np.empty(len(rows)+1,dtype=index_type);indptr[0]=0
        np.cumsum(lengths,out=indptr[1:])
        data=np.empty(nnz,dtype=self.data.dtype)
        indices=np.empty(nnz,dtype=index_type)
        for k,row in enumerate(rows):
            start,end=int(self.indptr[row]),int(self.indptr[row+1])
            dest_start,dest_end=int(indptr[k]),int(indptr[k+1])
            data[dest_start:dest_end]=self.data[start:end]
            indices[dest_start:dest_end]=self.indices[start:end]
        return csr_matrix((data,indices,indptr),shape=(len(rows),self.shape[1]),copy=False)


def open_csr_npz_mmap(source, folder):
    source,folder=Path(source),Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    # Files are streamed, never read wholesale into Python bytes or np.load(npz).
    members=['data.npy','indices.npy','indptr.npy','shape.npy','format.npy']
    record={}
    with zipfile.ZipFile(source) as archive:
        assert set(members)<=set(archive.namelist())
        for member in members:
            dest=folder/member
            temp=folder/(member+'.partial')
            digest=hashlib.sha256()
            with archive.open(member) as src,temp.open('wb') as dst:
                while True:
                    block=src.read(1<<20)
                    if not block:break
                    dst.write(block);digest.update(block)
            assert temp.stat().st_size==archive.getinfo(member).file_size
            os.replace(temp,dest)
            record[member]={'npy_size':dest.stat().st_size,'sha256':digest.hexdigest()}
    (folder/'UNPACKED.json').write_text(json.dumps({'source':str(source),'members':record},indent=2)+'\n')
    return CSRMemmap(folder)
