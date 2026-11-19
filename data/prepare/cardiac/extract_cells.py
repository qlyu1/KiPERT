"""Cache observed D14 cells for post-freeze expression reproducibility diagnostics.

This script reads the completed expression experiment without changing its
sources or results. It does not fit models or generate simulated cells.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from datetime import datetime, timezone
import gzip
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time
import warnings

import numpy as np
import scipy
from scipy import sparse

EXPECTED_RETAINED = 4347
DEFAULT_DATA = Path('@STUDY@/datasets/Cai2026_PerturbSeq/zenodo_19185135')
TRIPLETS = np.dtype([('row', '<i4'), ('column', '<i4'), ('value', '<f4')])
MEAN_RTOL = 2e-7
MEAN_ATOL = 1e-8


def load_preparer(experiment_root):
    """Import only the frozen input reader, suppressing new __pycache__ files."""
    source = experiment_root / 'src/prepare_expression.py'
    assert source.is_file(), source
    spec = importlib.util.spec_from_file_location('frozen_expression_input_reader', source)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module, source


def extract_triplets(matrix_path, shape, keep, spool_path, chunk_lines):
    """Validate every source entry; spool retained nonzero coordinates to disk.

    The only full cell-by-gene object here is a one-bit duplicate tracker. The
    triplet spool avoids holding all source chunks and concatenated copies.
    """
    genes, all_cells = shape
    cells = int(keep.sum())
    lookup = np.full(all_cells, -1, dtype=np.int32)
    lookup[keep] = np.arange(cells, dtype=np.int32)
    seen = np.zeros((cells * genes + 7) // 8, dtype=np.uint8)
    row_counts = np.zeros(cells, dtype=np.int64)
    entries_read = selected_entries = stored_entries = fractional_entries = 0
    zero_entries = float32_underflows = 0
    maximum = 0.; minimum = np.inf
    max_rounding_absolute = max_rounding_relative = 0.
    start = time.monotonic(); chunks = 0
    with gzip.open(matrix_path, 'rt', encoding='ascii') as source, spool_path.open('wb') as spool:
        header = source.readline().strip()
        assert header.lower().split() == ['%%matrixmarket', 'matrix', 'coordinate', 'real', 'general'], header
        line = source.readline()
        while line and (not line.strip() or line.lstrip().startswith('%')):
            line = source.readline()
        dimensions = tuple(map(int, line.split()))
        assert len(dimensions) == 3 and dimensions[:2] == (genes, all_cells), dimensions
        assert dimensions[2] >= 0
        while True:
            raw = list(itertools.islice(source, chunk_lines))
            if not raw:
                break
            lines = [line for line in raw if line.strip() and not line.lstrip().startswith('%')]
            if not lines:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('error', DeprecationWarning)
                parsed = np.fromstring(''.join(lines), sep=' ', dtype=np.float64)
            assert parsed.size == 3 * len(lines), 'Malformed MatrixMarket entry'
            values = parsed.reshape(-1, 3)
            assert np.isfinite(values).all(), 'Nonfinite MatrixMarket index/value'
            assert np.array_equal(values[:, :2], np.floor(values[:, :2])), 'Noninteger coordinate'
            rr = values[:, 0].astype(np.int64) - 1
            cc = values[:, 1].astype(np.int64) - 1
            vv = values[:, 2]
            assert rr.min() >= 0 and rr.max() < genes, 'Gene coordinate out of bounds'
            assert cc.min() >= 0 and cc.max() < all_cells, 'Cell coordinate out of bounds'
            assert (vv >= 0).all(), 'Negative published expression'
            entries_read += len(vv)
            assert entries_read <= dimensions[2], 'More entries than declared NNZ'
            fractional_entries += int(np.count_nonzero(vv != np.floor(vv)))
            minimum = min(minimum, float(vv.min())); maximum = max(maximum, float(vv.max()))
            selected = keep[cc]
            rr = rr[selected]; cc = lookup[cc[selected]]; vv = vv[selected]
            selected_entries += len(vv)
            if len(vv):
                keys = cc.astype(np.int64) * genes + rr
                assert len(np.unique(keys)) == len(keys), 'Duplicate retained coordinate inside chunk'
                byte = keys >> 3
                bit = np.left_shift(np.uint8(1), (keys & 7).astype(np.uint8))
                assert not np.any(np.bitwise_and(seen[byte], bit)), 'Duplicate retained coordinate across chunks'
                np.bitwise_or.at(seen, byte, bit)
                with np.errstate(over='raise', invalid='raise'):
                    cast = vv.astype(np.float32)
                assert np.isfinite(cast).all(), 'Published expression exceeds float32 storage range'
                rounding = np.abs(cast.astype(np.float64) - vv)
                max_rounding_absolute = max(max_rounding_absolute, float(rounding.max()))
                positive = vv > 0
                if positive.any():
                    max_rounding_relative = max(max_rounding_relative, float((rounding[positive] / vv[positive]).max()))
                zero_entries += int(np.count_nonzero(vv == 0))
                float32_underflows += int(np.count_nonzero(positive & (cast == 0)))
                nonzero = cast != 0
                block = np.empty(int(nonzero.sum()), dtype=TRIPLETS)
                block['row'] = cc[nonzero]
                block['column'] = rr[nonzero]
                block['value'] = cast[nonzero]
                block.tofile(spool)
                row_counts += np.bincount(block['row'], minlength=cells)
                stored_entries += len(block)
            chunks += 1
            if chunks % 100 == 0:
                print('EXTRACT_MATRIX', entries_read, '/', dimensions[2], 'retained_nonzero', stored_entries,
                      'seconds', round(time.monotonic() - start, 1), flush=True)
    assert entries_read == dimensions[2], (entries_read, dimensions[2])
    assert stored_entries == int(row_counts.sum()) <= cells * genes
    assert spool_path.stat().st_size == stored_entries * TRIPLETS.itemsize
    audit = dict(header=header, source_shape=list(dimensions[:2]), declared_entries=int(dimensions[2]),
        parsed_entries=entries_read, selected_entries=selected_entries, stored_nonzeros=stored_entries,
        all_source_entries_finite_nonnegative=True, source_fractional_entries=fractional_entries,
        selected_coordinates_unique=True, unretained_coordinate_uniqueness_not_checked=True,
        selected_explicit_zeros_omitted=zero_entries, float32_underflows_to_zero=float32_underflows,
        float32_max_absolute_rounding=max_rounding_absolute, float32_max_relative_rounding=max_rounding_relative,
        minimum_explicit_value=None if not entries_read else minimum, maximum_explicit_value=maximum,
        duplicate_bitset_bytes=int(seen.nbytes), triplet_spool_bytes=int(spool_path.stat().st_size),
        chunk_lines=chunk_lines, extraction_seconds=time.monotonic() - start)
    return row_counts, audit


def build_csr(spool_path, row_counts, gene_count, chunk_lines):
    """Fill each CSR row directly without a concatenated COO or dense copy."""
    cells = len(row_counts); nnz = int(row_counts.sum())
    assert nnz < np.iinfo(np.int32).max
    indptr = np.r_[0, np.cumsum(row_counts)].astype(np.int32)
    data = np.empty(nnz, dtype=np.float32)
    indices = np.empty(nnz, dtype=np.int32)
    cursor = indptr[:-1].astype(np.int64)
    read = 0
    with spool_path.open('rb') as source:
        while True:
            block = np.fromfile(source, dtype=TRIPLETS, count=chunk_lines)
            if not len(block):
                break
            order = np.argsort(block['row'], kind='stable')
            rows = block['row'][order]
            counts = np.bincount(rows, minlength=cells)
            chunk_starts = np.cumsum(counts) - counts
            offsets = np.arange(len(rows), dtype=np.int64) - np.repeat(chunk_starts, counts)
            destinations = cursor[rows] + offsets
            indices[destinations] = block['column'][order]
            data[destinations] = block['value'][order]
            cursor += counts
            read += len(block)
    assert read == nnz
    np.testing.assert_array_equal(cursor, indptr[1:])
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(cells, gene_count), copy=False)
    matrix.sort_indices()
    matrix.check_format(full_check=True)
    assert matrix.has_canonical_format and matrix.nnz == nnz
    assert matrix.dtype == np.float32 and np.isfinite(matrix.data).all() and (matrix.data > 0).all()
    return matrix


def compare_reference(matrix, genes, metadata, preparer, experiment_root, roles):
    """Compare float32 cache means to the original float64 target statistics."""
    target = metadata.perturb_identity.to_numpy(dtype=str)
    comparisons = []

    def check(name, expected_mean, expected_n, group):
        rows = np.flatnonzero(target == name)
        assert len(rows) == int(expected_n) and len(rows) > 0, (name, len(rows), expected_n)
        # SciPy's float32 CSR axis=0 sum may accumulate in float32 even when
        # dtype=float64 is supplied. Cast the stored entries before reduction.
        actual = np.asarray(matrix[rows].astype(np.float64).sum(axis=0)).ravel() / len(rows)
        expected_mean = np.asarray(expected_mean, dtype=np.float64)
        assert expected_mean.shape == (len(genes),) and np.isfinite(expected_mean).all()
        error = np.abs(actual - expected_mean)
        np.testing.assert_allclose(actual, expected_mean, rtol=MEAN_RTOL, atol=MEAN_ATOL,
                                   err_msg=f'{group} expression mismatch: {name}')
        comparisons.append(dict(target=str(name), group=group, n=len(rows),
            maximum_absolute_error=float(error.max()),
            maximum_tolerance_fraction=float(np.max(error / (MEAN_ATOL + MEAN_RTOL * np.abs(expected_mean))))))

    with np.load(experiment_root / 'prepared/training_expression.npz', allow_pickle=False) as training:
        np.testing.assert_array_equal(training['genes'].astype(str), genes)
        np.testing.assert_array_equal(training['targets'].astype(str), roles['train'])
        means = training['mean']; ns = training['n']
        for i, name in enumerate(roles['train']):
            check(name, means[i], ns[i], 'train')
        check(preparer.NT, training['nt_mean'], training['nt_n'], 'fixed_NT_reference')
    with np.load(experiment_root / 'prepared/evaluation_expression.npz', allow_pickle=False) as evaluation:
        np.testing.assert_array_equal(evaluation['genes'].astype(str), genes)
        np.testing.assert_array_equal(evaluation['targets'].astype(str), roles['query'])
        np.testing.assert_array_equal(evaluation['train_targets'].astype(str), roles['train'])
        means = evaluation['mean']; ns = evaluation['n']
        for i, name in enumerate(roles['query']):
            check(name, means[i], ns[i], 'query')
        for names, counts in [(roles['query'], evaluation['k']), (roles['train'], evaluation['train_k'])]:
            observed = [int(metadata.loc[target == name, 'is_cm'].sum()) for name in names]
            np.testing.assert_array_equal(observed, counts)
    return dict(passed=True, groups_compared=len(comparisons), mean_relative_tolerance=MEAN_RTOL,
        mean_absolute_tolerance=MEAN_ATOL, accumulation_dtype='CSR entries explicitly cast to float64 before sum(axis=0)',
        tolerance_reason='Original moments use float64 values; cache rounds each published nonnegative value once to float32.',
        query_and_train_CM_counts_match_frozen_evaluation=True, comparisons=comparisons)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent,
                        help='Diagnostics root, containing src/ and prepared/')
    parser.add_argument('--experiment-root', type=Path,
                        help='Frozen experiment root; default is diagnostics root parent on server')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--chunk-lines', type=int, default=100000)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    experiment_root = (args.experiment_root or root.parent).resolve()
    assert root != experiment_root, 'Diagnostic outputs must not replace frozen experiment outputs'
    assert args.chunk_lines > 0
    prepared = root / 'prepared'; prepared.mkdir(parents=True, exist_ok=True)
    output_names = ['cells.npz', 'cell_metadata.csv', 'genes.npy', 'inventory.json']
    if not args.overwrite:
        existing = [name for name in output_names if (prepared / name).exists()]
        assert not existing, ('Diagnostic outputs already exist; use --overwrite deliberately', existing)
    preparer, preparer_path = load_preparer(experiment_root)
    protected_paths = [preparer_path, experiment_root / 'src/split.json',
        experiment_root / 'prepared/training_expression.npz',
        experiment_root / 'prepared/evaluation_expression.npz', experiment_root / 'prepared/inventory.json']
    protected_before = {str(path): preparer.file_hashes(path) for path in protected_paths}
    original_inventory = json.loads((experiment_root / 'prepared/inventory.json').read_text())
    roles, _, paths, split_path, genes, barcodes, meta, keep, identity, batches, input_audit = preparer.read_inputs(
        experiment_root, args.data_dir.resolve())
    assert int(keep.sum()) == EXPECTED_RETAINED, (int(keep.sum()), EXPECTED_RETAINED)
    metadata = meta.loc[keep].copy()
    added_columns = {'csr_row', 'matrix_column_0based', 'split_role', 'is_cm'}
    assert not (added_columns & set(metadata.columns)), 'Derived metadata columns already exist'
    metadata.insert(0, 'csr_row', np.arange(len(metadata), dtype=np.int64))
    metadata.insert(1, 'matrix_column_0based', np.flatnonzero(keep))
    role_lookup = {str(name): role for role, names in roles.items() for name in names}
    role_lookup[preparer.NT] = 'fixed_NT_reference'
    metadata['split_role'] = [role_lookup.get(name, 'outside_fixed_split') for name in identity[keep]]
    metadata['is_cm'] = metadata.celltype.eq(preparer.CM_LABEL)
    for name in ['perturb_captured', 'predicted_doublet']:
        metadata[name] = preparer.explicit_boolean(metadata[name], name)
    for name in ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']:
        metadata[name] = metadata[name].astype(np.float64)
    metadata.index.name = 'barcode'
    np.testing.assert_array_equal(metadata.index.to_numpy(dtype=str), barcodes[keep])
    start = time.monotonic()
    checkpoint = prepared / 'unverified_checkpoint'
    checkpoint_manifest = checkpoint / 'checkpoint.json'
    resumed = checkpoint_manifest.is_file()
    with tempfile.TemporaryDirectory(prefix='.extract_cells_', dir=prepared) as temporary:
        temp = Path(temporary); spool = temp / 'triplets.bin'
        if resumed:
            checkpoint_info = json.loads(checkpoint_manifest.read_text())
            assert checkpoint_info['status'] in ['unverified', 'validation_failed']
            assert checkpoint_info['frozen_inputs'] == protected_before, 'Checkpoint frozen inputs differ'
            for name, hashes in checkpoint_info['outputs'].items():
                assert preparer.file_hashes(checkpoint / name) == hashes, ('Checkpoint file changed', name)
            np.testing.assert_array_equal(np.load(checkpoint / 'genes.npy', allow_pickle=False), genes)
            assert (checkpoint / 'cell_metadata.csv').read_text() == metadata.to_csv(index_label='barcode'), 'Checkpoint metadata differ'
            matrix = sparse.load_npz(checkpoint / 'cells.npz')
            assert matrix.format == 'csr' and matrix.dtype == np.float32 and matrix.shape == (len(metadata), len(genes))
            matrix.check_format(full_check=True)
            assert matrix.has_canonical_format and np.isfinite(matrix.data).all() and (matrix.data > 0).all()
            matrix_audit = checkpoint_info['matrix_audit']
            assert matrix.nnz == matrix_audit['stored_nonzeros']
            print('RESUME_UNVERIFIED_CELL_CHECKPOINT', str(checkpoint), flush=True)
        else:
            row_counts, matrix_audit = extract_triplets(paths['matrix'], (len(genes), len(barcodes)), keep,
                                                      spool, args.chunk_lines)
            matrix = build_csr(spool, row_counts, len(genes), args.chunk_lines)
        source_hashes = {name: dict(path=str(path), **preparer.file_hashes(path)) for name, path in paths.items()}
        for name, value in source_hashes.items():
            assert value['sha256'] == original_inventory['source_files'][name]['sha256'], ('Source changed', name)
        protected_after = {str(path): preparer.file_hashes(path) for path in protected_paths}
        assert protected_after == protected_before, 'Frozen input changed during extraction'
        if resumed:
            assert checkpoint_info['source_files'] == source_hashes, 'Checkpoint raw sources differ'
        else:
            # Persist extraction before expression checks, so a reduction or
            # diagnostic failure cannot discard an eight-minute matrix scan.
            sparse.save_npz(temp / 'cells.npz', matrix, compressed=True)
            np.save(temp / 'genes.npy', genes, allow_pickle=False)
            metadata.to_csv(temp / 'cell_metadata.csv', index_label='barcode')
            checkpoint_info = dict(status='unverified', usable_for_analysis=False,
                created_UTC=datetime.now(timezone.utc).isoformat(),
                frozen_inputs=protected_after, source_files=source_hashes, matrix_audit=matrix_audit,
                outputs={name: preparer.file_hashes(temp / name) for name in output_names if name != 'inventory.json'})
            checkpoint.mkdir(exist_ok=True)
            for name in checkpoint_info['outputs']:
                (temp / name).replace(checkpoint / name)
            (temp / 'checkpoint.json').write_text(json.dumps(checkpoint_info, indent=2, allow_nan=False))
            (temp / 'checkpoint.json').replace(checkpoint_manifest)
        try:
            equivalence = compare_reference(matrix, genes, metadata, preparer, experiment_root, roles)
            protected_after = {str(path): preparer.file_hashes(path) for path in protected_paths}
            assert protected_after == protected_before, 'Frozen input changed during validation'
        except Exception as error:
            checkpoint_info.update(status='validation_failed', usable_for_analysis=False,
                validation_error=type(error).__name__ + ': ' + str(error)[:4000],
                failed_UTC=datetime.now(timezone.utc).isoformat())
            (temp / 'checkpoint.json').write_text(json.dumps(checkpoint_info, indent=2, allow_nan=False))
            (temp / 'checkpoint.json').replace(checkpoint_manifest)
            print('UNVERIFIED_CELL_CHECKPOINT_RETAINED', str(checkpoint), flush=True)
            raise
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_bytes = int(peak_rss if sys.platform == 'darwin' else peak_rss * 1024)
        inventory = dict(status='complete', completed_UTC=datetime.now(timezone.utc).isoformat(),
            purpose='Post-freeze diagnostic cache of observed captured+QC D14 cells; no simulated cells and no model retraining.',
            experiment_root=str(experiment_root), diagnostics_root=str(root),
            source_files=source_hashes, frozen_files_unchanged=protected_after,
            shared_input_reader=str(preparer_path), shared_input_reader_writes_suppressed=True,
            input_audit=input_audit, matrix_audit=matrix_audit, reference_comparison=equivalence,
            resumed_unverified_checkpoint=resumed,
            shape=list(matrix.shape), nnz=int(matrix.nnz), dtype=str(matrix.dtype), sparse_format='csr',
            row_order='Original matrix/barcodes order restricted to captured+QC; cell_metadata.csv rows and csr_row match CSR rows.',
            column_order='Original first feature column; genes.npy order matches CSR columns.',
            expression_scale='Published normalized real-valued expression; only float32 rounding, no log transform, inversion or count normalization.',
            metadata_rows=len(metadata), metadata_columns=metadata.columns.tolist(),
            sample_batches=batches.tolist(), sample_batch_is_not_assumed_to_be_independent_animal=True,
            split_counts={str(name): int(n) for name, n in metadata.split_role.value_counts().items()},
            csr_array_bytes=int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes),
            worst_case_dense_CSR_array_bytes=int(len(metadata) * len(genes) * 8 + (len(metadata) + 1) * 4),
            peak_RSS_bytes=peak_rss_bytes,
            memory_method='Disk triplet spool then preallocated float32/int32 CSR arrays; no dense expression or concatenated COO copy.',
            versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
            elapsed_seconds=time.monotonic() - start,
            outputs={name: preparer.file_hashes(checkpoint / name) for name in output_names if name != 'inventory.json'})
        (temp / 'inventory.json').write_text(json.dumps(inventory, indent=2, allow_nan=False))
        for name in output_names:
            ((temp if name == 'inventory.json' else checkpoint) / name).replace(prepared / name)
        checkpoint_manifest.unlink()
        checkpoint.rmdir()
    print('CELL_EXTRACTION_COMPLETE', dict(cells=len(metadata), genes=len(genes), nnz=int(matrix.nnz),
        mean_comparison_passed=True, peak_RSS_GiB=round(peak_rss_bytes / 2 ** 30, 3)), flush=True)


if __name__ == '__main__':
    main()
