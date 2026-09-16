# %%

import numpy as np
import pandas as pd
import scanpy as sc

import time
import argparse
import cProfile
import json
import platform
import os
from contextlib import contextmanager
from pathlib import Path

# %%
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
# sc.settings.n_jobs = int(sys.argv[4])
timings: dict[str, float] = {}


@contextmanager
def timed_section(name: str):
    """Measure one analysis section and print a machine-readable record."""
    started = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - started
        timings[name] = elapsed
        print(f"PROFILE section={name} seconds={elapsed:.6f}")

# %%
parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('--data-dir', type=str, help='Directory containing the dataset subdirectories', default='data')
parser.add_argument('--data-set', type=str, help='Dataset name, which is the subdirectory name', default='pbmc3k')
parser.add_argument('--out-dir', type=str, help='Output directory', required=False, default='data')
parser.add_argument('--num-threads', type=int, help='Number of threads', default=1, required=False)
parser.add_argument('--profile-output', type=str, help='Optional cProfile .prof output path', required=False)

args = parser.parse_args()

datadir = args.data_dir if args.data_dir.endswith('/') else args.data_dir + '/'
dataset = args.data_set 
outdir = args.out_dir if args.out_dir.endswith('/') else args.out_dir + '/'
nthreads = args.num_threads
if nthreads < 1:
    parser.error("--num-threads must be positive")
Path(outdir).mkdir(parents=True, exist_ok=True)
sc.settings.n_jobs = nthreads

print(f"using {sc.settings.n_jobs} threads")


#%%

# I/O
results_file = "/".join([outdir, dataset + '.scanpy.h5ad'])  # the file that will store the analysis results

with timed_section("load"):
    adata = sc.read_10x_mtx(
        "/".join([datadir, dataset, 'filtered_gene_bc_matrices']),
        var_names='gene_symbols',
        # Avoid Scanpy's shared cache when multiple Slurm jobs run concurrently.
        cache=False)

with timed_section("unique_gene_names"):
    adata.var_names_make_unique()
input_shape = list(adata.shape)
input_nnz = int(adata.X.nnz)


# %%
# preprocessing

# basic filtering
with timed_section("filter"):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

#%%
# metric
#adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
#sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# filtering by slicing the AnnData object
#adata = adata[adata.obs.n_genes_by_counts < 2500, :]
#adata = adata[adata.obs.pct_counts_mt < 5, :]


# and normalize to 10K reads per cell
with timed_section("normalize"):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


# %%
# highly variable genes

#sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
with timed_section("highly_variable_genes"):
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

with timed_section("select_hvg"):
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable].copy()


#%%
# regres out effects of total counts per cell an d% mitochondrial genes
#sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
with timed_section("scale"):
    sc.pp.scale(adata)

# %%
# report adata - so we can check ot see if we are comparable to Seurat
# adata.write(results_file)
# adata

# %%
# pca.  parallel via OMP_NUM_THREADS
with timed_section("pca"):
    sc.tl.pca(adata, svd_solver='arpack', n_comps=30)

# adata.write(results_file)
# adata

# %%
# neighborhood graph
with timed_section("neighbors"):
    sc.pp.neighbors(adata, n_pcs=30)

# %% 
# for fixing disconnected clusters or connectivity issues:
#sc.tl.paga(adata)
#sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
#cs.tl.umap(adata, init_pos='paga')


# adata.write(results_file)
# adata


# %%
# clustering  (currently uses leiden,  previously using louvain (like Seurat).)
#sc.tl.leiden(adata)
with timed_section("louvain"):
    sc.tl.louvain(adata, resolution=0.5)
    cluster_key = "louvain"


#%%
# umap
with timed_section("umap"):
    sc.tl.umap(adata, n_components=30)

#%%

# %%
# support t-test, wilcoxon, logistic regression
# find marker genes
with timed_section("rank_genes_groups"):
    if args.profile_output:
        Path(args.profile_output).parent.mkdir(parents=True, exist_ok=True)
        cProfile.runctx(
            "sc.tl.rank_genes_groups(adata, cluster_key, method='wilcoxon', use_raw=True)",
            globals(), locals(), filename=args.profile_output)
        print(f"PROFILE cprofile_file={args.profile_output}")
    else:
        sc.tl.rank_genes_groups(adata, cluster_key, method='wilcoxon', use_raw=True)

with timed_section("write"):
    adata.write(results_file)
metadata = {
    "dataset": dataset, "input_shape": input_shape, "input_nonzero_counts": input_nnz,
    "filtered_shape": list(adata.shape), "raw_genes": adata.raw.n_vars,
    "cluster_key": cluster_key, "clusters": int(adata.obs[cluster_key].nunique()),
    "python": platform.python_version(), "platform": platform.platform(),
    "hostname": platform.node(), "scanpy": sc.__version__, "threads": nthreads,
    "job_id": os.environ.get("SLURM_JOB_ID"), "cache": False,
    "umap_components": 30, "random_seed": 0,
}
(Path(outdir) / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("METADATA " + json.dumps(metadata))
timing_file = Path(outdir) / f"{dataset}.timings.csv"
timing_file.parent.mkdir(parents=True, exist_ok=True)
with timing_file.open("w", encoding="utf-8") as handle:
    handle.write("section,seconds\n")
    for section, seconds in timings.items():
        handle.write(f"{section},{seconds:.6f}\n")
print(f"PROFILE timings_file={timing_file}")
