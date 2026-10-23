# Hypotheses (written and committed before running the experiments)

## Part 3 — Compression (SIFT)

**Setup.** IVF-PQ on SIFT-1M with nlist fixed at 1024 (the IVF-PQ coarse
quantiser from Part 2), 8 bits per sub-quantiser, and the number of PQ
sub-vectors m ∈ {8, 16, 32, 64} (8, 16, 32, 64 bytes per code, i.e. 16, 8, 4, 2
dimensions per sub-vector). Each m is run twice: plain IVF-PQ, and IVF-PQ
whose top 100 candidates are re-ranked with exact distances on the raw
vectors (FAISS `IndexRefineFlat`, k_factor = 10). nprobe is swept as in Part 2.
Only m and re-ranking change; nlist, training sample and queries stay fixed.

**Hypothesis.**
1. *Recall ceiling of plain IVF-PQ rises with m but never reaches 0.95.* Even
   with large nprobe, the code distances have quantisation error comparable to
   the gap between the 10th and 11th neighbours, so the ranking inside the top
   10 is scrambled. I expect ceilings of roughly 0.3 (m=8), 0.5 (m=16),
   0.75 (m=32), 0.9 (m=64).
2. *Re-ranking the top 100 removes the ceiling for m ≥ 16.* The true top-10
   are almost always inside the PQ top-100 even when their order is wrong, so
   after exact re-scoring recall is limited only by nprobe (the IVF part) and
   reaches ≥ 0.95, like IVF-Flat. For m = 8 the codes are so coarse that some
   true neighbours fall outside the top 100, so the ceiling stays below 0.95.
3. *Speed.* Re-ranking costs 100 extra exact distance computations per
   query (plus random memory reads), which is small next to scanning
   thousands of codes: under 25% slowdown at the same nprobe. At recall 0.95,
   re-ranked IVF-PQ with m = 16 or 32 is faster than IVF-Flat, because
   scanning 16–32-byte codes is cheaper than scanning 512-byte vectors.
4. *Memory.* Re-ranking needs the raw vectors, so bytes per vector goes from
   about m + 8 (code + 64-bit id) to about m + 8 + 512. Unless the raw vectors
   live on disk, re-ranking gives up almost all of PQ's memory saving.

## Part 4 — Intrinsic dimension of the data (synthetic)

**Question.** Does the *intrinsic* dimension of the data, with the ambient
dimension held fixed, decide how hard ANN search is, and does it hurt
graph search (HNSW) and partition search (IVF-Flat) equally?

**Setup.** N = 200,000 base vectors and 10,000 queries, ambient dimension
D = 128 for every setting. Each vector is a Gaussian sample in a random
d_int-dimensional linear subspace of R^128 (variance 1/d_int per direction, so
E‖x‖² = 1 in every setting) plus small isotropic noise (total variance
0.001). Queries come from the same distribution. The property varied is
d_int ∈ {8, 16, 32, 64, 128}; d_int = 128 means no low-dimensional structure
(plain isotropic Gaussian). Control: the d_int = 16 setting is generated a
second time with a different seed and nothing else changed, which shows how
much the numbers move without any change in the property. Indexes: HNSW
(M = 16, efConstruction = 200) and IVF-Flat (nlist = 512); same sweeps and
timing as Part 2. Generator is seeded.

**Hypothesis.**
1. *Relative contrast falls as d_int grows* (mean distance / nearest
   neighbour distance), from well above 2 at d_int = 8 to near 1.2 at 128:
   neighbours become barely closer than everything else.
2. *Both indexes need more work for recall 0.95 as d_int grows, but IVF-Flat
   degrades faster.* In high intrinsic dimension a query's 10 neighbours are
   scattered over many Voronoi cells (most points sit near cell boundaries),
   so the nprobe needed grows towards a large fraction of nlist and IVF
   approaches the cost of exact search: at d_int = 128 I expect nprobe ≥ 25%
   of nlist. HNSW also needs a larger efSearch (greedy routing gets less
   guidance when all distances look alike), but I expect it to stay faster
   than IVF-Flat at recall 0.95 in every setting.
3. *At d_int = 8 both are easy:* recall 0.95 with efSearch ≤ 32 and nprobe ≤ 8.
4. *Control:* the seed replicate of d_int = 16 differs from the original by
   less than 10% in QPS at recall 0.95 and by less than 0.01 in recall at a
   fixed parameter.
