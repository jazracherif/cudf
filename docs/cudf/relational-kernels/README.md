# cuDF relational operations: a CUDA source guide

This guide documents the cuDF 26.06 development source at commit
[`e81ad54ed957866c235cf0720fc3ef8986aa39b0`][revision]. Source links are pinned to
that revision; dispatch rules and dependency implementations can change.
The scope is the principal algorithm families, not every dtype or aggregation
specialization.

## 1. Repository map

The repository root for this analysis is `/home/runner/work/cudf/cudf`.
Paths in the table are absolute local paths; the source links below are portable
GitHub permalinks.

| Location | Responsibility |
| --- | --- |
| `/home/runner/work/cudf/cudf/python/cudf` | Python DataFrame and Series API |
| `/home/runner/work/cudf/cudf/python/pylibcudf` | Python bindings to libcudf |
| `/home/runner/work/cudf/cudf/cpp/include/cudf` | Public C++ APIs and device-facing types |
| `/home/runner/work/cudf/cudf/cpp/include/cudf/detail` | Internal APIs, CUDA templates, row operators, and gather |
| `/home/runner/work/cudf/cudf/cpp/src/groupby` | Hash and sort group by |
| `/home/runner/work/cudf/cudf/cpp/src/join` | Hash, sort-merge, conditional, mixed, and specialized joins |
| `/home/runner/work/cudf/cudf/cpp/src/reductions` | Scalar/segmented reduction and prefix scan |
| `/home/runner/work/cudf/cudf/cpp/src/stream_compaction` | Boolean filtering, null filtering, distinct, and unique |
| `/home/runner/work/cudf/cudf/cpp/src/sort` | Sorting permutations, radix/comparator sorting, segmented sorting |
| `/home/runner/work/cudf/cudf/cpp/tests` | C++ correctness tests |
| `/home/runner/work/cudf/cudf/cpp/benchmarks` | Performance benchmarks |

### Reading a call chain

The usual path is **public API -> internal dispatch -> kernel/library primitive
-> device functor -> output materialization**.

Search for `CUDF_KERNEL` as well as `__global__`: under CUDA compilation,
[`CUDF_KERNEL` expands to `__global__ static`][kernel-macro].
`__device__` functions execute on the GPU but are not independently launched.
`CUDF_HOST_DEVICE` functions can also participate in device execution.

Many launches belong to **CUB, Thrust, or cuCollections**, not to a handwritten
cuDF kernel. In those cases, the cuDF device predicate, comparator, or operator
is the right entry point for understanding relational semantics.
RMM supplies memory allocation and stream integration, not the relational algorithm.

## 2. Optimization lens: PMPP Chapter 6

The requested lens is *Programming Massively Parallel Processors*, fifth edition,
Chapter 6, Figure 6.13, "A checklist of optimizations."
The table below reproduces the checklist supplied in the
[PR comment][checklist-comment] and [image][checklist-image]. Its benefits describe
the optimization rationale, not measurements or guarantees for cuDF.

### FIGURE 6.13: A checklist of optimizations

| Category | Optimization | Benefit to Compute Cores | Benefit to Memory | Strategies |
| :--- | :--- | :--- | :--- | :--- |
| **Compute utilization** | Occupancy tuning | More work to hide pipeline latency | More parallel memory accesses to hide DRAM latency | Tune the usage of SM resources such as threads per block, shared memory per block, and registers per thread |
| | Loop unrolling | Fewer branch instructions and more independent instruction sequences with fewer stalls | May enable promoting local arrays to registers to reduce global memory traffic | <ul><li>Performed automatically by the compiler</li><li>Use loops with constant bounds where possible to facilitate the compiler's job</li></ul> |
| | Reducing control divergence | High SIMD efficiency (fewer idle cores during SIMD execution) | - | <ul><li>Rearrange the assignment of threads to work and/or data</li><li>Rearranging the layout of the data</li></ul> |
| **Memory utilization** | Using coalesceable global memory accesses | Fewer pipeline stalls waiting for global memory accesses | Less global memory traffic and better utilization of bursts/cache-lines | <ul><li>Rearranging the mapping of threads to data</li><li>Rearranging the layout of the data (e.g., matrix transposition, converting an array-of-structures to a structure-of-arrays)</li><li>Transfer data between global memory and shared memory in a coalesceable manner and perform irregular accesses in shared memory (e.g., corner turning, packing)</li></ul> |
| | Shared memory tiling | Fewer pipeline stalls waiting for global memory accesses | Less global memory traffic | Place data that is reused within a block in shared memory so that it is transferred between global memory and the SM only once |
| | Register tiling | Fewer pipeline stalls waiting for shared memory accesses | Less shared memory traffic | Place data that is reused within a warp or thread in registers so that it is transferred between shared memory and registers only once |
| | Vector loads and stores | Fewer instructions executed to manage memory accesses | More parallel memory accesses to hide DRAM latency | Assign each thread to load 16 B chunks of contiguous data |
| | Avoiding shared memory bank conflicts | Fewer pipeline stalls waiting for shared memory accesses | Less shared memory traffic and better utilization of shared memory banks | <ul><li>Rearranging the mapping of threads to data</li><li>Rearranging the layout of the data (e.g., padding)</li></ul> |
| | Privatization (covered later) | Fewer pipeline stalls waiting for atomic updates | Less contention and serialization of atomic updates | Apply partial updates to a private copy of the data then update the public copy when done |
| **Synchronization latency** | Warp-level primitives | Reduce block-wide barrier synchronizations | Less global and shared memory traffic | Perform operations requiring barrier synchronization at the warp-level, then consolidate warp-level results at the block-level |
| | Double buffering | Eliminates barriers that enforce false dependences | - | Eliminate false (write-after-read) dependences by using different buffers for the writes and the preceding reads |
| **General** | Thread coarsening | Depends on the overhead of parallelization | Depends on the overhead of parallelization | Assign multiple units of parallelism to each thread in order to reduce the overhead of parallelization when it is incurred unnecessarily |

Each operation below separates **source observations** from **profiling questions**.
No speedup, occupancy level, bandwidth, or bottleneck is claimed to have been
measured. In particular, shared memory does not automatically mean faster execution,
and maximizing occupancy is not itself the objective.

**How to read the mappings.** A shared allocation is not by itself evidence of
**Shared memory tiling**; scalar locals do not establish **Register tiling**.
Two input/output buffers do not establish the false-dependence-removal pattern
of **Double buffering**. Adjacent scalar accesses across lanes are not
**Vector loads and stores**, which concerns each thread's memory instructions.
Likewise, alignment alone does not establish **Avoiding shared memory bank
conflicts**. Confirm generated instructions and access patterns where needed.
Grid-stride loops assign multiple items per thread when inputs exceed the grid,
but do not by themselves prove that a tuned **Thread coarsening** factor was used.

For library-owned kernels, the mappings identify relevant checklist items rather
than assuming that every CUB/Thrust/cuCollections specialization implements them.
Optimizations not evidenced below remain unestablished for the cited paths;
that does not mean they are absent from the compiler output or dependencies.

## 3. API-to-kernel index

Signatures below list the identifying parameters and omit defaults.
`stream` means `rmm::cuda_stream_view`; `mr` means
`rmm::device_async_resource_ref`. Follow the linked declarations/definitions for
complete types and overloads.

| API | Execution entry and device work |
| --- | --- |
| `groupby::aggregate(host_span<aggregation_request const>, stream, mr)` | [Dispatch][groupby-entry] -> hash mapping/atomic aggregation or sort/segmented reduction |
| `groupby::scan(host_span<scan_request const>, stream, mr)` | [Entry][groupby-entry] -> sort-based grouped scan |
| `inner_join/left_join/full_join(table_view const&, table_view const&, null_equality, stream, mr)` | [Wrappers][join-entry] -> reusable hash join -> cuCollections insertion/retrieval |
| `hash_join::inner_join/left_join/full_join(probe, optional<size_t> output_size, stream, mr)` | [Object methods][hash-build] -> [probe/retrieve][hash-retrieve] |
| `sort_merge_join::inner_join/left_join(left, sorted, stream, mr)` | [Declarations][smj-api] -> bounds searches, scans, range expansion |
| `conditional_*_join(left, right, ast::expression const&, ...)` | [Wrappers][conditional-api] -> explicit nested-loop kernels |
| `scan(column_view const&, scan_aggregation const&, scan_type, null_policy, stream, mr)` | [Entry][scan-entry] -> Thrust inclusive/exclusive scan |
| `apply_boolean_mask(table_view const&, column_view const&, stream, mr)` | [Entry/predicate][filter-entry] -> `copy_if` -> gather |
| `reduce(column_view const&, reduce_aggregation const&, data_type, [init,] stream, mr)` | [Declarations][reduction-api] -> typed operator -> CUB reduction |
| `segmented_reduce(values, offsets, agg, output_type, null_policy, [init,] stream, mr)` | [Declarations][reduction-api] -> CUB segmented reduction for corresponding paths |
| `sorted_order/sort(input, column_order, null_precedence, stream, mr)` | [Entry][sort-entry] -> radix sort or comparator sort |
| `sort_by_key(values, keys, column_order, null_precedence, stream, mr)` | [Entry][sort-entry] -> sorted permutation -> gather |

Joins generally return row-index vectors rather than a materialized joined table.
Likewise, `sorted_order` produces a permutation, whereas `sort` materializes values.

## 4. Group by

### Hash aggregation

**Algorithm.** Hash key rows, identify representative rows through a concurrent
set, map input rows to output groups, aggregate values, and finalize the results.
The [set definitions][groupby-types] use cuCollections linear probing.

The optimized path first finds block-local groups in shared memory, then maps
those representatives to the global set. If the observed block cardinality exceeds
the threshold, the [driver falls back][groupby-driver] to global-memory aggregation.
Shared-memory eligibility also depends on aggregation/type and resource availability.

**Kernel and device entry points:**

- [`mapping_indices_kernel<SetRef>`][mapping]: `CUDF_KERNEL`, grid-stride mapping.
- `find_local_mapping`: `__device__`, shared-set insertion and local group indexing.
- `find_global_mapping`: `__device__`, insertion of representatives into the global set.
- [`single_pass_shmem_aggs_kernel`][shared-aggs]: `CUDF_KERNEL`, shared partial aggregation.
- [`compute_single_pass_aggs_*_output_fn::operator()`][global-functors]:
  `__device__`, global-path row/column work.
- [`update_target_element_gmem<Source, Kind>::operator()`][atomic-aggs]:
  `__device__`, aggregation-specific atomic updates.

**Checklist mapping (source observations).**

- **Compute utilization / Occupancy tuning:** the [driver][groupby-driver] sizes
  its grid using occupancy information for both mapping and aggregation kernels
  and checks shared-memory compatibility.
- **Memory utilization / Privatization:** [block-local aggregation][shared-aggs]
  accumulates partial results before global updates. For low cardinality, it
  replicates aggregation locations across threads to reduce contention further.
  This is privatization, not an input-data **Shared memory tiling** claim.
- **Memory utilization / Using coalesceable global memory accesses:** adjacent
  threads traverse adjacent source rows in `compute_pre_aggregations`, enabling
  coalesced reads for fixed-width column data. Group-indexed output remains irregular.
- **General / Thread coarsening:** grid-stride mapping and aggregation reuse
  threads for multiple rows when needed, though no tuned per-thread tile is
  established. Shared aggregation phases still require block-wide barriers.

**Profiling questions.** How do key skew, distinct-group count, and aggregation
width affect atomics, shared-memory pressure, and fallback frequency? Compare the
cost of mapping/materialization against aggregation itself; do not assume the
shared path wins for every workload. For **Avoiding shared memory bank conflicts**,
inspect actual shared-index patterns; replicated counters target atomic contention,
not a demonstrated conflict-free bank layout.

### Sort aggregation and grouped scan

**Algorithm.** Obtain a stable sorting permutation, detect adjacent unequal keys,
construct group offsets/labels, group values, and perform segmented operations.
Already sorted keys and cached helper data can avoid repeated work.
Common reductions use `thrust::reduce_by_key`; grouped scans use
`thrust::inclusive_scan_by_key`.

Implementation: [sort helper][groupby-sort], [reduction functors][groupby-reduce],
and [grouped scan][groupby-scan]. Device work includes
`value_accessor<T>::operator()` and `null_replaced_value_accessor::operator()`;
the device-wide kernels are library-owned.

**Dispatch.** [Hash aggregation is selected][groupby-entry] only for unsorted
keys, no previously established sort helper, and fully supported requests.
Otherwise aggregation uses sorting. `groupby::scan` uses the sort path.

**Checklist mapping (source observations and library boundary).**

- **Memory utilization / Using coalesceable global memory accesses:** grouped
  values and contiguous labels expose sequential ranges to segmented reduction
  and scan. Obtaining the permutation and gathering values adds traffic, with
  indirect source reads; grouping is not a guarantee of coalescing at every stage.
- **Compute utilization / Reducing control divergence:** typed aggregation
  specializations select supported operators. Data-dependent null and key
  comparisons can still diverge; specialization is not a measured SIMD improvement.
- **Warp-level primitives**, **Register tiling**, and **Thread coarsening** within
  `reduce_by_key`/`inclusive_scan_by_key` require inspecting the selected library
  kernels. Device accessors avoid a separate null-replaced column, but iterator
  fusion is not itself proof of any tiling optimization.

**Profiling questions.** How much sorting can be amortized across requests? Are
gathers, nested-key comparisons, or highly unequal segment sizes dominant?

**Study:** [Blelloch][blelloch] for scan/segmentation foundations. This is background,
not a claim that the hash group by implements a named paper.

## 5. Joins

### General hash equijoins

**Algorithm.** Build `(hash, row_index)` records, probe using hash and actual row
equality, count output when necessary, and retrieve matching index pairs.
Full join adds unmatched build rows. The free inner join builds on the smaller
input; left/full join build on the right input.

This revision uses a [`cuco::static_multiset` with double hashing][hash-types],
**not cuckoo hashing**.

Execution: [build/insert][hash-build] -> [size/retrieve and full-join complement][hash-retrieve].
Library calls include `insert`, `insert_if`, `retrieve`, and `retrieve_outer`.
Their principal `__global__` kernels live in cuCollections.
cuDF device entry points include `hasher1::operator()`, `hasher2::operator()`,
and [`pair_equal::operator()` / `primitive_pair_equal::operator()`][join-compare].
Hash equality alone is not enough to establish a match.

**Checklist mapping (source observations and library boundary).**

- **Compute utilization / Reducing control divergence:** [dispatch][join-compare]
  separates primitive and nested row comparison paths. This avoids a universal
  comparator, but probe lengths and match multiplicities remain data-dependent.
- **Memory utilization / Using coalesceable global memory accesses:** row-wise
  hashing can read adjacent fixed-width column elements, while hash-table probing
  and matched-row comparisons access irregular locations. Do not label the whole
  join as coalesced.
- **Occupancy tuning**, **Warp-level primitives**, and **Vector loads and stores**
  in the main insertion/retrieval kernels are library-level questions, not
  established by cuDF's `insert`/`retrieve` calls.

Building the smaller input, reusing build preprocessing, and supplying a known
output size can reduce work. These are additional algorithm/pipeline optimizations,
not evidence of **Thread coarsening** or **Shared memory tiling**.

**Profiling questions.** Examine load factor, collisions, duplicate-key skew,
probe memory access, and output size. Many-to-many output expansion can dominate
even if hash lookup is efficient.

### Distinct, semi, and anti joins

[`distinct_hash_join`][distinct-join] exploits distinct build keys and probes with
`find_async`/`find_if_async`; device key and output functors adapt hash records.
[`mark_join`][mark-join] provides specialized semi/anti execution with explicit
`mark_probe_kernel`, `mark_retrieve_kernel`, `compact_if_kernel`, and
`clear_marks_kernel` entry points.

**Checklist mapping (source observations).**

- **Synchronization latency / Warp-level primitives:** in [mark join][mark-join],
  `cg::reduce` combines per-thread mark counts at warp level before block/global
  updates. Retrieval/compaction use warp synchronization and broadcast reservations.
- **Memory utilization / Privatization:** thread-local counts and a block-local
  counter reduce global counter updates. Warp-local output buffers batch index
  writes; global hash-slot marking still requires atomics.
- **Memory utilization / Using coalesceable global memory accesses:** retrieval
  packs indices into shared buffers, then lanes flush consecutive output positions.
  This is the checklist's packing strategy, not reused-input tiling.
- **Compute utilization / Loop unrolling:** `#pragma unroll` explicitly requests
  unrolling the constant-bounded full-buffer flush loops in retrieval/compaction.
- **Compute utilization / Occupancy tuning:** mark-probe launch sizing queries
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

These explicit observations apply to mark join, not automatically to
`distinct_hash_join`'s library-owned find kernels. Existence-based execution
avoids pair materialization in both families, but is not itself a tiling mechanism.

**Profiling questions.** Compare hit rate, duplicates, mark contention, and the
cost of output compaction rather than extrapolating from general join throughput.

### Sort-merge join

Construct `sort_merge_join(right, is_right_sorted, compare_nulls, stream)`,
then call its [inner/left probe methods][smj-api].

**Algorithm.** Prepare sorted row orders, perform parallel lower/upper-bound
searches for equal ranges, scan match counts into offsets, and expand ranges
into index pairs. The [matching implementation][smj-code] is not simply a single
linear two-pointer merge.

**Checklist mapping (source observations and library boundary).**

- **Memory utilization / Using coalesceable global memory accesses:** match counts
  and scanned offsets form contiguous arrays. Binary searches and permutation-based
  comparisons still read irregularly; disjoint output ranges alone do not prove
  coalesced writes across lanes.
- **Compute utilization / Reducing control divergence:** unequal search paths and
  match-range lengths are potential divergence sources, not optimizations shown to
  have been eliminated.
- **Warp-level primitives**, **Register tiling**, and **Thread coarsening** in
  bounds searches, scans, and copies must be checked in the chosen library kernels.

Presorted inputs and reused build preprocessing save sorting. Scanning counts
assigns disjoint output ranges; that does not establish **Privatization** of shared
atomic state or **Double buffering**.

**Profiling questions.** Separate sorting, searches, scans, and expansion. Check
duplicate distribution and the cost of nested-key comparisons.

### Conditional and mixed joins

**Conditional algorithm.** Parallel nested loops evaluate an AST predicate.
An outer-row thread traverses inner rows; the size pass combines thread counts
with `cub::BlockReduce` and a global atomic update.

Explicit [`CUDF_KERNEL` entry points][conditional-kernels]:
`compute_conditional_join_output_size<block_size, has_nulls>`,
`conditional_join<block_size, output_cache_size, has_nulls>`, and
`conditional_join_anti_semi`.
Device helpers include `add_pair_to_cache` and `flush_output_cache`.

**Conditional checklist mapping (source observations).**

- **Synchronization latency / Warp-level primitives:** `__ballot_sync`,
  `__syncwarp`, and `cub::ShuffleIndex` coordinate active lanes and share output
  reservations without a block-wide barrier for each cache flush.
- **Memory utilization / Privatization:** the size pass accumulates thread counts
  and reduces them per block before a global atomic update. Output caches also
  batch global counter reservations, though shared cache insertion uses atomics.
- **Memory utilization / Using coalesceable global memory accesses:** flushes
  distribute consecutive output positions across lanes. This is output packing,
  not input **Shared memory tiling**.
- **General / Thread coarsening:** each outer-row thread evaluates many inner-row
  pairs, amortizing outer-row assignment. This increases serial work per thread
  and does not establish a performance-tuned coarsening factor.
- **Compute utilization / Reducing control divergence:** compile-time nullability
  selects expression-handling code, but predicate outcomes still diverge.

This remains quadratic candidate work for a general all-pairs predicate; caching
does not change that.

**Profiling questions.** Inspect predicate instruction cost, divergence, repeated
inner-table reads, match density, and shared-memory/occupancy tradeoffs.
The cache is an output cache, not evidence of input tiling.

**Mixed algorithm.** Equality keys narrow candidate matches before applying the
residual expression. See the [`mixed_join` kernel][mixed-kernel] and
[`mixed_join_count`][mixed-size]. Predicate selectivity and candidate skew determine
how much work remains after hashing.

**Mixed checklist mapping (source observations).** **Thread coarsening** is visible
as grid-stride outer-row traversal plus multiple candidate checks per row.
**Using coalesceable global memory accesses** applies to adjacent input-pair,
hash-index, and count entries, not irregular bucket reads or differently sized
output ranges. **Reducing control divergence** benefits from compile-time
nullability, but variable probe chains and predicates still diverge. Per-thread
expression scratch occupies shared memory; it does not demonstrate **Shared memory
tiling** or **Register tiling**. The kernel writes to precomputed offsets rather
than using the conditional join's warp output cache, so that cache's
**Warp-level primitives** must not be attributed to this path.

**Study for the join families:** [He et al., Relational Joins on Graphics
Processors][join-paper]. It is historical/algorithmic background, not a specification
of the current cuCollections hash table or this sort-merge implementation.

## 6. Scan / prefix operations

Here "scan" means a prefix computation, not reading a database table.
The [API dispatcher][scan-entry] selects inclusive/exclusive and special rank paths.
[Typed dispatch][scan-dispatch] selects sum, min, max, product, count, or EWMA.

**Algorithm.** Inclusive scan includes the current value; exclusive scan starts
with the identity. [Inclusive][inclusive] and [exclusive][exclusive] implementations
call Thrust. Null replacement and output validity are handled separately; null
policy affects propagation. There is no ordinary scalar-scan cuDF `__global__`
body to identify in these wrappers.

**Device entry points.** Operator functors feed Thrust. Count scan uses a
`__device__` validity-to-count lambda. An especially instructive specialization is
[`recurrence_functor<T>::operator()`][ewma]: it composes affine-transform pairs,
turning the EWMA recurrence into a parallel scan.

**Checklist mapping (source observations and library boundary).**

- **Memory utilization / Using coalesceable global memory accesses:** numeric
  inputs/outputs are contiguous ranges supplied to Thrust. Null replacement and
  count transforms do not require separately materialized value columns.
- **Compute utilization / Reducing control divergence:** typed operator and
  null-handling dispatch specialize execution; element validity can still vary.
- **Warp-level primitives**, **Register tiling**, **Thread coarsening**, and
  **Double buffering** are possible backend strategies, not established by
  `inclusive_scan`/`exclusive_scan` at the cuDF call site.

EWMA's affine-pair composition exposes parallelism in the recurrence without
proving a register tile. Null-mask processing and EWMA intermediates add
passes/traffic; inspect their backend scans separately.

**Profiling questions.** Measure launch overhead for short columns, traffic for
long columns, and null-processing overhead. Floating-point regrouping means
parallel results need not be bitwise equal to sequential accumulation.

**Study:** [Blelloch][blelloch] is explicitly cited by the EWMA source.
[Merrill and Garland, Decoupled Look-back][lookback] explains modern device-wide
scan techniques, but the cuDF call alone does not establish the exact Thrust
backend selected for every type and dependency version.

## 7. Filters / stream compaction

**Algorithm.** [`apply_boolean_mask`][filter-entry] selects rows whose mask is
true and valid, compacts row indices with `thrust::copy_if`, and gathers all
columns using the selected indices. [The compaction driver][copy-if] handles
empty/all-selected cases as well.

**Device entry points.** `boolean_mask_filter<has_nulls>::operator()(size_type)`
is the `__device__` predicate. The selection kernel is library-owned.
The [gather implementation][gather] specializes fixed-width, string, list,
dictionary, and struct data, including a device validity-gather functor.

**Checklist mapping (source observations and library boundary).**

- **Compute utilization / Reducing control divergence:** the driver selects
  `boolean_mask_filter<true>` or `<false>` once according to mask nullability.
  The non-null case removes validity handling; varying mask values still affect
  selection work.
- **Memory utilization / Using coalesceable global memory accesses:** mask
  evaluation traverses adjacent rows. Fixed-width gather writes a dense output,
  but selected source rows may be scattered, so input coalescing depends on
  selectivity and locality.
- **Warp-level primitives** and **Thread coarsening** for the actual selection
  kernel are library-owned and need backend evidence.

Counting iterators avoid an allocated input-index array; one compacted map shares
selection across columns. Neither mechanism establishes **Register tiling** or
**Shared memory tiling**.

**Profiling questions.** Vary selectivity, table width, row locality, and
variable-length data. Predicate evaluation may be cheap while gathering strings
or nested children dominates.

**Study:** [GPU Gems 3, Chapter 39][gpu-gems] explains the foundational
predicate -> prefix positions -> scatter approach to stream compaction.
This is not a claim about the exact current Thrust `copy_if` implementation.

## 8. Reductions

### Simple and segmented reductions

**Algorithm.** Select a typed operator, transform/cast input through iterators,
replace nulls with the identity as appropriate, and reduce to a scalar.
The [simple driver][simple-reduce] calls the
[`cub::DeviceReduce::Reduce` wrapper][cub-reduce]. Output validity is separate
from the identity used during computation.

[Segmented reduction][segmented-reduce] uses
`cub::DeviceSegmentedReduce::Reduce` for the corresponding paths, producing one
result per supplied offset interval. Other reduction kinds can select different
implementations; not every aggregation is a simple sum-like tree reduction.

**Device entry points.** Reduction operator/transformer functors execute inside
CUB-owned kernels. A `__device__` finalization lambda handles compound results.

**Checklist mapping (source observations and library boundary).**

- **Memory utilization / Using coalesceable global memory accesses:** ordinary
  fixed-width reductions expose contiguous element ranges to CUB. Dictionary and
  nested access can be indirect; iterator fusion avoids a cast/null-filled temporary
  but does not guarantee every underlying access is coalesced.
- **Memory utilization / Privatization** and **Synchronization latency /
  Warp-level primitives** are relevant to hierarchical partial reduction, but the
  wrapper delegates its implementation to CUB. Verify the selected kernels before
  claiming a particular thread/warp/block reduction strategy.
- **Register tiling**, **Loop unrolling**, **Vector loads and stores**, and
  **Thread coarsening** require dependency/generated-code evidence; the typed
  operator alone proves none of them.

The wrapper first queries temporary-storage size, allocates storage, and then
executes the reduction; the query is not a second full reduction.

**Profiling questions.** Examine input bandwidth, operator cost, segment-size
imbalance, allocation/launch overhead, and output dtype conversions.

### Mean, variance, and standard deviation

Mean reduces a sum and finalizes using valid count. Variance/std reduce
[`var_std<ResultType>`][stats], holding sum, squared deviations, and count.
Its `CUDF_HOST_DEVICE operator+` merges partial statistics using the pairwise
Chan-Golub-LeVeque algorithm; this is not naive subtraction of squared sums.

**Checklist mapping (source observations and library boundary).** Mergeable
statistics allow partial accumulation, making **Privatization** relevant to the
CUB reduction as above. The `var_std` sum/deviation/count fields are intermediate
state, not proof of **Register tiling** or spill-free registers.
For **Occupancy tuning**, their larger state and arithmetic cost relative to sum
are resource considerations to measure, not evidence that cuDF explicitly tunes
this operator's launch. The associative formulation improves parallelism and
numerical stability independently of the checklist's low-level mechanisms.

**Profiling questions.** Evaluate numerical tolerance as well as throughput;
operator state can affect register use and arithmetic intensity.

**Study:** [Chan, Golub, and LeVeque, Algorithms for Computing the Sample
Variance: Analysis and Recommendations][variance-paper], explicitly cited in
the implementation. [GPU Gems][gpu-gems] also introduces tree-style parallel
combination, although its primary subject is scan.

## 9. Sorting

### Radix fast path

**Algorithm.** Eligible single fixed-width columns without nulls use CUB
`DeviceRadixSort`: `SortKeys` for values and `SortPairs` for keys plus indices.
The [radix implementation][radix] and [sorted-order implementation][radix-order]
also contain device transformations/decomposers for floating-point ordering.

**Checklist mapping (source observations and library boundary).**

- **Compute utilization / Reducing control divergence:** the eligible fixed-width,
  no-null path avoids arbitrary row-comparator calls and selects sort direction
  outside the kernel. This does not establish branch-free execution inside CUB.
- **Memory utilization / Using coalesceable global memory accesses:** contiguous
  key/index arrays expose a regular input layout, while digit redistribution and
  temporary traffic are handled by CUB.
- **Shared memory tiling**, **Register tiling**, **Vector loads and stores**, and
  **Avoiding shared memory bank conflicts** must be verified in the selected radix
  kernels. Separate input/output arrays alone are not the checklist's
  **Double buffering** optimization.

The value-only fast path avoids the general permutation-and-gather pipeline,
but multiple sorting passes and temporary buffers still incur traffic.

**Profiling questions.** Inspect key width, temporary traffic, input size, and
floating-point preprocessing; do not assume a fixed pass count without checking CUB.

### Comparator and multi-column sorting

Single-column comparator paths use CUB `DeviceMergeSort`, sorting row indices.
[`simple_comparator<T>::operator()`][column-sort] is the `__device__` entry for
value comparisons and null precedence.

[Multi-column/nested sorting][table-sort] initializes indices and calls
`thrust::sort`/`stable_sort` with a lexicographic comparator.
[`device_row_comparator`][row-compare] contains the device-level column and nested
comparisons. `sort_by_key` then gathers values using the permutation.

**Checklist mapping (source observations and library boundary).**

- **Compute utilization / Reducing control divergence:** `if constexpr` selects
  stable/unstable code and separate nested/non-nested comparators specialize
  execution. This eliminates some choices, not data-dependent lexicographic
  early exits, null branches, or varying string lengths.
- **Memory utilization / Using coalesceable global memory accesses:** the index
  array is contiguous, but permutation-based key reads and final gathers are
  indirect. Sorting indices reduces wide-row movement without proving coalescing
  for those key reads.
- **Loop unrolling**, **Register tiling**, and **Warp-level primitives** in the
  sorting machinery require CUB/Thrust backend evidence.

Radix paths are stable even when the public operation does not require stability.

**Profiling questions.** Compare early versus late differing key columns,
string lengths, nesting, null density, and final gather cost.
Do not infer a fixed comparator-sort backend solely from a Thrust API name.

### Segmented sorting

[Eligible integral/decimal inputs without nulls][segmented-sort] use
`cub::DeviceSegmentedSort`; int128 is excluded from this fast path.
The general path can incorporate segment identifiers into lexicographic sorting.

**Checklist mapping (source observations and library boundary).**
**Reducing control divergence** is supported by selecting the eligible fixed-width
no-null fast path, but segment sizes still vary.
For **Using coalesceable global memory accesses**, contiguous offsets and arrays
offer regular storage while segment boundaries complicate work assignment.
**Occupancy tuning**, **Thread coarsening**, and **Shared memory tiling** inside
`DeviceSegmentedSort` are delegated to CUB, not established by this wrapper.

**Profiling questions.** Segment size distribution changes available parallelism
and load balance. Measure metadata generation and temporary copies as well as
sorting; this path should not be described simply as segmented radix sort.

**Study for sorting:** [Satish, Harris, and Garland, Designing Efficient Sorting
Algorithms for Manycore GPUs][sort-paper]. It explains GPU radix/merge design
principles, not necessarily the precise current CUB implementation.

## 10. C++ concepts in the CUDA implementation

| Mechanism | Concrete example and purpose |
| --- | --- |
| Runtime-to-compile-time dispatch | [`type_dispatcher`][type-dispatch] maps a runtime column dtype to a typed functor invocation |
| Non-type template parameters | Aggregation kind, block size, nullability, and sorting method specialize generated code |
| Constraints and traits | [`requires`-constrained reductions][reduce-dispatch] admit supported type/operator combinations |
| `if constexpr` | Selects stable sort and removes unneeded branches/instantiations |
| Device function objects | Predicates, hashers, comparators, and aggregation operators plug cuDF semantics into generic kernels |
| Iterator composition | Counting/transform/zip iterators fuse element access with computation rather than allocating every intermediate |
| CRTP inheritance | [`sum : simple_op<sum>` and `variance : compound_op<variance>`][stats] reuse interfaces statically |
| Ordinary inheritance | [`null_replaced_value_accessor : value_accessor<SourceType>`][groupby-reduce] extends typed access |
| Runtime polymorphism | The [host aggregation descriptor hierarchy][aggregation-types] represents requested operations |
| PImpl and RAII | Hash-join implementation objects, owning columns, `unique_ptr`, and RMM buffers manage implementation boundaries and lifetimes |

Host-side virtual aggregation descriptors should not be confused with the GPU
hot path: device execution largely uses statically selected functors/templates.
Views do not own underlying data; source buffers and comparator preprocessing
must remain alive for the asynchronous device work using them.

## 11. Studying the library-owned kernels

For the actual dependency-level `__global__` bodies, continue into:

- [NVIDIA CCCL](https://github.com/NVIDIA/cccl): CUB device dispatch/agent code and
  Thrust CUDA backends for selection, scan, reduction, and sorting.
- [NVIDIA cuCollections](https://github.com/NVIDIA/cuCollections): concurrent
  hash-container insertion, lookup, counting, and retrieval.

These upstream links are navigation aids, not revision-matched evidence. Resolve
the dependency revisions used by the build before attributing a particular kernel
or optimization to this cuDF revision.

For performance study, separate preprocessing, size/count passes, core execution,
and output materialization. Use representative nulls, skew, cardinality, nesting,
and output sizes. Source inspection identifies optimization mechanisms; profiling
is needed to establish bottlenecks and tradeoffs.

[revision]: https://github.com/jazracherif/cudf/tree/e81ad54ed957866c235cf0720fc3ef8986aa39b0
[kernel-macro]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/include/cudf/types.hpp#L10-L25
[groupby-entry]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/groupby/groupby.cu#L53-L259
[groupby-types]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/groupby/hash/helpers.cuh#L20-L101
[groupby-driver]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/groupby/hash/compute_single_pass_aggs.cuh#L59-L160
[mapping]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/groupby/hash/compute_mapping_indices.cuh#L23-L152
[shared-aggs]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/groupby/hash/compute_shared_memory_aggs.cu#L94-L264
[global-functors]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/groupby/hash/single_pass_functors.cuh#L101-L156
[atomic-aggs]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/groupby/hash/global_memory_aggregator.cuh#L48-L115
[groupby-sort]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/groupby/sort/sort_helper.cu#L80-L195
[groupby-reduce]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/groupby/sort/group_single_pass_reduction_util.cuh#L41-L198
[groupby-scan]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/groupby/sort/group_scan_util.cuh
[join-entry]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/join/join.cu#L27-L147
[hash-build]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/join/hash_join/hash_join.cu
[hash-types]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/join/hash_join/hash_join_impl.cuh#L17-L59
[hash-retrieve]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/join/hash_join/retrieve_impl.cuh#L22-L190
[join-compare]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/join/hash_join/dispatch.cuh#L24-L106
[distinct-join]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/join/distinct_hash_join.cu
[mark-join]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/join/mark_join.cu
[smj-api]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/include/cudf/join/sort_merge_join.hpp#L49-L112
[smj-code]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/join/sort_merge_join.cu#L207-L370
[conditional-api]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/join/conditional_join.cu#L371-L435
[conditional-kernels]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/join/conditional_join_kernels.cuh
[mixed-kernel]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/join/mixed_join_kernel.cuh
[mixed-size]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/join/mixed_join_size_kernel.cuh
[scan-entry]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/reductions/scan/scan.cpp#L14-L54
[scan-dispatch]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/reductions/scan/scan.cuh#L34-L65
[inclusive]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/reductions/scan/scan_inclusive.cu
[exclusive]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/reductions/scan/scan_exclusive.cu
[ewma]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/reductions/scan/ewm.cu#L27-L78
[filter-entry]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/stream_compaction/apply_boolean_mask.cu#L23-L88
[copy-if]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/include/cudf/detail/copy_if.cuh#L43-L69
[gather]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/include/cudf/detail/gather.cuh
[reduction-api]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/include/cudf/reduction.hpp#L90-L230
[simple-reduce]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/reductions/simple.cuh#L47-L84
[cub-reduce]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/include/cudf/reduction/detail/reduction.cuh
[segmented-reduce]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/include/cudf/reduction/detail/segmented_reduction.cuh
[stats]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/include/cudf/reduction/detail/reduction_operators.cuh
[reduce-dispatch]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/reductions/reductions.cpp#L55-L137
[sort-entry]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/sort/sort.cu#L22-L99
[radix]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/sort/sort_radix.cu
[radix-order]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/sort/sorted_order_radix.cu
[column-sort]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/sort/sort_column_impl.cuh#L35-L108
[table-sort]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/sort/sort_impl.cuh#L31-L95
[row-compare]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/include/cudf/detail/row_operator/lexicographic.cuh#L158-L605
[segmented-sort]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/sort/segmented_sort_impl.cuh
[type-dispatch]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/include/cudf/utilities/type_dispatcher.hpp#L353-L465
[aggregation-types]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/include/cudf/aggregation.hpp#L67-L203
[blelloch]: https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf
[lookback]: https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
[gpu-gems]: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
[join-paper]: https://doi.org/10.1145/1376616.1376670
[variance-paper]: https://doi.org/10.1080/00031305.1983.10483115
[sort-paper]: https://mgarland.org/files/papers/gpusort-ipdps09.pdf
[checklist-comment]: https://github.com/jazracherif/cudf/pull/1#discussion_r4331173782
[checklist-image]: https://github.com/user-attachments/assets/f6c559f0-2b28-4a06-babc-60b9a06367a1
