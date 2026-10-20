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
The exact figure was not supplied or verified for this analysis. The following
categories are an original, PMPP-inspired working framework, **not a transcription
or an exact mapping to that figure**:

1. **Work assignment:** enough parallelism, grid-stride traversal, load balance.
2. **Memory access:** coalescing, bandwidth, irregular gathers, temporary traffic.
3. **Reuse:** registers, shared-memory staging, tiling, and avoiding repeated work.
4. **Control flow:** divergence, specialization, and unnecessary instructions.
5. **Coordination:** synchronization, atomic contention, and privatization.
6. **Resources:** registers/shared memory, occupancy, and launch overhead.

Each operation below separates **source observations** from **profiling questions**.
No speedup, occupancy level, bandwidth, or bottleneck is claimed to have been
measured. In particular, shared memory does not automatically mean faster execution,
and maximizing occupancy is not itself the objective.

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

**Optimization observations.** Shared partials reduce global update traffic.
For low cardinality, the shared kernel replicates aggregation locations across
threads to reduce contention before combining them. Its phases synchronize around
shared state. The driver sizes its grid using occupancy information for both
mapping and aggregation kernels. Group indices make output access irregular.

**Profiling questions.** How do key skew, distinct-group count, and aggregation
width affect atomics, shared-memory pressure, and fallback frequency? Compare the
cost of mapping/materialization against aggregation itself; do not assume the
shared path wins for every workload.

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

**Optimization observations.** Sorted groups enable contiguous segmented work,
but obtaining the permutation and gathering values adds traffic. Device accessors
handle null replacement without materializing a separate transformed column.

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

**Optimization observations.** Building the smaller input reduces build work.
The reusable object amortizes build preprocessing across probes. Primitive and
nested row paths select different comparators. An optional known output size can
avoid counting. Probe chains and variable match multiplicity imply irregular work.

**Profiling questions.** Examine load factor, collisions, duplicate-key skew,
probe memory access, and output size. Many-to-many output expansion can dominate
even if hash lookup is efficient.

### Distinct, semi, and anti joins

[`distinct_hash_join`][distinct-join] exploits distinct build keys and probes with
`find_async`/`find_if_async`; device key and output functors adapt hash records.
[`mark_join`][mark-join] provides specialized semi/anti execution with explicit
`mark_probe_kernel`, `mark_retrieve_kernel`, `compact_if_kernel`, and
`clear_marks_kernel` entry points.

**Optimization observations.** Specialized existence/marking operations avoid
materializing every matching pair. Mark storage and compaction introduce their
own traffic and coordination.

**Profiling questions.** Compare hit rate, duplicates, mark contention, and the
cost of output compaction rather than extrapolating from general join throughput.

### Sort-merge join

Construct `sort_merge_join(right, is_right_sorted, compare_nulls, stream)`,
then call its [inner/left probe methods][smj-api].

**Algorithm.** Prepare sorted row orders, perform parallel lower/upper-bound
searches for equal ranges, scan match counts into offsets, and expand ranges
into index pairs. The [matching implementation][smj-code] is not simply a single
linear two-pointer merge.

**Optimization observations.** Presorted inputs/reused build preprocessing save
sorting. Match-count scans assign disjoint output ranges. Bounds searches create
irregular reads; later expansion may have very uneven output per key.

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

**Optimization observations.** Warp-local shared output caches batch global
writes and counter reservations. Warp synchronization protects coordination.
Compile-time nullability specializes expression handling. This remains quadratic
candidate work for a general all-pairs predicate; caching does not change that.

**Profiling questions.** Inspect predicate instruction cost, divergence, repeated
inner-table reads, match density, and shared-memory/occupancy tradeoffs.
The cache is an output cache, not evidence of input tiling.

**Mixed algorithm.** Equality keys narrow candidate matches before applying the
residual expression. See the [`mixed_join` kernel][mixed-kernel] and
[`mixed_join_count`][mixed-size]. Predicate selectivity and candidate skew determine
how much work remains after hashing.

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

**Optimization observations.** Transform iterators avoid separate transformed
inputs. Prefix computation exposes parallelism despite the apparent sequential
recurrence. Null-mask processing and EWMA intermediates still add passes/traffic.

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

**Optimization observations.** Counting iterators avoid allocating input row
indices. Compile-time mask nullability selects the predicate variant.
Compacting an index map shares selection across columns, at the cost of the
map and subsequent gather traffic. Selected source locations may be scattered.

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

**Optimization observations.** Iterator fusion avoids a temporary cast/null-filled
column. Hierarchical combination is delegated to CUB. The wrapper first queries
temporary-storage size, allocates storage, and then executes the reduction;
the query is not a second full reduction.

**Profiling questions.** Examine input bandwidth, operator cost, segment-size
imbalance, allocation/launch overhead, and output dtype conversions.

### Mean, variance, and standard deviation

Mean reduces a sum and finalizes using valid count. Variance/std reduce
[`var_std<ResultType>`][stats], holding sum, squared deviations, and count.
Its `CUDF_HOST_DEVICE operator+` merges partial statistics using the pairwise
Chan-Golub-LeVeque algorithm; this is not naive subtraction of squared sums.

**Optimization observations.** Mergeable statistics expose parallelism while
improving numerical stability, but increase per-item arithmetic and intermediate
state relative to sum.

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

**Optimization observations.** Fixed-width keys allow bit-oriented sorting
without arbitrary row-comparator calls. The value-only fast path avoids the
general permutation-and-gather pipeline. Multiple sorting passes and temporary
buffers still incur traffic.

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

**Optimization observations.** Sorting indices avoids repeatedly moving wide
rows, but comparisons/gathers add indirect accesses. `if constexpr` selects
stable/unstable code, and nested/non-nested paths specialize comparisons.
Radix paths are stable even when the public operation does not require stability.

**Profiling questions.** Compare early versus late differing key columns,
string lengths, nesting, null density, and final gather cost.
Do not infer a fixed comparator-sort backend solely from a Thrust API name.

### Segmented sorting

[Eligible integral/decimal inputs without nulls][segmented-sort] use
`cub::DeviceSegmentedSort`; int128 is excluded from this fast path.
The general path can incorporate segment identifiers into lexicographic sorting.

**Optimization observations/questions.** Segment size distribution changes available
parallelism and load balance. Measure metadata generation and temporary copies as
well as sorting; this path should not be described simply as segmented radix sort.

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
[shared-aggs]: https://github.com/jazracherif/cudf/blob/e81ad54ed957866c235cf0720fc3ef8986aa39b0/cpp/src/groupby/hash/compute_shared_memory_aggs.cu#L172-L264
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
[join-paper]: https://www.microsoft.com/en-us/research/publication/relational-joins-on-graphics-processors/
[variance-paper]: https://doi.org/10.1080/00031305.1983.10483115
[sort-paper]: https://mgarland.org/files/papers/gpusort-ipdps09.pdf
