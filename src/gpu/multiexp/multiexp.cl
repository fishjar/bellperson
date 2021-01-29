/*
 * Same multiexp algorithm used in Bellman, with some modifications.
 * https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L174
 * The CPU version of multiexp parallelism is done by dividing the exponent
 * values into smaller windows, and then applying a sequence of rounds to each
 * window. The GPU kernel not only assigns a thread to each window but also
 * divides the bases into several groups which highly increases the number of
 * threads running in parallel for calculating a multiexp instance.
 */

__kernel void POINT_bellman_multiexp(
    __global POINT_affine *bases, // n*
    // 每个 thread 分配 bucket_len(2**10-1=1023) 个存储桶
    // 总共 `num_groups` * `num_windows` * `bucket_len` 个存储桶
    __global POINT_projective *buckets,
    __global POINT_projective *results, // 1*
    __global EXPONENT *exps, // n*
    uint n,
    uint num_groups, // num_groups * num_windows ~= 2 * CUDA_CORES
    uint num_windows, // num_windows = exp_bits / window_size = 256/10 = 26
    uint window_size) {

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = get_global_id(0);
  if(gid >= num_windows * num_groups) return;

  // We have (2^window_size - 1) buckets.
  // 当前 thread 存储桶数量（1023）
  const uint bucket_len = ((1 << window_size) - 1);

  // Each thread has its own set of buckets in global memory.
  // 指针计算，相当于移动n个单位
  // 移动到当前 thread 的第一个存储桶位置
  buckets += bucket_len * gid;

  // 初始化当前 thread 的所有存储桶
  const POINT_projective local_zero = POINT_ZERO;
  for(uint i = 0; i < bucket_len; i++) buckets[i] = local_zero;

  // 每个组分配的任务数量(7488999/334=22422)
  const uint len = (uint)ceil(n / (float)num_groups); // Num of elements in each group

  // This thread runs the multiexp algorithm on elements from `nstart` to `nened`
  // on the window [`bits`, `bits` + `w`)
  // 当前 thread 进行 `nstart` to `nened` 的任务计算

  // 每个组分配的任务数量 × 当前第?组（thread序号/每个组的thread） = 当前组的任务开始序号
  const uint nstart = len * (gid / num_windows);
  // 当前组的任务开始序号 + 每个组分配的任务数量 = 当前组结束的任务序号
  const uint nend = min(nstart + len, n);
  // 当前 thread 在当前组的位置 × 10，bits 的范围是 0-250
  const uint bits = (gid % num_windows) * window_size;
  // EXPONENT_BITS/num_windows 不能整除时，最后一个的 window_size 是余数
  const ushort w = min((ushort)window_size, (ushort)(EXPONENT_BITS - bits));

  POINT_projective res = POINT_ZERO;
  // 迭代每个分组的任务数量(7488999/334=22422)
  for(uint i = nstart; i < nend; i++) {
    // ind 是当前任务在存储桶中的指针？
    uint ind = EXPONENT_get_bits(exps[i], bits, w);

    #ifdef NVIDIA
      // O_o, weird optimization, having a single special case makes it
      // tremendously faster!
      // 511 is chosen because it's half of the maximum bucket len, but
      // any other number works... Bigger indices seems to be better...
      if(ind == 511) buckets[510] = POINT_add_mixed(buckets[510], bases[i]);
      else if(ind--) buckets[ind] = POINT_add_mixed(buckets[ind], bases[i]);
    #else
      if(ind--) buckets[ind] = POINT_add_mixed(buckets[ind], bases[i]);
    #endif
  }

  // Summation by parts
  // e.g. 3a + 2b + 1c = a +
  //                    (a) + b +
  //                    ((a) + b) + c
  POINT_projective acc = POINT_ZERO;
  for(int j = bucket_len - 1; j >= 0; j--) {
    acc = POINT_add(acc, buckets[j]);
    res = POINT_add(res, acc);
  }

  results[gid] = res;
}
