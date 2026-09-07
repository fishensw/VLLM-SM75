// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// firefly: B1-hard 反 gptq_marlin_repack(int4/fp16, 非 act-order, num_bits=4, is_a_8bit=false)
// 并 per-group->per-channel 折叠反量化到 int8。索引推导见 firefly.py。
//
// 每个 block 负责一行 n(256 线程), 两遍: 先求 per-row amax(得 c_n), 再量化。
// fetch_q 在寄存器里现算 marlin 布局位置(无 [N,K] 索引数组, 规避 PyTorch 版 50ms/层)。
//
// 反量化数学统一为 w_deq = (q - zp) * s:
//   对称(uint4b8, compressed-tensors/GPTQ-sym): zp=8 常数, zp 指针传 null。
//   非对称(uint4, AWQ): zp 传干净 per-group zp [N, K/gs] float32。
// 对称路径 zp=null→8.0f, 与改动前 (q-8)*s 逐 bit 一致(不回归)。
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>  // 轻量 stream API, 不拖 cusparse/cublas

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

// build 用 -D__CUDA_NO_HALF_CONVERSIONS__, 需显式转换(不能 static_cast<float>)。
template <typename ScalarT>
__device__ __forceinline__ float scalar_to_float(ScalarT v);
template <>
__device__ __forceinline__ float scalar_to_float<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(
    __nv_bfloat16 v) {
  return __bfloat162float(v);
}

// 干净 (n, k) -> marlin 扁平张量位置 (src_flat) 与位移 (shift=4*i)。
__device__ __forceinline__ int fetch_q(
    const int32_t* __restrict__ marlin, int n, int k, int n_tiles_global) {
  const int nt = n >> 6, dn = n & 63;
  const int kt = k >> 4, dk = k & 15;
  const int dk_lo = dk & 7, dk_hi = dk >> 3;
  const int tc_row = dk_lo & ~1;  // 向下取偶
  const int dk_off_lo = dk_lo & 1;
  const int dn_off8 = (dn & 15) >= 8 ? 1 : 0;
  const int cur_n = dn - 8 * dn_off8;
  const int i = 4 * dk_off_lo + 2 * dn_off8 + dk_hi;
  const int th_id = (cur_n & 15) * 4 + (tc_row >> 1);
  const int p = th_id * 4 + (cur_n >> 4);
  const int tile_idx = kt * n_tiles_global + nt;
  const int src_flat = tile_idx * 128 + p;
  return (marlin[src_flat] >> (4 * i)) & 0xF;
}

template <typename ScalarT>
__global__ void firefly_dequant_kernel(
    const int32_t* __restrict__ marlin,
    const ScalarT* __restrict__ scale,
    const float* __restrict__ zp,  // null → 对称(zp=8); 非对称 → [N, K/gs]
    int8_t* __restrict__ out,
    float* __restrict__ c_n,
    int N, int K, int n_tiles_global, int group_size) {
  const int n = blockIdx.x;
  if (n >= N) return;
  const int tid = threadIdx.x;
  const int scale_cols = K / group_size;
  __shared__ float sh_warp[32];
  __shared__ float s_max;

  // Pass 1: per-row amax |w_deq|
  float local_max = 0.f;
  for (int k = tid; k < K; k += blockDim.x) {
    const int q = fetch_q(marlin, n, k, n_tiles_global);
    const float s = scalar_to_float(scale[n * scale_cols + k / group_size]);
    const float z = zp ? zp[n * scale_cols + k / group_size] : 8.0f;
    local_max = fmaxf(local_max, fabsf(static_cast<float>(q - z) * s));
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    local_max = fmaxf(local_max, __shfl_xor_sync(~0u, local_max, off));
  if ((tid & 31) == 0) sh_warp[tid >> 5] = local_max;
  __syncthreads();
  const int num_warps = (blockDim.x + 31) >> 5;
  if (tid == 0) {
    float v = sh_warp[0];
    for (int i = 1; i < num_warps; i++) v = fmaxf(v, sh_warp[i]);
    s_max = v;
  }
  __syncthreads();
  const float c = s_max / 127.0f;
  if (tid == 0) c_n[n] = c;

  // Pass 2: 量化到 int8。用除法(非乘倒数)以与 PyTorch 版 torch.round(w_deq/c_n)
  // 逐 bit 一致(同为 fp32 IEEE 除法 + round-to-nearest-even)。
  for (int k = tid; k < K; k += blockDim.x) {
    const int q = fetch_q(marlin, n, k, n_tiles_global);
    const float s = scalar_to_float(scale[n * scale_cols + k / group_size]);
    const float z = zp ? zp[n * scale_cols + k / group_size] : 8.0f;
    const float w_deq = static_cast<float>(q - z) * s;
    int v = (c > 0.f) ? __float2int_rn(w_deq / c) : 0;
    if (v > 127) v = 127;
    if (v < -127) v = -127;
    out[n * K + k] = static_cast<int8_t>(v);
  }
}

// 单遍反量化: c_n 已在 load 时预计算(纯权重导出, M/chunk 无关), 运行时只跑量化 pass,
// 省掉 firefly_dequant 的 pass1 amax(~46% 反量化开销)。用除法(非乘倒数), 且 c_n[n]
// 正是 firefly_dequant pass1 写出的 c(同一 fp32 值), 故与现网两遍逐 bit 一致。
template <typename ScalarT>
__global__ void firefly_dequant_cached_kernel(
    const int32_t* __restrict__ marlin,
    const ScalarT* __restrict__ scale,
    const float* __restrict__ zp,  // null → 对称(zp=8); 非对称 → [N, K/gs]
    int8_t* __restrict__ out,
    const float* __restrict__ c_n,  // [N] 预计算 per-channel scale
    int N, int K, int n_tiles_global, int group_size) {
  const int n = blockIdx.x;
  if (n >= N) return;
  const int tid = threadIdx.x;
  const int scale_cols = K / group_size;
  const float c = c_n[n];
  for (int k = tid; k < K; k += blockDim.x) {
    const int q = fetch_q(marlin, n, k, n_tiles_global);
    const float s = scalar_to_float(scale[n * scale_cols + k / group_size]);
    const float z = zp ? zp[n * scale_cols + k / group_size] : 8.0f;
    const float w_deq = static_cast<float>(q - z) * s;
    int v = (c > 0.f) ? __float2int_rn(w_deq / c) : 0;
    if (v > 127) v = 127;
    if (v < -127) v = -127;
    out[n * K + k] = static_cast<int8_t>(v);
  }
}

// 单遍反量化(fast): c_n 预计算, per-row 只算一次倒数 r=1/c_n, 量化用乘(非除)。
// 比除法版再省 ~30%(除法是每元素一次 FP32 div, T10 上吞吐低、抢 ALU); 代价 off-by-one
// ≤0.06%(int8 可忽略, 见 tmp/PLAN §8 bench)。VLLM_FIREFLY_DEQUANT_MODEL=fast 时启用。
template <typename ScalarT>
__global__ void firefly_dequant_cached_recip_kernel(
    const int32_t* __restrict__ marlin,
    const ScalarT* __restrict__ scale,
    const float* __restrict__ zp,  // null → 对称(zp=8); 非对称 → [N, K/gs]
    int8_t* __restrict__ out,
    const float* __restrict__ c_n,  // [N] 预计算 per-channel scale
    int N, int K, int n_tiles_global, int group_size) {
  const int n = blockIdx.x;
  if (n >= N) return;
  const int tid = threadIdx.x;
  const int scale_cols = K / group_size;
  const float r = (c_n[n] > 0.f) ? 1.0f / c_n[n] : 0.f;
  for (int k = tid; k < K; k += blockDim.x) {
    const int q = fetch_q(marlin, n, k, n_tiles_global);
    const float s = scalar_to_float(scale[n * scale_cols + k / group_size]);
    const float z = zp ? zp[n * scale_cols + k / group_size] : 8.0f;
    const float w_deq = static_cast<float>(q - z) * s;
    int v = __float2int_rn(w_deq * r);
    if (v > 127) v = 127;
    if (v < -127) v = -127;
    out[n * K + k] = static_cast<int8_t>(v);
  }
}

}  // namespace

void firefly_dequant(
    at::Tensor marlin_w, at::Tensor scale, at::Tensor zp, at::Tensor out_int8,
    at::Tensor c_n, int64_t N, int64_t K, int64_t padded_n, int64_t padded_k,
    int64_t group_size) {
  const int n_tiles_global = static_cast<int>(padded_n) / 64;
  const dim3 grid(static_cast<unsigned>(N));
  constexpr int threads = 256;
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const int32_t* marlin = marlin_w.data_ptr<int32_t>();
  int8_t* out = out_int8.data_ptr<int8_t>();
  float* cn = c_n.data_ptr<float>();
  // zp 未定义/为空 → 对称(zp=8); 否则视为干净 per-group zp [N, K/gs] float32。
  // 用 numel() 判空: at::Tensor 无 is_empty()(此 PyTorch C++ API 版本), numel() 恒可用。
  const float* zp_ptr =
      (zp.defined() && zp.numel() > 0) ? zp.data_ptr<float>() : nullptr;
// 用无模板 data_ptr()(void*, 恒可用) + static_cast, 避免实例化
// data_ptr<__half>()/data_ptr<__nv_bfloat16>()(libtorch 未导出, 链接期 undefined symbol)。
#define LAUNCH(ScalarT)                                             \
  firefly_dequant_kernel<ScalarT><<<grid, threads, 0, \
                                           stream>>>(marlin,        \
                                                     static_cast<const ScalarT*>(scale.data_ptr()), \
                                                     zp_ptr, \
                                                     out, cn,      \
                                                     static_cast<int>(N),      \
                                                     static_cast<int>(K), n_tiles_global, \
                                                     static_cast<int>(group_size))
  if (scale.scalar_type() == at::kHalf) {
    LAUNCH(__half);
  } else if (scale.scalar_type() == at::kBFloat16) {
    LAUNCH(__nv_bfloat16);
  } else {
    TORCH_CHECK(false, "unsupported scale dtype: ", scale.scalar_type());
  }
#undef LAUNCH
}

// 单遍反量化 launcher: c_n 预计算传入(与 firefly_dequant 同签名, 仅 c_n 由输入提供)。
void firefly_dequant_cached(
    at::Tensor marlin_w, at::Tensor scale, at::Tensor zp, at::Tensor out_int8,
    at::Tensor c_n, int64_t N, int64_t K, int64_t padded_n, int64_t padded_k,
    int64_t group_size) {
  const int n_tiles_global = static_cast<int>(padded_n) / 64;
  const dim3 grid(static_cast<unsigned>(N));
  constexpr int threads = 256;
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const int32_t* marlin = marlin_w.data_ptr<int32_t>();
  int8_t* out = out_int8.data_ptr<int8_t>();
  const float* cn = c_n.data_ptr<float>();
  const float* zp_ptr =
      (zp.defined() && zp.numel() > 0) ? zp.data_ptr<float>() : nullptr;
#define LAUNCH(ScalarT)                                             \
  firefly_dequant_cached_kernel<ScalarT><<<grid, threads, 0, \
                                           stream>>>(marlin,        \
                                                     static_cast<const ScalarT*>(scale.data_ptr()), \
                                                     zp_ptr, \
                                                     out, cn,      \
                                                     static_cast<int>(N),      \
                                                     static_cast<int>(K), n_tiles_global, \
                                                     static_cast<int>(group_size))
  if (scale.scalar_type() == at::kHalf) {
    LAUNCH(__half);
  } else if (scale.scalar_type() == at::kBFloat16) {
    LAUNCH(__nv_bfloat16);
  } else {
    TORCH_CHECK(false, "unsupported scale dtype: ", scale.scalar_type());
  }
#undef LAUNCH
}

// 单遍 fast launcher: 同签名, 用 firefly_dequant_cached_recip_kernel(乘倒数)。
void firefly_dequant_cached_recip(
    at::Tensor marlin_w, at::Tensor scale, at::Tensor zp, at::Tensor out_int8,
    at::Tensor c_n, int64_t N, int64_t K, int64_t padded_n, int64_t padded_k,
    int64_t group_size) {
  const int n_tiles_global = static_cast<int>(padded_n) / 64;
  const dim3 grid(static_cast<unsigned>(N));
  constexpr int threads = 256;
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const int32_t* marlin = marlin_w.data_ptr<int32_t>();
  int8_t* out = out_int8.data_ptr<int8_t>();
  const float* cn = c_n.data_ptr<float>();
  const float* zp_ptr =
      (zp.defined() && zp.numel() > 0) ? zp.data_ptr<float>() : nullptr;
#define LAUNCH(ScalarT)                                             \
  firefly_dequant_cached_recip_kernel<ScalarT><<<grid, threads, 0, \
                                           stream>>>(marlin,        \
                                                     static_cast<const ScalarT*>(scale.data_ptr()), \
                                                     zp_ptr, \
                                                     out, cn,      \
                                                     static_cast<int>(N),      \
                                                     static_cast<int>(K), n_tiles_global, \
                                                     static_cast<int>(group_size))
  if (scale.scalar_type() == at::kHalf) {
    LAUNCH(__half);
  } else if (scale.scalar_type() == at::kBFloat16) {
    LAUNCH(__nv_bfloat16);
  } else {
    TORCH_CHECK(false, "unsupported scale dtype: ", scale.scalar_type());
  }
#undef LAUNCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  // NOLINT
  m.def("firefly_dequant", &firefly_dequant,
        "firefly: reverse gptq_marlin_repack + dequant to int8 "
        "(sym zp=8 / asym per-group zp)");
  m.def("firefly_dequant_cached", &firefly_dequant_cached,
        "firefly: single-pass dequant to int8 with load-time-cached c_n "
        "(bit-exact with firefly_dequant pass2)");
  m.def("firefly_dequant_cached_recip", &firefly_dequant_cached_recip,
        "firefly: single-pass FAST dequant to int8 (reciprocal multiply, "
        "off-by-one <=0.06%)");
}
