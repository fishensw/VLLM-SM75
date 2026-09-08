// SPDX-License-Identifier: Apache-2.0
// firefly_allreduce: fp8 2-GPU allreduce (SHM backend, PHB/无 P2P 如 T10)。
//
// 目标: TP2 每层 2 次 AllReduce([M,H] fp16 residual stream)在 PCIe 瓶颈上量减半
// (33.55MB fp16 -> 16.78MB fp8), 省通信时间 (PLAN-fp8-allreduce.md §2)。
//
// 传输: /dev/shm + cudaHostRegister (POC 验证 vfio 下可行, PLAN §7 风险4 已解除)。
//   cudaIpcGetMemHandle 对 host-mapped 内存 err=1 不通, 故不用 IPC, 改两进程 map 同一
//   tmpfs 区 + 各自 cudaHostRegister 成 pinned 设备可访问, 物理页经 tmpfs 共享。
//
// 流程 (每 rank, **全 GPU, 可被 cudagraph capture**):
//   round1 (GPU): 全局 amax (atomicMax) -> SHM 交换 amax (flag_scale spin) -> common scale
//   round2 (GPU):
//     quant (dev fp16 -> dev fp8 buffer, 用 scale_dev)
//     cudaMemcpyAsync D2H (dev fp8 -> shm.data[r] host)
//     set+spin data flag (release/acquire, 用 seq_dev)
//     cudaMemcpyAsync H2D (shm.data[1-r] host -> dev fp8 buffer)
//     dequant+sum ((self_q + peer_q) * scale_dev -> dev fp16, fp32 累加)
//     set+spin barrier flag (保证两 rank 都读完对端 data 才复用 slot)
//     bump seq_dev (本 allreduce 完成, flag 计数 +1)
//   barrier 保证两 rank 都读完对端 data 后才复用本端 slot (防 drift 覆盖)。
//   vllm forward 天然 lockstep(两 rank 同层同 allreduce), spin ~0 开销。
//
// **cudagraph 兼容**: seq/scale 放 device scratch (by-pointer, replay 时读当前值),
//   不用 by-value host 参数 (capture 会固定值, flag 不递增 → 无法区分本次/上次)。
//   seq_dev 每 allreduce +1 (ar_bump_seq), scale_dev 每步重算 (ar_scale_exchange)。
//   round1 不再走 CPU `.item()` D2H sync (那与 capture 的 stream 语义冲突)。
//
// fp8 转换: CUDA 原生 __nv_cvt_float_to_fp8 / __nv_cvt_fp8_to_halfraw (sm75 软转正确)。
//   c10 fp8e4m3fn_from_fp32_value 的 device 路径实测坏, 勿用 (proto 诊断确认)。
//
// SHM 布局 (字节, 与 python 侧约定一致; D = data_half = max M*H, fp8 1B/元素):
//   [0, D)            data_slot_0 (fp8)
//   [D, 2D)           data_slot_1 (fp8)
//   [2D, 2D+4)        amax_0 (f32)        round1 GPU 侧 (ar_scale_exchange)
//   [2D+4, 2D+8)      amax_1 (f32)
//   [2D+8, 2D+16)     flag_scale_0 (u64)  round1 GPU 侧
//   [2D+16, 2D+24)    flag_scale_1 (u64)
//   [2D+24, 2D+32)    flag_data_0 (u64)   round2 GPU 侧
//   [2D+32, 2D+40)    flag_data_1 (u64)
//   [2D+40, 2D+48)    barrier_0 (u64)     round2 GPU 侧 (读完对端后置位)
//   [2D+48, 2D+56)    barrier_1 (u64)
//   TOTAL = 2D + 56
//
// device scratch 布局 (16 字节, by-pointer 传, replay 读当前值):
//   [+0, +4)  amax_dev (f32)   全局 amax accumulator (每步 memset 0 后 atomicMax)
//   [+4, +8)  scale_dev (f32)  common scale (ar_scale_exchange 算, quant/dequant 读)
//   [+8, +16) seq_dev (u64)    flag 计数 (初始 1, 每 allreduce +1, 所有 flag 用它)
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>  // 轻量 stream API, 不拖 cusparse/cublas

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>  // __nv_cvt_float_to_fp8 / __nv_cvt_fp8_to_halfraw

namespace {

// ---- 内存序原语 (.sys 域, 跨设备; 参考 custom_collective_common.cuh) ----
__device__ __forceinline__ void st_release_sys_u64(uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v));
}
__device__ __forceinline__ uint64_t ld_acquire_sys_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p));
  return v;
}

// fp8 转换 (sm75 无 fp8 硬件, CUDA 原生软转; c10 device 路径实测坏)
__device__ __forceinline__ uint8_t quant_to_fp8(float v) {
  return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float dequant_from_fp8(uint8_t q) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
  return __half2float(__half(h));
}

// ---- kernel: 全局 amax (atomicMax 到 amax_dev; |x| 非负) ----
// 非负 float 的 __float_as_uint 位模式随值单调递增, 故 atomicMax(uint)==float max
// (sm75 无 f32 atom 硬件, 用 uint atomicMax 绕开)。
__global__ void ar_amax_partial(const __half* __restrict__ x,
                                float* __restrict__ amax_dev, int64_t n) {
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  unsigned int* acc = reinterpret_cast<unsigned int*>(amax_dev);
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    atomicMax(acc, __float_as_uint(fabsf(__half2float(x[i]))));
  }
}

// ---- kernel: 1 block, SHM 交换 amax -> common scale (round1, GPU 侧) ----
// seq 从 seq_dev 读 (by-pointer, replay 安全); 写 scale_dev 供 quant/dequant 读。
__global__ void ar_scale_exchange(char* __restrict__ shm_base,
                                  int64_t data_half, int64_t rank,
                                  const float* __restrict__ amax_dev,
                                  float* __restrict__ scale_dev,
                                  const uint64_t* __restrict__ seq_dev) {
  if (threadIdx.x == 0) {
    uint64_t seq = *seq_dev;
    float local_amax =
        __uint_as_float(*reinterpret_cast<const unsigned int*>(amax_dev));
    char* base = shm_base;
    float* amax_shm = reinterpret_cast<float*>(base + 2 * data_half);
    uint64_t* flag_scale =
        reinterpret_cast<uint64_t*>(base + 2 * data_half + 8);
    amax_shm[rank] = local_amax;                       // 写本端 amax
    st_release_sys_u64(flag_scale + rank, seq);         // 发布 (release)
    while (ld_acquire_sys_u64(flag_scale + (1 - rank)) != seq) {
    }                                                  // 等对端 (acquire)
    float peer_amax = amax_shm[1 - rank];
    *scale_dev = fmaxf(local_amax, peer_amax) / 448.0f;  // e4m3fn max finite
  }
}

// ---- kernel: quantize (x dev fp16 -> xq dev fp8; scale 从 scale_dev 读) ----
__global__ void ar_quant(const __half* __restrict__ x,
                         uint8_t* __restrict__ dst,
                         const float* __restrict__ scale_dev, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = quant_to_fp8(__half2float(x[i]) / (*scale_dev));
}

// ---- kernel: dequant + sum (self_q + peer_q dev fp8 -> out dev fp16) ----
__global__ void ar_dequant_sum(const uint8_t* __restrict__ self_q,
                               const uint8_t* __restrict__ peer_q,
                               __half* __restrict__ out,
                               const float* __restrict__ scale_dev,
                               int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float a = dequant_from_fp8(self_q[i]);
    float b = dequant_from_fp8(peer_q[i]);
    out[i] = __float2half((a + b) * (*scale_dev));  // fp32 累加, 回 fp16
  }
}

// ---- kernel: set 本端 flag (release) + spin 对端 flag (acquire), 用 seq_dev ----
__global__ void ar_set_spin(uint64_t* __restrict__ my_flag,
                            const uint64_t* __restrict__ peer_flag,
                            const uint64_t* __restrict__ seq_dev) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint64_t seq = *seq_dev;
    st_release_sys_u64(my_flag, seq);
    while (ld_acquire_sys_u64(peer_flag) != seq) {
    }
  }
}

// ---- kernel: seq_dev += 1 (本 allreduce 完成, 供下次 flag 递增) ----
__global__ void ar_bump_seq(uint64_t* __restrict__ seq_dev) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *seq_dev += 1;
}

}  // namespace

// host launcher: 全 GPU round1+round2 (无 host sync, 可 cudagraph capture)。
// x: fp16 [M,H] contiguous (输入); xq/xq_peer: fp8 [n] device scratch;
//   out: fp16 [M,H] (x0+x1); scratch: device [16B] (amax/scale/seq, by-pointer);
//   shm_base: 已 cudaHostRegister 的 shm 首址; data_half: 数据半区字节数(>=n);
//   rank: 0/1; n: 实际元素数(M*H)。
void firefly_ar_exchange(at::Tensor x, at::Tensor xq, at::Tensor xq_peer,
                         at::Tensor out, at::Tensor scratch,
                         int64_t shm_base, int64_t data_half,
                         int64_t rank, int64_t n) {
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const __half* xp = reinterpret_cast<const __half*>(x.data_ptr());
  uint8_t* xqp = reinterpret_cast<uint8_t*>(xq.data_ptr());
  uint8_t* xqpp = reinterpret_cast<uint8_t*>(xq_peer.data_ptr());
  __half* outp = reinterpret_cast<__half*>(out.data_ptr());
  // scratch: [+0] amax(f32) [+4] scale(f32) [+8] seq(u64)
  void* sp = scratch.data_ptr();
  float* amax_dev = reinterpret_cast<float*>(sp);
  float* scale_dev = reinterpret_cast<float*>(sp) + 1;
  uint64_t* seq_dev = reinterpret_cast<uint64_t*>(sp) + 1;
  char* base = reinterpret_cast<char*>(shm_base);
  uint8_t* my_data = reinterpret_cast<uint8_t*>(base) + rank * data_half;
  uint8_t* peer_data = reinterpret_cast<uint8_t*>(base) + (1 - rank) * data_half;
  uint64_t* flag_data =
      reinterpret_cast<uint64_t*>(base + 2 * data_half + 24);
  uint64_t* my_flag_data = flag_data + rank;
  uint64_t* peer_flag_data = flag_data + (1 - rank);
  uint64_t* flag_barrier =
      reinterpret_cast<uint64_t*>(base + 2 * data_half + 40);
  uint64_t* my_flag_barrier = flag_barrier + rank;
  uint64_t* peer_flag_barrier = flag_barrier + (1 - rank);
  constexpr int threads = 256;
  const dim3 grid((unsigned)((n + threads - 1) / threads));

  // round1 (GPU): 全局 amax -> SHM 交换 -> common scale
  cudaMemsetAsync(amax_dev, 0, sizeof(float), stream);
  ar_amax_partial<<<grid, threads, 0, stream>>>(xp, amax_dev, n);
  ar_scale_exchange<<<1, 1, 0, stream>>>(reinterpret_cast<char*>(shm_base),
                                         data_half, rank, amax_dev, scale_dev,
                                         seq_dev);
  // round2 (GPU): quant + D2H + data flag + H2D + dequant+sum + barrier + bump
  ar_quant<<<grid, threads, 0, stream>>>(xp, xqp, scale_dev, n);
  cudaMemcpyAsync(my_data, xqp, (size_t)n, cudaMemcpyDeviceToHost, stream);
  ar_set_spin<<<1, 1, 0, stream>>>(my_flag_data, peer_flag_data, seq_dev);
  cudaMemcpyAsync(xqpp, peer_data, (size_t)n, cudaMemcpyHostToDevice, stream);
  ar_dequant_sum<<<grid, threads, 0, stream>>>(xqp, xqpp, outp, scale_dev, n);
  ar_set_spin<<<1, 1, 0, stream>>>(my_flag_barrier, peer_flag_barrier, seq_dev);
  ar_bump_seq<<<1, 1, 0, stream>>>(seq_dev);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  // NOLINT
  m.def("firefly_ar_exchange", &firefly_ar_exchange,
        "firefly allreduce SHM full-GPU (round1 amax + round2 exchange): "
        "amax + scale-exchange + quant + D2H + flag + H2D + dequant+sum + "
        "barrier + bump-seq, capture-safe");
}
