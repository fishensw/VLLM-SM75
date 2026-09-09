// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// firefly fused: fp8 权重在 GEMM 的 B-load 阶段即时反量化为 int8, 再走 IMMA tensor core。
//
// 数学(与 firefly.py fp8_to_int8 + int8_prefill_linear 逐 bit 一致, 首测用 def 除法):
//     w_deq[n,k]  = fp8_decode(w_fp8[n,k]) * block_scale[n//128, k//128]   // fp32 乘
//     w_int8[n,k] = clamp(round_to_even(w_deq[n,k] / c_n[n]), -127, 127)   // def: 除法
//     y[m,n]      = (x_q[m,:] . w_int8[:,n]) * x_s[m] * c_n[n]             // int32 accum + 2x 缩放
//
// 做法: 复用 vllm SM75 int8 GEMM 的全部结构(DefaultMmaCore / MmaPipelined / GemmWithEpilogueVisitor
// + EVT epilogue), 只把 B 的 PredicatedTileIterator 换成 Fp8DequantIteratorB —— 它用同一个
// PredicatedTileAccessIterator 读 16 字节 fp8(和 reference 逐字节相同), 再按 (n,k) 现算
// fp8->int8 反量化。坐标用 access iterator 的 get() 指针反解(B 列主序 [K,N], 元素 (k,n) 在
// n*K+k), 不依赖 thread map 的 contig/strided 约定(该 thread map 相对 pitch-linear 是转置的)。
//
// 编译: torch.utils.cpp_extension.load, TORCH_CUDA_ARCH_LIST=7.5, -O3,
//       include = flashinfer cutlass include(2.x API + cute + fp8)。

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>  // 轻量 stream API

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/float8.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/functional.h>

// arch 头必须最先: mma_singlestage.h(default_mma_core.h 引入)在解析期引用
// cutlass::arch::Sm70, 若 arch.h 未先包含则 "no member Sm70"。
#include <cutlass/arch/mma_sm75.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>

#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/predicated_tile_access_iterator.h>
#include <cutlass/gemm/threadblock/default_mma_core.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm75.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

// visitor_2x.hpp 里 OutputTileThreadLayout 引用 DefaultThreadMapTensorOp(解析期),
// 该头不自己 include 定义头; 必须在此先包含, 否则 "not a template"。
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/epilogue/threadblock/fusion/visitor_2x.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
// vllm 自定义的 row/col-or-scalar broadcast visitors(与现有 cutlass_scaled_mm 的
// ScaledEpilogue 同款)。包含它即可拿到 VisitorColOrScalarBroadcast /
// VisitorRowOrScalarBroadcast; 它自身会 include visitor_2x.hpp/visitors.hpp/cute。
#include "cutlass_extensions/epilogue/broadcast_load_epilogue_c2x.hpp"

// 关键: GemmWithEpilogueVisitor 要求 Epilogue 有 FusionCallbacks/OutputOp/Params,
// 裸 EVTD(TreeVisitor2x) 没有。必须把 EVTD 包进 EpilogueWithVisitorCallbacks
// (与 vllm DefaultGemmWithVisitor 完全一致)。BaseEpilogue 取标准 GEMM 的完整
// LinearCombination epilogue(有 Shape/WarpMmaOperator/... 供 wrapper 使用)。
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/epilogue_base_streamk.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h>
#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/gemm/kernel/gemm_universal.h>
#include <cutlass/gemm/kernel/gemm_universal_with_visitor.h>
#include <cutlass/gemm/kernel/gemm_universal_with_visitor_streamk.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>

using namespace cute;

#define FIREFLY_CUTLASS_CHECK(status)                                        \
  {                                                                          \
    cutlass::Status firefly_cutlass_error_ = (status);                       \
    TORCH_CHECK(firefly_cutlass_error_ == cutlass::Status::kSuccess,         \
                cutlassGetStatusString(firefly_cutlass_error_));             \
  }

namespace firefly_fused {

// fp8 e4m3 -> fp32 精确解码(sm75 上走 C++ 位运算路径; 所有 e4m3 值在 fp32 中精确可表示,
// 与 torch.float8_e4m3fn.to(float32) 逐 bit 一致)。
__device__ __forceinline__ float fp8_to_float(uint8_t b) {
  cutlass::float_e4m3_t v;
  v.storage = b;
  return static_cast<float>(v);
}

// ---- int4(W4A16) fused: marlin int4 在 GEMM B-load 内即时反量化 int8 ----
// 数学(与 firefly.cu firefly_dequant_cached 逐 bit 一致, def 除法):
//     w_deq[n,k]  = (fetch_q(marlin, n, k) - zp[n, k/gs]) * scale[n, k/gs]  // fp32
//     w_int8[n,k] = clamp(round_to_even(w_deq[n,k] / c_n[n]), -127, 127)
//     y[m,n]      = (x_q[m,:] . w_int8[:,n]) * x_s[m] * c_n[n]
// 对称(uint4b8) zp=null→8.0f; 非对称(uint4/AWQ) zp=per-group 干净 [N, K/gs] float32。
// scale 是干净 [N, K/gs] fp16/bf16(非 marlin permute 布局)。

// build 用 -D__CUDA_NO_HALF_CONVERSIONS__, 需显式转换(不能 static_cast<float>)。
template <typename ScalarT>
__device__ __forceinline__ float ff_scalar_to_float(ScalarT v);
template <>
__device__ __forceinline__ float ff_scalar_to_float<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float ff_scalar_to_float<__nv_bfloat16>(
    __nv_bfloat16 v) {
  return __bfloat162float(v);
}

// 干净 (n, k) -> marlin 扁平 int32 张量位置, 取 4-bit 值 q。
// 镜像 firefly.cu fetch_q / gptq_marlin_repack(int4/fp16, 非 act-order) 的反解。
__device__ __forceinline__ int ff_fetch_q(
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

// B-load 阶段即时反量化 iterator。接口对齐 PredicatedTileIterator(供 MmaPipelined 使用):
//   load(Fragment&) / operator++() / clear_mask(bool) / Fragment / Element / Layout / AccessType / Params。
// Element=int8_t(mma 看到的 B), Layout=ColumnMajor, AccessType=16(=kAlignmentB)。
// 实际读 w_fp8(1 字节 fp8, 与 int8 同尺寸), 在 load 里按 (n,k) 反量化成 int8。
template <typename Shape, typename ThreadMap, int AccessSize>
struct Fp8DequantIteratorB {
  using Element = int8_t;
  using Layout = cutlass::layout::ColumnMajor;
  static int const kAdvanceRank = 0;
  using TensorCoord = cutlass::MatrixCoord;
  using Pointer = Element*;

  using AccessType =
      cutlass::AlignedArray<Element, AccessSize, AccessSize * sizeof_bits<Element>::value / 8>;
  using Fragment =
      cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  // 读原始 fp8 字节的 access iterator(与 MmaCore::IteratorB 内部同款: 同 thread map / 同坐标)。
  using U8AccessType =
      cutlass::AlignedArray<uint8_t, AccessSize, AccessSize * sizeof_bits<uint8_t>::value / 8>;
  using AccessIter = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, uint8_t,
      cutlass::layout::PitchLinear, 0, ThreadMap, U8AccessType>;

  class Params {
   public:
    using Base = typename AccessIter::Params::Base;  // PredicatedTileAccessIteratorParams
    friend Fp8DequantIteratorB;

   private:
    AccessIter::Params access_params_;
    const float* block_scale_ = nullptr;  // [N/128, K/128] fp32
    const float* c_n_ = nullptr;          // [N] fp32 per-channel 正 scale
    int N_ = 0;
    int K_ = 0;

   public:
    Params() = default;
    CUTLASS_HOST_DEVICE
    Params(cutlass::layout::ColumnMajor const& layout)
        : access_params_(cutlass::layout::PitchLinear(layout.stride(0))) {}
    CUTLASS_HOST_DEVICE
    Params(Base const& base) : access_params_(base) {}

    // 由 Fp8GemmKernel::Params 在构造时注入(反量化所需的额外数据)。
    CUTLASS_HOST_DEVICE
    void set_dequant_info(const float* block_scale, const float* c_n, int N, int K) {
      block_scale_ = block_scale;
      c_n_ = c_n;
      N_ = N;
      K_ = K;
    }

    CUTLASS_HOST_DEVICE
    AccessIter::Params const& access_params() const { return access_params_; }
    CUTLASS_HOST_DEVICE
    const float* block_scale() const { return block_scale_; }
    CUTLASS_HOST_DEVICE
    const float* c_n() const { return c_n_; }
    CUTLASS_HOST_DEVICE
    int N() const { return N_; }
    CUTLASS_HOST_DEVICE
    int K() const { return K_; }
  };

 private:
  AccessIter access_it_;
  uint8_t const* base_ptr_;  // B 原始指针(元素 (k=0,n=0)), 用于反解 (n,k)
  const float* block_scale_;
  const float* c_n_;
  int N_;
  int K_;

 public:
  CUTLASS_HOST_DEVICE
  Fp8DequantIteratorB(Params const& params, Pointer pointer, TensorCoord const& extent,
                      int thread_id, TensorCoord const& tb_offset, int const* indices = nullptr)
      : access_it_(params.access_params(), reinterpret_cast<uint8_t*>(pointer),
                   cutlass::layout::PitchLinearCoord(extent.row(), extent.column()), thread_id,
                   cutlass::layout::PitchLinearCoord(tb_offset.row(), tb_offset.column()), indices),
        base_ptr_(reinterpret_cast<uint8_t const*>(pointer)),
        block_scale_(params.block_scale()),
        c_n_(params.c_n()),
        N_(params.N()),
        K_(params.K()) {}

  CUTLASS_DEVICE
  void load(Fragment& frag) {
    // 越界 / 末 tile(被 clear_mask 置 0) 的元素保持 0(与 reference global_load(valid=false)→0 一致)。
    frag.clear();
    const int scale_k = K_ / 128;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      access_it_.set_iteration_index(s);
      bool valid = access_it_.valid();
      uint8_t const* p = reinterpret_cast<uint8_t const*>(access_it_.get());
      ++access_it_;
      if (!valid) continue;
      // B 列主序 [K,N]: 元素 (k,n) 在扁平下标 n*K+k。16 字节沿 K 连续。
      // 指针差 = ptrdiff_t(64bit), 直接转 int64_t, 避免 pointer->long 转换报错。
      int64_t flat = static_cast<int64_t>(p - base_ptr_);
      int64_t n = flat / K_;
      int k0 = static_cast<int>(flat - n * K_);  // = flat % K, 已含 K tile 偏移
      if (n < 0 || n >= N_) continue;
      float c = c_n_[n];
      int ns = static_cast<int>(n) / 128;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < AccessSize; ++j) {
        int kg = k0 + j;
        float w = fp8_to_float(p[j]);
        float bsv = block_scale_[ns * scale_k + kg / 128];
        float wdeq = w * bsv;
        // def: fp32 除法 + round-to-nearest-even(与 torch.round / firefly.cu 一致)。
        int v = (c > 0.f) ? __float2int_rn(wdeq / c) : 0;
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        frag[s * AccessSize + j] = static_cast<int8_t>(v);
      }
    }
  }

  CUTLASS_DEVICE
  Fp8DequantIteratorB& operator++() {
    // AdvanceRank=0 -> 沿 K 推进一个 tile(与 reference PredicatedTileIterator 一致)。
    access_it_.add_tile_offset(cutlass::layout::PitchLinearCoord(1, 0));
    return *this;
  }

  CUTLASS_DEVICE
  void clear_mask(bool enable = true) { access_it_.clear_mask(enable); }
};

// int4(W4A16) B-load 即时反量化 iterator。结构与 Fp8DequantIteratorB 同(同接口、
// 同虚拟 [K,N] 坐标发生器), 差别只在 load 体: 源是 marlin packed int4(非干净行主序),
// 故 access iterator 只当"坐标发生器"(借 get()-base 解 (n,k0), 不解引用其数据), 真正取
// 值走 ff_fetch_q 直接读 marlin。一次 access = 同一 n 的 16 个连续 k(见 Fp8DequantIteratorB
// 注释的 thread map 约定), 故 j 对应 kg = k0 + j。
template <typename Shape, typename ThreadMap, int AccessSize, typename ScalarT>
struct Int4DequantIteratorB {
  using Element = int8_t;
  using Layout = cutlass::layout::ColumnMajor;
  static int const kAdvanceRank = 0;
  using TensorCoord = cutlass::MatrixCoord;
  using Pointer = Element*;

  using AccessType =
      cutlass::AlignedArray<Element, AccessSize, AccessSize * sizeof_bits<Element>::value / 8>;
  using Fragment =
      cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  // 虚拟 [K,N] 坐标发生器(读 1 字节占位, 不解引用; 与 fp8 版同 thread map/坐标)。
  using U8AccessType =
      cutlass::AlignedArray<uint8_t, AccessSize, AccessSize * sizeof_bits<uint8_t>::value / 8>;
  using AccessIter = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, uint8_t,
      cutlass::layout::PitchLinear, 0, ThreadMap, U8AccessType>;

  class Params {
   public:
    using Base = typename AccessIter::Params::Base;
    friend Int4DequantIteratorB;

   private:
    AccessIter::Params access_params_;
    const ScalarT* scale_ = nullptr;  // 干净 [N, K/gs]
    const float* zp_ = nullptr;       // null → 对称(zp=8); 非对称 → [N, K/gs]
    const float* c_n_ = nullptr;      // [N]
    int N_ = 0, K_ = 0, group_size_ = -1, n_tiles_global_ = 0;

   public:
    Params() = default;
    CUTLASS_HOST_DEVICE
    Params(cutlass::layout::ColumnMajor const& layout)
        : access_params_(cutlass::layout::PitchLinear(layout.stride(0))) {}
    CUTLASS_HOST_DEVICE
    Params(Base const& base) : access_params_(base) {}

    CUTLASS_HOST_DEVICE
    void set_dequant_info(const ScalarT* scale, const float* zp, const float* c_n, int N,
                          int K, int group_size, int n_tiles_global) {
      scale_ = scale;
      zp_ = zp;
      c_n_ = c_n;
      N_ = N;
      K_ = K;
      group_size_ = group_size;
      n_tiles_global_ = n_tiles_global;
    }

    CUTLASS_HOST_DEVICE
    AccessIter::Params const& access_params() const { return access_params_; }
    CUTLASS_HOST_DEVICE
    const ScalarT* scale() const { return scale_; }
    CUTLASS_HOST_DEVICE
    const float* zp() const { return zp_; }
    CUTLASS_HOST_DEVICE
    const float* c_n() const { return c_n_; }
    CUTLASS_HOST_DEVICE
    int N() const { return N_; }
    CUTLASS_HOST_DEVICE
    int K() const { return K_; }
    CUTLASS_HOST_DEVICE
    int group_size() const { return group_size_; }
    CUTLASS_HOST_DEVICE
    int n_tiles_global() const { return n_tiles_global_; }
  };

 private:
  AccessIter access_it_;
  uint8_t const* base_ptr_;  // 虚拟 [K,N] 起点(=marlin base), 用于反解 (n,k)
  const int32_t* marlin_;    // marlin packed int4 [padded_k/16, padded_n*2]
  const ScalarT* scale_;
  const float* zp_;
  const float* c_n_;
  int N_, K_, group_size_, n_tiles_global_;

 public:
  CUTLASS_HOST_DEVICE
  Int4DequantIteratorB(Params const& params, Pointer pointer, TensorCoord const& extent,
                       int thread_id, TensorCoord const& tb_offset, int const* indices = nullptr)
      : access_it_(params.access_params(), reinterpret_cast<uint8_t*>(pointer),
                   cutlass::layout::PitchLinearCoord(extent.row(), extent.column()), thread_id,
                   cutlass::layout::PitchLinearCoord(tb_offset.row(), tb_offset.column()), indices),
        base_ptr_(reinterpret_cast<uint8_t const*>(pointer)),
        marlin_(reinterpret_cast<int32_t const*>(pointer)),
        scale_(params.scale()),
        zp_(params.zp()),
        c_n_(params.c_n()),
        N_(params.N()),
        K_(params.K()),
        group_size_(params.group_size()),
        n_tiles_global_(params.n_tiles_global()) {}

  CUTLASS_DEVICE
  void load(Fragment& frag) {
    frag.clear();
    const int scale_cols = K_ / group_size_;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      access_it_.set_iteration_index(s);
      bool valid = access_it_.valid();
      uint8_t const* p = reinterpret_cast<uint8_t const*>(access_it_.get());
      ++access_it_;
      if (!valid) continue;
      // 虚拟 [K,N] 列主序: 元素 (k,n) 在 flat = n*K+k; 16 字节沿 K 连续。
      int64_t flat = static_cast<int64_t>(p - base_ptr_);
      int64_t n = flat / K_;
      int k0 = static_cast<int>(flat - n * K_);
      if (n < 0 || n >= N_) continue;
      float c = c_n_[n];
      int64_t nn = n;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < AccessSize; ++j) {
        int kg = k0 + j;
        int q = ff_fetch_q(marlin_, static_cast<int>(nn), kg, n_tiles_global_);
        // 注意: 变量名不能用 s —— 外层 strided 循环也叫 s, 会被遮蔽导致
        // frag[s*AccessSize+j] 误用 float scale 当索引(fp8 版没此变量故无此坑)。
        float sc = ff_scalar_to_float(scale_[nn * scale_cols + kg / group_size_]);
        float z = zp_ ? zp_[nn * scale_cols + kg / group_size_] : 8.0f;
        float wdeq = static_cast<float>(q - z) * sc;
        int v = (c > 0.f) ? __float2int_rn(wdeq / c) : 0;
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        frag[s * AccessSize + j] = static_cast<int8_t>(v);
      }
    }
  }

  CUTLASS_DEVICE
  Int4DequantIteratorB& operator++() {
    access_it_.add_tile_offset(cutlass::layout::PitchLinearCoord(1, 0));
    return *this;
  }

  CUTLASS_DEVICE
  void clear_mask(bool enable = true) { access_it_.clear_mask(enable); }
};

// 在 GemmWithEpilogueVisitorStreamk 基础上(与 vllm DefaultGemmWithVisitor + StreamK swizzle
// 一致), 把 block_scale/c_n/N/K 注入 B iterator 的 Params。
// Arguments / Params 用继承(非影子)以便适配器直接访问标准字段(problem_size/lda/ldb...)。
template <typename Mma, typename Epilogue, typename ThreadblockSwizzle>
struct Fp8GemmKernel
    : cutlass::gemm::kernel::GemmWithEpilogueVisitorStreamk<Mma, Epilogue, ThreadblockSwizzle> {
  using BaseKernel =
      cutlass::gemm::kernel::GemmWithEpilogueVisitorStreamk<Mma, Epilogue, ThreadblockSwizzle>;

  struct Arguments : BaseKernel::Arguments {
    // 继承 base 的构造函数(本类有默认成员初始化器 -> 非聚合, 不能直接聚合初始化)。
    // 调用方用继承的 (mode, problem_size, ..., lda, ldb, ldc, ldd) 构造, 再赋额外字段。
    using BaseKernel::Arguments::Arguments;
    // GemmUniversalAdapter::to_underlying_arguments 里 transposed_problem() 返回 base 类型,
    // 需能转回 derived(派生类没有隐式 base->derived 拷贝)。显式提供 base 拷贝构造。
    CUTLASS_HOST_DEVICE
    Arguments(BaseKernel::Arguments const& base) : BaseKernel::Arguments(base) {}
    const float* block_scale = nullptr;
    const float* c_n = nullptr;
    int N = 0;
    int K = 0;
  };

  struct Params : BaseKernel::Params {
    Params() = default;
    CUTLASS_HOST_DEVICE
    Params(Arguments const& args, int device_sms, int sm_occupancy)
        : BaseKernel::Params(args, device_sms, sm_occupancy) {
      this->params_B.set_dequant_info(args.block_scale, args.c_n, args.N, args.K);
    }
  };
};

// int4 版: 在 streamk visitor kernel 基础上, 把 marlin/scale/zp/c_n/N/K/gs/n_tiles
// 注入 Int4DequantIteratorB::Params。Arguments/Params 继承 base(同 Fp8GemmKernel)。
// ScalarT = scale 干净 dtype(fp16/bf16)。
template <typename Mma, typename Epilogue, typename ThreadblockSwizzle, typename ScalarT>
struct Int4GemmKernel
    : cutlass::gemm::kernel::GemmWithEpilogueVisitorStreamk<Mma, Epilogue, ThreadblockSwizzle> {
  using BaseKernel =
      cutlass::gemm::kernel::GemmWithEpilogueVisitorStreamk<Mma, Epilogue, ThreadblockSwizzle>;

  struct Arguments : BaseKernel::Arguments {
    using BaseKernel::Arguments::Arguments;
    CUTLASS_HOST_DEVICE
    Arguments(BaseKernel::Arguments const& base) : BaseKernel::Arguments(base) {}
    const ScalarT* scale = nullptr;
    const float* zp = nullptr;
    const float* c_n = nullptr;
    int N = 0;
    int K = 0;
    int group_size = -1;
    int n_tiles_global = 0;
  };

  struct Params : BaseKernel::Params {
    Params() = default;
    CUTLASS_HOST_DEVICE
    Params(Arguments const& args, int device_sms, int sm_occupancy)
        : BaseKernel::Params(args, device_sms, sm_occupancy) {
      this->params_B.set_dequant_info(args.scale, args.zp, args.c_n, args.N, args.K,
                                      args.group_size, args.n_tiles_global);
    }
  };
};

// per-token(x_s, 按 M/行) 用 ColOrScalarBroadcast; per-channel(c_n, 按 N/列) 用
// RowOrScalarBroadcast。类型与 vllm ScaledEpilogue 完全一致(保证和现有
// cutlass_scaled_mm 逐 bit 相同); 仅 prepare_args 改吃裸指针(不用 torch::stable::Tensor)。
// col_broadcast / row_broadcast = true 表示按向量广播(x_s numel=M>1, c_n numel=N>1)。
template <typename ElementD, typename OutputTileThreadMap>
struct Fp8ScaledEpilogue {
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using ScaleA = cutlass::epilogue::threadblock::VisitorColOrScalarBroadcast<
      OutputTileThreadMap, float, Stride<Int<1>, Int<0>, Int<0>>>;  // x_s [M]
  using ScaleB = cutlass::epilogue::threadblock::VisitorRowOrScalarBroadcast<
      OutputTileThreadMap, float, Stride<Int<0>, Int<1>, Int<0>>>;  // c_n [N]

  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 = cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute = cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(const float* x_s, const float* c_n) {
    // OrScalar Arguments = {ptr, bool broadcast}; stride 走模板静态默认(= StrideMNL{})。
    typename ScaleA::Arguments a_args{x_s, true};
    typename ScaleB::Arguments b_args{c_n, true};
    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, {}};
  }
};

// SM75 大 M 配置: Tile 128x128x64, Warp 64x64x64, Instr 8x8x16, 2-stage pipelined。
template <typename ElementD>
struct cutlass_2x_gemm_fused {
  using ElementD_ = ElementD;
  using ElementAB = int8_t;
  using ElementAcc = int32_t;
  using Operator = cutlass::arch::OpMultiplyAddSaturate;

  using RowMajor = cutlass::layout::RowMajor;
  using ColumnMajor = cutlass::layout::ColumnMajor;

  using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
  static constexpr int Stages = 2;

  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      TileShape, WarpShape, float, 4, 1 /* epilogue stages */>;

  static constexpr int AlignmentAB = 16;
  static constexpr int AlignmentCD = 4;

  // EVT fusion 计算树(vllm ScaledEpilogue 同款类型, 提供 EVTCompute)。
  using Fusion = Fp8ScaledEpilogue<ElementD, OutputTileThreadMap>;
  using EVTCompute = typename Fusion::EVTCompute;
  using D = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, Int<1>, Int<0>>>;
  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;

  // 标准 GEMM kernel: 仅用于取完整 LinearCombination epilogue(BaseEpilogue)。
  // GemmWithEpilogueVisitor 要求 Epilogue 暴露 FusionCallbacks/OutputOp/Params,
  // 裸 EVTD(TreeVisitor2x) 三者皆无 -> 必须包进 EpilogueWithVisitorCallbacks
  // (vllm DefaultGemmWithVisitor 即如此, 见 default_gemm_universal_with_visitor.h)。
  using GemmBase = cutlass::gemm::kernel::DefaultGemmUniversal<
      ElementAB, RowMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      ElementAB, ColumnMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      float, RowMajor, ElementAcc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      TileShape, WarpShape, InstructionShape,
      cutlass::epilogue::thread::LinearCombination<float, AlignmentCD, ElementAcc, float>,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, Stages, Operator>::GemmKernel;
  using BaseEpilogue = typename GemmBase::Epilogue;
  // 包好的 epilogue: Epilogue::FusionCallbacks = EVTD, Epilogue::OutputOp::Params = EVTD::Arguments。
  using Epilogue = cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<
      BaseEpilogue, EVTD, 1>;

  using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
      TileShape, WarpShape, InstructionShape, int8_t, RowMajor, int8_t, ColumnMajor, int32_t,
      RowMajor, cutlass::arch::OpClassTensorOp, Stages, Operator>;
  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, int8_t, RowMajor, 1,
      typename MmaCore::IteratorThreadMapA, AlignmentAB, false, cutlass::layout::NoPermute>;
  using IteratorB = Fp8DequantIteratorB<
      cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
      typename MmaCore::IteratorThreadMapB, AlignmentAB>;
  using Mma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA, IteratorB,
      typename MmaCore::SmemIteratorB, int32_t, RowMajor, typename MmaCore::MmaPolicy>;

  // streamk kernel + ThreadblockSwizzleStreamK(与 vllm DefaultGemmWithVisitor 完全一致)。
  // 本测试形状(M/N tile 数 >> SM 数)下 streamk K-partitions=1, 无 K 切分, 逐元素 bit-exact。
  using KernelType =
      Fp8GemmKernel<Mma, Epilogue, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK>;
  using Op = cutlass::gemm::device::GemmUniversalAdapter<KernelType>;
};

// int4(W4A16) SM75 大 M 配置: 同 fp8 的 GEMM 结构(Tile/Warp/Instr/Stages/epilogue),
// 仅 B iterator 换 Int4DequantIteratorB + kernel 换 Int4GemmKernel。ScalarT=scale 干净
// dtype(fp16/bf16)。Epilogue 复用 Fp8ScaledEpilogue(只依赖 x_s/c_n 广播, 与 fp8 无关)。
template <typename ElementD, typename ScalarT>
struct cutlass_2x_gemm_fused_int4 {
  using ElementD_ = ElementD;
  using Scalar_ = ScalarT;  // scale 干净 dtype, 供 caller 取指针
  using ElementAB = int8_t;
  using ElementAcc = int32_t;
  using Operator = cutlass::arch::OpMultiplyAddSaturate;

  using RowMajor = cutlass::layout::RowMajor;
  using ColumnMajor = cutlass::layout::ColumnMajor;

  using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
  static constexpr int Stages = 2;

  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      TileShape, WarpShape, float, 4, 1 /* epilogue stages */>;

  static constexpr int AlignmentAB = 16;
  static constexpr int AlignmentCD = 4;

  using Fusion = Fp8ScaledEpilogue<ElementD, OutputTileThreadMap>;
  using EVTCompute = typename Fusion::EVTCompute;
  using D = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, Int<1>, Int<0>>>;
  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;

  using GemmBase = cutlass::gemm::kernel::DefaultGemmUniversal<
      ElementAB, RowMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      ElementAB, ColumnMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
      float, RowMajor, ElementAcc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      TileShape, WarpShape, InstructionShape,
      cutlass::epilogue::thread::LinearCombination<float, AlignmentCD, ElementAcc, float>,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, Stages, Operator>::GemmKernel;
  using BaseEpilogue = typename GemmBase::Epilogue;
  using Epilogue = cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<
      BaseEpilogue, EVTD, 1>;

  using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
      TileShape, WarpShape, InstructionShape, int8_t, RowMajor, int8_t, ColumnMajor, int32_t,
      RowMajor, cutlass::arch::OpClassTensorOp, Stages, Operator>;
  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, int8_t, RowMajor, 1,
      typename MmaCore::IteratorThreadMapA, AlignmentAB, false, cutlass::layout::NoPermute>;
  using IteratorB = Int4DequantIteratorB<
      cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
      typename MmaCore::IteratorThreadMapB, AlignmentAB, ScalarT>;
  using Mma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA, IteratorB,
      typename MmaCore::SmemIteratorB, int32_t, RowMajor, typename MmaCore::MmaPolicy>;

  using KernelType = Int4GemmKernel<
      Mma, Epilogue, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, ScalarT>;
  using Op = cutlass::gemm::device::GemmUniversalAdapter<KernelType>;
};

}  // namespace firefly_fused

template <typename Gemm>
static void fp8_fused_gemm_caller(
    at::Tensor& out, at::Tensor const& a, at::Tensor const& b_fp8, at::Tensor const& block_scale,
    at::Tensor const& c_n, at::Tensor const& x_s) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD_;

  int32_t m = a.size(0);
  int32_t k = a.size(1);
  int32_t n = b_fp8.size(0);
  cutlass::gemm::GemmCoord problem_size{m, n, k};

  int64_t lda = a.stride(0);       // x_q [M,K] row-major
  // w_fp8 [N,K] row-major 视为 cutlass B [K,N] 列主序: B[k,n]=w[n,k] 在 n*K+k,
  // ldb = 沿 N(输出维)的 stride = b_fp8.stride(0)=K(与 reference 的 b.stride(1) 等价,
  // 因 reference 传的是 w_int8.t() [K,N], 其 stride(1)=K)。
  int64_t ldb = b_fp8.stride(0);
  int64_t ldc = out.stride(0);     // out [M,N] row-major

  auto a_ptr = static_cast<ElementAB const*>(a.data_ptr());
  auto b_ptr = static_cast<uint8_t const*>(b_fp8.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  auto bs_ptr = static_cast<float const*>(block_scale.data_ptr());
  auto cn_ptr = static_cast<float const*>(c_n.data_ptr());
  auto xs_ptr = static_cast<float const*>(x_s.data_ptr());

  using StrideC = Stride<int64_t, Int<1>, Int<0>>;
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};
  typename Gemm::D::Arguments d_args{c_ptr, c_stride};

  // prepare_args 在 Fusion(=Fp8ScaledEpilogue) 上; EVTD::Arguments{EVTCompute_args, D_args}。
  auto evt_args = Gemm::Fusion::prepare_args(xs_ptr, cn_ptr);
  typename Gemm::EVTD::Arguments epilogue_args{evt_args, d_args};

  // 继承的 base 构造函数(16 参, 末 3 个 gather/scatter 指针默认 nullptr)。
  // streamk kernel 用 kGemmSplitKParallel(与 vllm 一致); 本测试形状下 K-partitions=1。
  typename Gemm::Op::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
      problem_size,
      1,
      epilogue_args,
      a_ptr,
      b_ptr,
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      lda,
      ldb,
      ldc,
      ldc);
  args.block_scale = bs_ptr;
  args.c_n = cn_ptr;
  args.N = n;
  args.K = k;

  typename Gemm::Op gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto device = a.device();
  auto workspace =
      at::empty({static_cast<int64_t>(workspace_size)},
                at::TensorOptions().dtype(at::kByte).device(device));

  auto stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();
  FIREFLY_CUTLASS_CHECK(gemm_op.can_implement(args));
  FIREFLY_CUTLASS_CHECK(gemm_op(args, workspace.data_ptr(), stream));
}

at::Tensor fp8_fused_scaled_mm(at::Tensor x_q, at::Tensor w_fp8, at::Tensor block_scale,
                               at::Tensor c_n, at::Tensor x_s, at::ScalarType out_dtype) {
  TORCH_CHECK(x_q.is_cuda() && w_fp8.is_cuda());
  TORCH_CHECK(x_q.dim() == 2 && w_fp8.dim() == 2);
  int64_t M = x_q.size(0), K = x_q.size(1), N = w_fp8.size(0);
  TORCH_CHECK(x_q.size(1) == w_fp8.size(1), "x_q K != w_fp8 K");
  TORCH_CHECK(N % 128 == 0 && K % 128 == 0, "N, K must be multiples of 128 (block scale)");

  at::Tensor out = at::empty({M, N}, at::TensorOptions().dtype(out_dtype).device(x_q.device()));
  if (out_dtype == at::kHalf) {
    fp8_fused_gemm_caller<firefly_fused::cutlass_2x_gemm_fused<cutlass::half_t>>(
        out, x_q, w_fp8, block_scale, c_n, x_s);
  } else if (out_dtype == at::kBFloat16) {
    fp8_fused_gemm_caller<firefly_fused::cutlass_2x_gemm_fused<cutlass::bfloat16_t>>(
        out, x_q, w_fp8, block_scale, c_n, x_s);
  } else {
    TORCH_CHECK(false, "out_dtype must be fp16 or bf16");
  }
  return out;
}

// ---- int4(W4A16) fused caller + 入口 ----
template <typename Gemm>
static void int4_fused_gemm_caller(
    at::Tensor& out, at::Tensor const& a, at::Tensor const& b_marlin, at::Tensor const& scale,
    at::Tensor const& zp, at::Tensor const& c_n, at::Tensor const& x_s, int32_t N, int32_t K,
    int32_t group_size, int32_t padded_n) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD_;
  using ScalarT = typename Gemm::Scalar_;

  int32_t m = a.size(0);
  cutlass::gemm::GemmCoord problem_size{m, N, K};

  int64_t lda = a.stride(0);  // x_q [M,K] row-major
  int64_t ldb = K;            // 虚拟 [K,N] 列主序沿 N 的 stride(=K)
  int64_t ldc = out.stride(0);

  auto a_ptr = static_cast<ElementAB const*>(a.data_ptr());
  // B 指针 = marlin 字节地址(int32 packed); iterator 只借它当虚拟 [K,N] 坐标发生器
  // (get()-base 解 (n,k)), 不解引用其数据。
  auto b_ptr = static_cast<void const*>(b_marlin.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  auto scale_ptr = static_cast<ScalarT const*>(scale.data_ptr());
  auto zp_ptr =
      (zp.defined() && zp.numel() > 0) ? static_cast<float const*>(zp.data_ptr()) : nullptr;
  auto cn_ptr = static_cast<float const*>(c_n.data_ptr());
  auto xs_ptr = static_cast<float const*>(x_s.data_ptr());

  using StrideC = Stride<int64_t, Int<1>, Int<0>>;
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};
  typename Gemm::D::Arguments d_args{c_ptr, c_stride};

  auto evt_args = Gemm::Fusion::prepare_args(xs_ptr, cn_ptr);
  typename Gemm::EVTD::Arguments epilogue_args{evt_args, d_args};

  typename Gemm::Op::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
      problem_size,
      1,
      epilogue_args,
      a_ptr,
      b_ptr,
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      lda,
      ldb,
      ldc,
      ldc);
  args.scale = scale_ptr;
  args.zp = zp_ptr;
  args.c_n = cn_ptr;
  args.N = N;
  args.K = K;
  args.group_size = group_size;
  args.n_tiles_global = padded_n / 64;

  typename Gemm::Op gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto device = a.device();
  auto workspace =
      at::empty({static_cast<int64_t>(workspace_size)},
                at::TensorOptions().dtype(at::kByte).device(device));

  auto stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();
  FIREFLY_CUTLASS_CHECK(gemm_op.can_implement(args));
  FIREFLY_CUTLASS_CHECK(gemm_op(args, workspace.data_ptr(), stream));
}

// int4 fused 入口: x_q [M,K] int8 + marlin int4 + 干净 scale/zp + c_n -> [M,N]。
// 与 int8_prefill_linear(dequant_marlin_to_int8_cached 反量化后走 cutlass_scaled_mm)
// 逐 bit 一致(同 fetch_q 反解 + 同 (q-zp)*s/c_n def 除法 + 同 int8 IMMA GEMM)。
at::Tensor int4_fused_scaled_mm(
    at::Tensor x_q, at::Tensor w_marlin, at::Tensor scale, at::Tensor zp, at::Tensor c_n,
    at::Tensor x_s, int64_t size_k, int64_t size_n, int64_t group_size, int64_t padded_n,
    at::ScalarType out_dtype) {
  TORCH_CHECK(x_q.is_cuda() && w_marlin.is_cuda());
  TORCH_CHECK(scale.scalar_type() == at::kHalf || scale.scalar_type() == at::kBFloat16,
              "int4 fused: scale must be fp16 or bf16");
  int64_t M = x_q.size(0);
  int64_t N = size_n, K = size_k;
  TORCH_CHECK(x_q.size(1) == K, "int4 fused: x_q K != size_k");
  TORCH_CHECK(group_size > 0 && (K % group_size) == 0, "int4 fused: group_size must divide K");

  at::Tensor out = at::empty({M, N}, at::TensorOptions().dtype(out_dtype).device(x_q.device()));
  int32_t n32 = static_cast<int32_t>(N), k32 = static_cast<int32_t>(K);
  int32_t gs32 = static_cast<int32_t>(group_size), pn32 = static_cast<int32_t>(padded_n);
  if (scale.scalar_type() == at::kHalf) {
    if (out_dtype == at::kHalf)
      int4_fused_gemm_caller<
          firefly_fused::cutlass_2x_gemm_fused_int4<cutlass::half_t, __half>>(
          out, x_q, w_marlin, scale, zp, c_n, x_s, n32, k32, gs32, pn32);
    else
      int4_fused_gemm_caller<
          firefly_fused::cutlass_2x_gemm_fused_int4<cutlass::bfloat16_t, __half>>(
          out, x_q, w_marlin, scale, zp, c_n, x_s, n32, k32, gs32, pn32);
  } else {
    if (out_dtype == at::kHalf)
      int4_fused_gemm_caller<
          firefly_fused::cutlass_2x_gemm_fused_int4<cutlass::half_t, __nv_bfloat16>>(
          out, x_q, w_marlin, scale, zp, c_n, x_s, n32, k32, gs32, pn32);
    else
      int4_fused_gemm_caller<
          firefly_fused::cutlass_2x_gemm_fused_int4<cutlass::bfloat16_t, __nv_bfloat16>>(
          out, x_q, w_marlin, scale, zp, c_n, x_s, n32, k32, gs32, pn32);
  }
  return out;
}

// ---- 调试用: 独立反量化 kernel(与 GEMM B-load 里逐元素数学一致) ----
// 用于把 "反量化数学" 与 "mma/epilogue/cutlass 版本" 分离定位。
__global__ void fp8_dequant_kernel(uint8_t const* __restrict__ w_fp8,
                                   float const* __restrict__ block_scale,
                                   float const* __restrict__ c_n, int8_t* __restrict__ w_int8,
                                   int N, int K) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(N) * K;
  if (idx >= total) return;
  int n = static_cast<int>(idx / K);
  int k = static_cast<int>(idx % K);
  int ns = n / 128;
  int ks = k / 128;
  float w = firefly_fused::fp8_to_float(w_fp8[idx]);
  float bsv = block_scale[ns * (K / 128) + ks];
  float wdeq = w * bsv;
  float c = c_n[n];
  int v = (c > 0.f) ? __float2int_rn(wdeq / c) : 0;
  if (v > 127) v = 127;
  if (v < -127) v = -127;
  w_int8[idx] = static_cast<int8_t>(v);
}

// 调试: 独立 int4 反量化(与 GEMM B-load 完全相同的数学: fetch_q + (q-zp)*s /c_n),
// 但用干净 (n,k) 遍历(不走 GEMM 坐标)。与非 fused dequant_marlin_to_int8 对比:
// 一致 → 数学对, bug 在 GEMM B-load 坐标; 不一致 → 数学本身错。
template <typename ScalarT>
__global__ void int4_dequant_kernel(const int32_t* __restrict__ marlin,
                                    const ScalarT* __restrict__ scale,
                                    const float* __restrict__ zp,
                                    const float* __restrict__ c_n,
                                    int8_t* __restrict__ w_int8, int N, int K,
                                    int group_size, int n_tiles_global) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(N) * K;
  if (idx >= total) return;
  int n = static_cast<int>(idx / K);
  int k = static_cast<int>(idx % K);
  int scale_cols = K / group_size;
  int q = firefly_fused::ff_fetch_q(marlin, n, k, n_tiles_global);
  float s = firefly_fused::ff_scalar_to_float(scale[idx / K * scale_cols + k / group_size]);
  float z = zp ? zp[idx / K * scale_cols + k / group_size] : 8.0f;
  float wdeq = static_cast<float>(q - z) * s;
  float c = c_n[n];
  int v = (c > 0.f) ? __float2int_rn(wdeq / c) : 0;
  if (v > 127) v = 127;
  if (v < -127) v = -127;
  w_int8[idx] = static_cast<int8_t>(v);
}

at::Tensor int4_dequant_only(at::Tensor w_marlin, at::Tensor scale, at::Tensor zp,
                             at::Tensor c_n, int64_t size_k, int64_t size_n,
                             int64_t group_size, int64_t padded_n) {
  TORCH_CHECK(w_marlin.is_cuda());
  int64_t N = size_n, K = size_k;
  int64_t n_tiles_global = padded_n / 64;
  at::Tensor w_int8 = at::empty({N, K}, at::TensorOptions().dtype(at::kChar).device(w_marlin.device()));
  int64_t total = N * K;
  int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  auto stream = c10::cuda::getCurrentCUDAStream(w_marlin.device().index()).stream();
  const int32_t* marlin = static_cast<const int32_t*>(w_marlin.data_ptr());
  const float* zp_ptr =
      (zp.defined() && zp.numel() > 0) ? static_cast<const float*>(zp.data_ptr()) : nullptr;
  const float* cn = static_cast<const float*>(c_n.data_ptr());
  if (scale.scalar_type() == at::kHalf) {
    int4_dequant_kernel<__half><<<blocks, threads, 0, stream>>>(
        marlin, static_cast<const __half*>(scale.data_ptr()), zp_ptr, cn,
        static_cast<int8_t*>(w_int8.data_ptr()), static_cast<int>(N),
        static_cast<int>(K), static_cast<int>(group_size),
        static_cast<int>(n_tiles_global));
  } else {
    int4_dequant_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        marlin, static_cast<const __nv_bfloat16*>(scale.data_ptr()), zp_ptr, cn,
        static_cast<int8_t*>(w_int8.data_ptr()), static_cast<int>(N),
        static_cast<int>(K), static_cast<int>(group_size),
        static_cast<int>(n_tiles_global));
  }
  return w_int8;
}

at::Tensor fp8_dequant_only(at::Tensor w_fp8, at::Tensor block_scale, at::Tensor c_n) {
  TORCH_CHECK(w_fp8.is_cuda());
  int64_t N = w_fp8.size(0), K = w_fp8.size(1);
  at::Tensor w_int8 = at::empty({N, K}, at::TensorOptions().dtype(at::kChar).device(w_fp8.device()));
  int64_t total = N * K;
  int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  auto stream = c10::cuda::getCurrentCUDAStream(w_fp8.device().index()).stream();
  fp8_dequant_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<uint8_t const*>(w_fp8.data_ptr()),
      static_cast<float const*>(block_scale.data_ptr()),
      static_cast<float const*>(c_n.data_ptr()),
      static_cast<int8_t*>(w_int8.data_ptr()), N, K);
  return w_int8;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  // NOLINT
  m.def("fp8_fused_scaled_mm", &fp8_fused_scaled_mm,
        "firefly fused: fp8 weight dequant-on-load int8 GEMM (bit-exact vs int8_prefill_linear)");
  m.def("fp8_dequant_only", &fp8_dequant_only,
        "debug: standalone fp8 -> int8 dequant (same per-element math as the GEMM B-load)");
  m.def("int4_fused_scaled_mm", &int4_fused_scaled_mm,
        "firefly fused: int4(W4A16) marlin weight dequant-on-load int8 GEMM "
        "(bit-exact vs int8_prefill_linear via dequant_marlin_to_int8_cached)");
  m.def("int4_dequant_only", &int4_dequant_only,
        "debug: standalone int4 marlin -> int8 dequant (same math as the GEMM B-load)");
}
