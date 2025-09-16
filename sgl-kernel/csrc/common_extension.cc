/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ATen.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"
#include "cpu/shm.h"

// silu_and_mul
at::Tensor silu_and_mul_cpu(at::Tensor& input);

// gelu_and_mul
at::Tensor gelu_tanh_and_mul_cpu(const at::Tensor& input);
at::Tensor gelu_and_mul_cpu(const at::Tensor& input);

// l2norm
at::Tensor l2norm_cpu(at::Tensor& input, double eps);

// rmsnorm
at::Tensor rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps);

// fused_add_rmsnorm
void fused_add_rmsnorm_cpu(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps);

// topk
std::tuple<at::Tensor, at::Tensor>
topk_sigmoid_cpu(at::Tensor& hidden_states, at::Tensor& gating_output, int64_t topk, bool renormalize);
std::tuple<at::Tensor, at::Tensor>
topk_softmax_cpu(at::Tensor& hidden_states, at::Tensor& gating_output, int64_t topk, bool renormalize);

std::tuple<at::Tensor, at::Tensor> grouped_topk_cpu(
    at::Tensor& hidden_states,
    at::Tensor& gating_output,
    int64_t topk,
    bool renormalize,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t num_fused_shared_experts,
    std::optional<double> routed_scaling_factor,
    std::optional<at::Tensor> num_token_non_padded);

std::tuple<at::Tensor, at::Tensor> biased_grouped_topk_cpu(
    at::Tensor& hidden_states,
    at::Tensor& gating_output,
    at::Tensor& correction_bias,
    int64_t topk,
    bool renormalize,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t num_fused_shared_experts,
    std::optional<double> routed_scaling_factor,
    std::optional<at::Tensor> num_token_non_padded);

// attention
void decode_attention_cpu(
    at::Tensor& query,
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    at::Tensor& output,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& loc,
    at::Tensor& attn_logits,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    double sm_scale,
    double logit_cap);

void extend_attention_cpu(
    at::Tensor& q_extend,
    at::Tensor& k_extend,
    at::Tensor& v_extend,
    at::Tensor& o_extend,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_seq_lens,
    at::Tensor& extend_start_loc,
    int64_t max_len_extend,
    double sm_scale,
    double logit_cap);

// weight prepack
at::Tensor convert_weight_packed(at::Tensor& weight);

// quant
std::tuple<at::Tensor, at::Tensor> per_token_quant_int8_cpu(at::Tensor& A);

// gemm
at::Tensor
weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2, const std::optional<at::Tensor>& bias, bool is_vnni);

// igemm
at::Tensor int8_scaled_mm_cpu(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales1,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni);

// fp8 gemm
at::Tensor fp8_scaled_mm_cpu(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    std::vector<int64_t> block_size,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni);

// quant + igemm
at::Tensor int8_scaled_mm_with_quant(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni);

// bmm
void bmm_cpu(at::Tensor& out, at::Tensor& mat1, at::Tensor& mat2, bool is_vnni, const std::optional<at::Tensor>& scale);

// fused moe
at::Tensor fused_experts_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size,
    const std::optional<at::Tensor>& a1_scale,
    const std::optional<at::Tensor>& a2_scale,
    bool is_vnni);

at::Tensor shared_expert_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& fused_experts_out,
    double routed_scaling_factor,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size,
    const std::optional<at::Tensor>& a1_scale,
    const std::optional<at::Tensor>& a2_scale,
    bool is_vnni);

// weight absorption
std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_proj_with_rope(
    at::Tensor& hidden_states,
    at::Tensor& q_a_proj_weight,
    at::Tensor& q_b_proj_weight,
    at::Tensor& kv_a_proj_weight,
    at::Tensor& w_kc,
    at::Tensor& q_a_layernorm_weight,
    at::Tensor& kv_a_layernorm_weight,
    at::Tensor& positions,
    at::Tensor& cos_sin_cache,
    double eps,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    std::optional<at::Tensor> q_a_proj_scale,
    std::optional<at::Tensor> q_b_proj_scale,
    std::optional<at::Tensor> kv_a_proj_scale,
    bool is_vnni,
    std::optional<std::vector<int64_t>> block_size);

std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_proj_with_rope_fused_weight(
    at::Tensor& hidden_states,
    at::Tensor& qkv_a_proj_weight,
    at::Tensor& q_b_proj_weight,
    at::Tensor& w_kc,
    at::Tensor& q_a_layernorm_weight,
    at::Tensor& kv_a_layernorm_weight,
    at::Tensor& positions,
    at::Tensor& cos_sin_cache,
    double eps,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    std::optional<at::Tensor> qkv_a_proj_scale,
    std::optional<at::Tensor> q_b_proj_scale,
    bool is_vnni,
    std::optional<std::vector<int64_t>> block_size,
    int64_t q_lora_rank,
    int64_t kv_lora_rank,
    int64_t qk_rope_head_dim);

// shared memory init
void initialize(int64_t size, int64_t rank);

// shared mmeory all_reduce
void shm_allreduce(at::Tensor& data, int64_t op);

// shared memory all_gather
at::Tensor shm_allgather(at::Tensor& data, int64_t dim);

// rope
std::tuple<at::Tensor, at::Tensor> rotary_embedding_cpu(
    at::Tensor& positions,
    at::Tensor& query,
    at::Tensor& key,
    int64_t head_size,
    at::Tensor& cos_sin_cache,
    bool is_neox);

// CPU and memory binding
std::string init_cpu_threads_env(const std::string& cpu_ids);

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  /*
   * From csrc/allreduce
   */
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);
  m.def("dispose", &dispose);
  m.def("meta_size", &meta_size);
  m.def("register_buffer", &register_buffer);

  m.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool full_nvlink) -> int");
  m.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);

  m.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  m.impl("all_reduce", torch::kCUDA, &all_reduce);

  m.def("mscclpp_generate_unique_id", &mscclpp_generate_unique_id);
  m.def(
      "mscclpp_init_context(Tensor unique_id, int rank, int world_size, Tensor scratch, Tensor put_buffer, "
      "int nranks_per_node, int[] rank_to_node, int[] rank_to_ib, int context_selection) -> int");
  m.impl("mscclpp_init_context", torch::kCUDA, &mscclpp_init_context);

  m.def("mscclpp_allreduce(int context, Tensor inp, Tensor! out, int nthreads, int nblocks) -> ()");
  m.impl("mscclpp_allreduce", torch::kCUDA, &mscclpp_allreduce);

  /*
   * From csrc/attention
   */
  m.def(
      "lightning_attention_decode(Tensor q, Tensor k, Tensor v, Tensor past_kv, Tensor slope, Tensor! output, Tensor! "
      "new_kv) -> ()");
  m.impl("lightning_attention_decode", torch::kCUDA, &lightning_attention_decode);
  m.def("merge_state(Tensor v_a, Tensor s_a, Tensor v_b, Tensor s_b, Tensor! v_merged, Tensor! s_merged) -> ()");
  m.impl("merge_state", torch::kCUDA, &merge_state);
  m.def("merge_state_v2(Tensor v_a, Tensor s_a, Tensor v_b, Tensor s_b, Tensor! v_merged, Tensor! s_merged) -> ()");
  m.impl("merge_state_v2", torch::kCUDA, &merge_state_v2);
  m.def(
      "cutlass_mla_decode(Tensor! out, Tensor q_nope, Tensor q_pe, Tensor kv_c_and_k_pe_cache, Tensor seq_lens, Tensor "
      "page_table, Tensor! workspace, float sm_scale, int num_kv_splits) -> ()");
  m.impl("cutlass_mla_decode", torch::kCUDA, &cutlass_mla_decode);
  m.def("cutlass_mla_get_workspace_size", &cutlass_mla_get_workspace_size);

  /*
   * From csrc/elementwise
   */
  m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("rmsnorm", torch::kCUDA, &rmsnorm);

  m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("fused_add_rmsnorm", torch::kCUDA, &sgl_fused_add_rmsnorm);

  m.def("gemma_rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("gemma_rmsnorm", torch::kCUDA, &gemma_rmsnorm);

  m.def("gemma_fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("gemma_fused_add_rmsnorm", torch::kCUDA, &gemma_fused_add_rmsnorm);

  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  m.def(
      "apply_rope_pos_ids_cos_sin_cache(Tensor q, Tensor k, Tensor! q_rope, Tensor! k_rope, Tensor cos_sin_cache, "
      "Tensor pos_ids, bool interleave, bool enable_pdl, int cuda_stream, "
      "Tensor? v, Tensor!? k_buffer, Tensor!? v_buffer, Tensor? kv_cache_loc) -> ()");
  m.impl("apply_rope_pos_ids_cos_sin_cache", torch::kCUDA, &apply_rope_pos_ids_cos_sin_cache);

  m.def(
      "downcast_fp8(Tensor k, Tensor v, Tensor k_out, Tensor v_out, Tensor k_scale, Tensor v_scale, Tensor loc, int "
      "mult, int offset, int cuda_stream) -> ()");
  m.impl("downcast_fp8", torch::kCUDA, &downcast_fp8);

  m.def("copy_to_gpu_no_ce(Tensor input, Tensor! output) -> ()");
  m.impl("copy_to_gpu_no_ce", torch::kCUDA, &copy_to_gpu_no_ce);
  m.def("concat_mla_k(Tensor! k, Tensor k_nope, Tensor k_rope) -> ()");
  m.impl("concat_mla_k", torch::kCUDA, &concat_mla_k);

  /*
   * From csrc/gemm
   */
  m.def("awq_dequantize(Tensor qweight, Tensor scales, Tensor qzeros) -> Tensor");
  m.impl("awq_dequantize", torch::kCUDA, &awq_dequantize);

  m.def(
      "int8_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype, Tensor? "
      "bias) -> Tensor");
  m.impl("int8_scaled_mm", torch::kCUDA, &int8_scaled_mm);

  m.def(
      "fp8_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype, Tensor? "
      "bias) -> Tensor");
  m.impl("fp8_scaled_mm", torch::kCUDA, &fp8_scaled_mm);

  m.def(
      "fp8_blockwise_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype) -> "
      "Tensor");
  m.impl("fp8_blockwise_scaled_mm", torch::kCUDA, &fp8_blockwise_scaled_mm);

  m.def(
      "sgl_per_token_group_quant_fp8(Tensor input, Tensor output_q, Tensor output_s, int group_size,"
      " float eps, float fp8_min, float fp8_max, bool scale_ue8m0) -> ()");
  m.impl("sgl_per_token_group_quant_fp8", torch::kCUDA, &sgl_per_token_group_quant_fp8);

  m.def(
      "sgl_per_token_group_quant_int8(Tensor input, Tensor output_q, Tensor output_s, int group_size,"
      " float eps, float int8_min, float int8_max) -> ()");
  m.impl("sgl_per_token_group_quant_int8", torch::kCUDA, &sgl_per_token_group_quant_int8);

  m.def("sgl_per_tensor_quant_fp8(Tensor input, Tensor output_q, Tensor output_s, bool is_static) -> ()");
  m.impl("sgl_per_tensor_quant_fp8", torch::kCUDA, &sgl_per_tensor_quant_fp8);

  m.def("sgl_per_token_quant_fp8(Tensor input, Tensor output_q, Tensor output_s) -> ()");
  m.impl("sgl_per_token_quant_fp8", torch::kCUDA, &sgl_per_token_quant_fp8);

  m.def(
      "cutlass_scaled_fp4_mm(Tensor! out, Tensor a, Tensor b,"
      "                      Tensor block_scale_a, Tensor block_scale_b,"
      "                      Tensor alpha) -> ()");
  m.impl("cutlass_scaled_fp4_mm", torch::kCUDA, &cutlass_scaled_fp4_mm);

  m.def(
      "scaled_fp4_quant(Tensor! output, Tensor! input,"
      "                 Tensor! output_scale, Tensor! input_scale) -> ()");
  m.impl("scaled_fp4_quant", torch::kCUDA, &scaled_fp4_quant);

  m.def("dsv3_fused_a_gemm(Tensor! output, Tensor mat_a, Tensor mat_b) -> ()");
  m.impl("dsv3_fused_a_gemm", torch::kCUDA, &dsv3_fused_a_gemm);

  // Compute NVFP4 experts quantization.
  m.def(
      "scaled_fp4_experts_quant(Tensor! output, Tensor! output_scale,"
      "Tensor input, Tensor input_global_scale, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts) -> ()");
  m.impl("scaled_fp4_experts_quant", torch::kCUDA, &scaled_fp4_experts_quant);

  m.def(
      "silu_and_mul_scaled_fp4_experts_quant(Tensor! output, Tensor! output_scale,"
      "Tensor input, Tensor input_global_scale, Tensor mask, bool use_silu_and_mul) -> ()");
  m.impl("silu_and_mul_scaled_fp4_experts_quant", torch::kCUDA, &silu_and_mul_scaled_fp4_experts_quant);

  m.def(
      "cutlass_fp4_group_mm(Tensor! output, Tensor a, Tensor b,"
      "Tensor a_blockscale, Tensor b_blockscale, Tensor alphas,"
      "Tensor ab_strides, Tensor c_strides, Tensor problem_sizes,"
      " Tensor expert_offsets, Tensor sf_offsets) -> ()");
  m.impl("cutlass_fp4_group_mm", torch::kCUDA, &cutlass_fp4_group_mm);

  m.def("dsv3_router_gemm(Tensor! output, Tensor mat_a, Tensor mat_b) -> ()");
  m.impl("dsv3_router_gemm", torch::kCUDA, &dsv3_router_gemm);

  /*
   * From csrc/gemm/gptq
   */
  m.def(
      "gptq_marlin_gemm(Tensor! a, Tensor? c_or_none,"
      "Tensor! b_q_weight, Tensor! b_scales, Tensor? global_scale_or_none,"
      "Tensor? b_zeros_or_none, Tensor? g_idx_or_none, Tensor? perm_or_none,"
      "Tensor! workspace, int b_q_type_id, int size_m, int size_n, int size_k,"
      "bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  m.impl("gptq_marlin_gemm", torch::kCUDA, &gptq_marlin_gemm);

  m.def(
      "gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, Tensor b_gptq_scales, Tensor b_g_idx, bool "
      "use_shuffle, int bit) -> Tensor");
  m.impl("gptq_gemm", torch::kCUDA, &gptq_gemm);

  m.def("gptq_shuffle(Tensor! q_weight, Tensor q_perm, int bit) -> ()");
  m.impl("gptq_shuffle", torch::kCUDA, &gptq_shuffle);

  m.def("gptq_marlin_repack(Tensor! b_q_weight, Tensor! perm, int size_k, int size_n, int num_bits) -> Tensor");
  m.impl("gptq_marlin_repack", torch::kCUDA, &gptq_marlin_repack);

  m.def("awq_marlin_repack(Tensor! b_q_weight, int size_k, int size_n, int num_bits) -> Tensor");
  m.impl("awq_marlin_repack", torch::kCUDA, &awq_marlin_repack);

  /*
   * From csrc/moe
   */
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! "
      "experts_ids, Tensor! num_tokens_post_pad, Tensor! cumsum_buffer, bool "
      "pad_sorted_token_ids) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  m.def("topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output, bool renormalize) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  m.def(
      "moe_fused_gate(Tensor input, Tensor bias, int num_expert_group, int topk_group, int topk, int "
      "num_fused_shared_experts, float routed_scaling_factor, bool apply_routed_scaling_factor_on_output) -> "
      "(Tensor[])");
  m.impl("moe_fused_gate", torch::kCUDA, &moe_fused_gate);
  m.def(
      "fp8_blockwise_scaled_grouped_mm(Tensor output, Tensor a_ptrs, Tensor b_ptrs, Tensor out_ptrs, Tensor "
      "a_scales_ptrs, Tensor b_scales_ptrs, Tensor a, Tensor b, Tensor scales_a, Tensor scales_b, Tensor "
      "stride_a, Tensor stride_b, Tensor stride_c, Tensor layout_sfa, Tensor layout_sfb, Tensor problem_sizes, Tensor "
      "expert_offsets, Tensor workspace) -> ()");
  m.impl("fp8_blockwise_scaled_grouped_mm", torch::kCUDA, &fp8_blockwise_scaled_grouped_mm);
  m.def(
      "prepare_moe_input(Tensor topk_ids, Tensor expert_offsets, Tensor? blockscale_offsets, Tensor problem_sizes1,"
      " Tensor problem_sizes2, Tensor input_permutation, Tensor output_permutation, int num_experts, int n, int k) -> "
      "()");
  m.impl("prepare_moe_input", torch::kCUDA, &prepare_moe_input);

  m.def("shuffle_rows(Tensor input, Tensor dst2src_map, Tensor output) -> ()");
  m.impl("shuffle_rows", torch::kCUDA, &shuffle_rows);
  m.def("apply_shuffle_mul_sum(Tensor input, Tensor output, Tensor permutation, Tensor? factors) -> ()");
  m.impl("apply_shuffle_mul_sum", torch::kCUDA, &apply_shuffle_mul_sum);

  /*
   * From csrc/moe/marlin_moe_wna16
   */
  m.def(
      "moe_wna16_marlin_gemm(Tensor! a, Tensor? c_or_none,"
      "Tensor! b_q_weight, Tensor! b_scales, Tensor? b_zeros_or_none,"
      "Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor! workspace,"
      "Tensor sorted_token_ids,"
      "Tensor! expert_ids, Tensor! num_tokens_past_padded,"
      "Tensor! topk_weights, int moe_block_size, int top_k, "
      "bool mul_topk_weights, bool is_ep, int b_q_type_id,"
      "int size_m, int size_n, int size_k,"
      "bool is_k_full, bool use_atomic_add,"
      "bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  m.impl("moe_wna16_marlin_gemm", torch::kCUDA, &moe_wna16_marlin_gemm);

  /*
   * From csrc/moe/cutlass_moe/w4a8
   */
  m.def(
      "get_cutlass_w4a8_moe_mm_data(Tensor topk_ids, Tensor! expert_offsets, "
      "                        Tensor! problem_sizes1, Tensor! problem_sizes2, "
      "                        Tensor! input_permutation, "
      "                        Tensor! output_permutation, int num_experts, "
      "                        int n, int k) -> ()");
  m.impl("get_cutlass_w4a8_moe_mm_data", torch::kCUDA, &get_cutlass_w4a8_moe_mm_data);

  m.def(
      "cutlass_w4a8_moe_mm(Tensor! d, Tensor a, Tensor b, "
      "               Tensor a_scales, Tensor b_scales, Tensor expert_offsets, "
      "               Tensor problem_sizes, Tensor a_strides, "
      "               Tensor b_strides, Tensor d_strides, Tensor s_strides,"
      "               int chunk_size, int topk) -> ()");
  m.impl("cutlass_w4a8_moe_mm", torch::kCUDA, &cutlass_w4a8_moe_mm);

  /*
   * From csrc/speculative
   */
  m.def(
      "tree_speculative_sampling_target_only(Tensor! predicts, Tensor! accept_index, Tensor! accept_token_num, "
      "Tensor candidates, Tensor retrive_index, Tensor retrive_next_token, Tensor retrive_next_sibling, "
      "Tensor uniform_samples, Tensor uniform_samples_for_final_sampling, Tensor target_probs, Tensor draft_probs, "
      "float threshold_single, float threshold_acc, "
      "bool deterministic, int cuda_stream) -> ()");
  m.impl("tree_speculative_sampling_target_only", torch::kCUDA, &tree_speculative_sampling_target_only);

  m.def(
      "verify_tree_greedy(Tensor! predicts, Tensor! accept_index, Tensor! accept_token_num, "
      "Tensor candidates, Tensor retrive_index, Tensor retrive_next_token, Tensor retrive_next_sibling, "
      "Tensor target_predict, int cuda_stream) -> ()");
  m.impl("verify_tree_greedy", torch::kCUDA, &verify_tree_greedy);

  m.def(
      "build_tree_kernel_efficient(Tensor parent_list, Tensor selected_index, Tensor verified_seq_len, "
      "Tensor! tree_mask, Tensor! positions, Tensor! retrive_index, Tensor! retrive_next_token, "
      "Tensor! retrive_next_sibling, int topk, int depth, int draft_token_num, int tree_mask_mode) -> "
      "()");
  m.impl("build_tree_kernel_efficient", torch::kCUDA, &build_tree_kernel_efficient);

  m.def(
      "segment_packbits(Tensor x, Tensor input_indptr, Tensor output_indptr, Tensor! y, int batch_size, "
      "int cuda_stream) -> ()");
  m.impl("segment_packbits", torch::kCUDA, &segment_packbits);

  /*
   * From csrc/kvcacheio
   */
  m.def(
      "transfer_kv_per_layer(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor "
      "dst_indices, int item_size, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer", torch::kCUDA, &transfer_kv_per_layer);
  m.def(
      "transfer_kv_per_layer_pf_lf(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor "
      "dst_indices, int layer_id, int item_size, int src_layout_dim, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_pf_lf", torch::kCUDA, &transfer_kv_per_layer_pf_lf);
  m.def(
      "transfer_kv_all_layer(Tensor src_k_layers, Tensor dst_k_layers, Tensor src_v_layers, Tensor dst_v_layers, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int num_layers, int block_quota, int "
      "num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer", torch::kCUDA, &transfer_kv_all_layer);
  m.def(
      "transfer_kv_all_layer_lf_pf(Tensor src_k_layers, Tensor dst_k, Tensor src_v_layers, Tensor dst_v, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int dst_layout_dim, int num_layers, int block_quota, int "
      "num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_lf_pf", torch::kCUDA, &transfer_kv_all_layer_lf_pf);
  m.def(
      "transfer_kv_per_layer_mla(Tensor src, Tensor dst, Tensor src_indices, Tensor dst_indices, int item_size, int "
      "block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_mla", torch::kCUDA, &transfer_kv_per_layer_mla);
  m.def(
      "transfer_kv_per_layer_mla_pf_lf(Tensor src, Tensor dst, Tensor src_indices, Tensor dst_indices, int layer_id, "
      "int item_size, int src_layout_dim, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_mla_pf_lf", torch::kCUDA, &transfer_kv_per_layer_mla_pf_lf);
  m.def(
      "transfer_kv_all_layer_mla(Tensor src_layers, Tensor dst_layers, Tensor src_indices, Tensor dst_indices, int "
      "item_size, int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_mla", torch::kCUDA, &transfer_kv_all_layer_mla);
  m.def(
      "transfer_kv_all_layer_mla_lf_pf(Tensor src_layers, Tensor dst, Tensor src_indices, Tensor dst_indices, "
      "int item_size, int dst_layout_dim, int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_mla_lf_pf", torch::kCUDA, &transfer_kv_all_layer_mla_lf_pf);
  m.def(
      "transfer_kv_direct(Tensor[] src_layers, Tensor[] dst_layers, Tensor src_indices, Tensor dst_indices, int "
      "page_size) -> ()");
  m.impl("transfer_kv_direct", torch::kCUDA, &transfer_kv_direct);
  m.def(
      "transfer_kv_per_layer_direct_pf_lf(Tensor[] src_ptrs, Tensor[] dst_ptrs, Tensor src_indices, "
      "Tensor dst_indices, int layer_id, int page_size)->() ");
  m.impl("transfer_kv_per_layer_direct_pf_lf", torch::kCUDA, &transfer_kv_per_layer_direct_pf_lf);
  m.def(
      "transfer_kv_all_layer_direct_lf_pf(Tensor[] src_ptrs, Tensor[] dst_ptrs, Tensor src_indices, "
      "Tensor dst_indices, int page_size) ->() ");
  m.impl("transfer_kv_all_layer_direct_lf_pf", torch::kCUDA, &transfer_kv_all_layer_direct_lf_pf);

  /*
   * From csrc/memory
   */
  m.def("store_kv_cache(Tensor k_cache, Tensor v_cache, Tensor out_loc, Tensor k, Tensor v) -> ()");
  m.impl("store_kv_cache", &store_kv_cache);

  /*
   * From FlashInfer
   */
  m.def(
      "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, int "
      "cublas_handle, int cuda_stream) -> ()",
      {at::Tag::needs_fixed_stride_order});
  m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);

  m.def(
      "min_p_sampling_from_probs(Tensor probs, Tensor output, Tensor? maybe_indices, Tensor? maybe_min_p_arr, float "
      "min_p_val, bool deterministic, Generator? gen) -> ()");
  m.impl("min_p_sampling_from_probs", torch::kCUDA, &min_p_sampling_from_probs);

  m.def("top_k_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_k_arr, int top_k_val) -> ()");
  m.impl("top_k_renorm_probs", torch::kCUDA, &top_k_renorm_probs);

  m.def("top_p_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_p_arr, float top_p_val) -> ()");
  m.impl("top_p_renorm_probs", torch::kCUDA, &top_p_renorm_probs);

  m.def(
      "top_p_sampling_from_probs(Tensor probs, Tensor output, Tensor? maybe_indices, Tensor? "
      "maybe_top_p_arr, float top_p_val, bool deterministic, Generator? gen) -> ()");
  m.impl("top_p_sampling_from_probs", torch::kCUDA, &top_p_sampling_from_probs);

  m.def(
      "top_k_top_p_sampling_from_probs(Tensor probs, Tensor output, Tensor? maybe_indices, Tensor? maybe_top_k_arr, "
      "float top_k_val, Tensor? maybe_top_p_arr, float top_p_val, bool deterministic, Generator? gen) -> ()");
  m.impl("top_k_top_p_sampling_from_probs", torch::kCUDA, &top_k_top_p_sampling_from_probs);

  m.def("top_k_mask_logits(Tensor logits, Tensor mask_logits, Tensor? maybe_top_k_arr, int top_k_val) -> ()");
  m.impl("top_k_mask_logits", torch::kCUDA, &top_k_mask_logits);

  /*
   * From Sparse Flash Attention
   */
  m.def(
      "fwd_sparse(Tensor! q, Tensor k, Tensor v, "
      "Tensor block_count, Tensor block_offset, Tensor column_count, Tensor column_index, "
      "Tensor!? out, Tensor? alibi_slopes, "
      "float p_dropout, float softmax_scale, bool is_causal, "
      "float softcap, bool return_softmax, Generator? gen)"
      "-> Tensor[]");
  m.impl("fwd_sparse", torch::kCUDA, &flash::mha_fwd_sparse);

  m.def(
      "varlen_fwd_sparse(Tensor! q, Tensor k, Tensor v, "
      "Tensor block_count, Tensor block_offset, Tensor column_count, Tensor column_index, "
      "Tensor!? out, Tensor cu_seqlens_q, "
      "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? alibi_slopes, "
      "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
      "bool is_causal, float softcap, bool return_softmax, "
      "Generator? gen) -> Tensor[]");
  m.impl("varlen_fwd_sparse", torch::kCUDA, &flash::mha_varlen_fwd_sparse);

  // Sparse Attention utils
  m.def(
      "convert_vertical_slash_indexes("
      "   Tensor! block_count, Tensor! block_offset, "
      "   Tensor! column_count, Tensor! column_index, "
      "   Tensor q_seqlens, Tensor q_seqlens, "
      "   Tensor vertical_indexes, Tensor slash_indexes, "
      "   int context_size, int block_size_M, int block_size_N, "
      "   bool causal) -> ()");
  m.impl("convert_vertical_slash_indexes", torch::kCUDA, &convert_vertical_slash_indexes);

  m.def(
      "convert_vertical_slash_indexes_mergehead("
      "   Tensor! block_count, Tensor! block_offset, "
      "   Tensor! column_count, Tensor! column_index, "
      "   Tensor q_seqlens, Tensor q_seqlens, "
      "   Tensor vertical_indexes, Tensor slash_indexes, "
      "   Tensor vertical_indices_count, Tensor slash_indices_count, "
      "   int context_size, int block_size_M, int block_size_N, "
      "   bool causal) -> ()");
  m.impl("convert_vertical_slash_indexes_mergehead", torch::kCUDA, &convert_vertical_slash_indexes_mergehead);

  /*
   * From csrc/grammar
   */
  m.def("apply_token_bitmask_inplace_cuda(Tensor logits, Tensor bitmask, Tensor? indices=None) -> ()");
  m.impl("apply_token_bitmask_inplace_cuda", &ApplyTokenBitmaskInplace);

  /*
   * From csrc/gemm (QServe)
   */
  m.def(
      "qserve_w4a8_per_chn_gemm(Tensor _in_feats, Tensor _kernel, Tensor _wscales, Tensor _ascales, Tensor _w_szs, "
      "Tensor _a_ssums, Tensor! _out_feats) -> ()");
  m.impl("qserve_w4a8_per_chn_gemm", torch::kCUDA, &qserve_w4a8_per_chn_gemm);

  m.def(
      "qserve_w4a8_per_group_gemm(Tensor _in_feats, Tensor _kernel, Tensor _zeros, Tensor _scales_i8, Tensor _wscales, "
      "Tensor _ascales, Tensor! _out_feats) -> ()");
  m.impl("qserve_w4a8_per_group_gemm", torch::kCUDA, &qserve_w4a8_per_group_gemm);

  /*
   * From csrc/mamba
   */
  m.def(
      "causal_conv1d_update(Tensor! x,"
      "Tensor! conv_state,"
      "Tensor! weight,"
      "Tensor? bias_,"
      "bool silu_activation,"
      "Tensor? cache_seqlens_,"
      "Tensor? conv_state_indices,"
      "int pad_slot_id) -> ()");
  m.impl("causal_conv1d_update", torch::kCUDA, &causal_conv1d_update);

  m.def(
      "causal_conv1d_fwd(Tensor! x, Tensor! weight,"
      "Tensor? bias_,"
      "Tensor!? conv_states,"
      "Tensor? query_start_loc,"
      "Tensor? cache_indices,"
      "Tensor? has_initial_state,"
      "bool silu_activation,"
      "int pad_slot_id) -> ()");
  m.impl("causal_conv1d_fwd", torch::kCUDA, &causal_conv1d_fwd);
  // cpu stuff
    // activation
    m.def("silu_and_mul_cpu(Tensor input) -> Tensor");
    m.impl("silu_and_mul_cpu", torch::kCPU, &silu_and_mul_cpu);
    m.def("gelu_tanh_and_mul_cpu(Tensor input) -> Tensor");
    m.impl("gelu_tanh_and_mul_cpu", torch::kCPU, &gelu_tanh_and_mul_cpu);
    m.def("gelu_and_mul_cpu(Tensor input) -> Tensor");
    m.impl("gelu_and_mul_cpu", torch::kCPU, &gelu_and_mul_cpu);

    // norm
    m.def("rmsnorm_cpu(Tensor input, Tensor weight, float eps) -> Tensor");
    m.impl("rmsnorm_cpu", torch::kCPU, &rmsnorm_cpu);
    m.def("l2norm_cpu(Tensor input, float eps) -> Tensor");
    m.impl("l2norm_cpu", torch::kCPU, &l2norm_cpu);
    m.def("fused_add_rmsnorm_cpu(Tensor(a!) input, Tensor residual, Tensor weight, float eps) -> ()");
    m.impl("fused_add_rmsnorm_cpu", torch::kCPU, &fused_add_rmsnorm_cpu);

    // topk
    m.def("topk_sigmoid_cpu(Tensor hidden_states, Tensor gating_output, int topk, bool renormalize) -> (Tensor, Tensor)");
    m.impl("topk_sigmoid_cpu", torch::kCPU, &topk_sigmoid_cpu);
    m.def("topk_softmax_cpu(Tensor hidden_states, Tensor gating_output, int topk, bool renormalize) -> (Tensor, Tensor)");
    m.impl("topk_softmax_cpu", torch::kCPU, &topk_softmax_cpu);
    m.def(
        "grouped_topk_cpu(Tensor hidden_states, Tensor gating_output, int topk, bool renormalize, int num_expert_group, "
        "int topk_group, int num_fused_shared_experts, float? routed_scaling_factor, Tensor? num_token_non_padded) -> "
        "(Tensor, Tensor)");
    m.impl("grouped_topk_cpu", torch::kCPU, &grouped_topk_cpu);

    // biased group topk
    m.def(
        "biased_grouped_topk_cpu(Tensor hidden_states, Tensor gating_output, Tensor correction_bias, int topk, bool "
        "renormalize, int num_expert_group, int topk_group, int num_fused_shared_experts, float? routed_scaling_factor, "
        "Tensor? num_token_non_padded) -> (Tensor, Tensor)");
    m.impl("biased_grouped_topk_cpu", torch::kCPU, &biased_grouped_topk_cpu);

    // decode
    m.def(
        "decode_attention_cpu(Tensor query, Tensor k_cache, Tensor v_cahce, Tensor(a!) output, Tensor key, Tensor value, "
        "Tensor loc, Tensor attn_logits, Tensor req_to_token, Tensor req_pool_indices, Tensor seq_lens, float sm_scale, "
        "float logit_cap) -> ()");
    m.impl("decode_attention_cpu", torch::kCPU, &decode_attention_cpu);

    // extend
    m.def(
        "extend_attention_cpu(Tensor q_extend, Tensor k_extend, Tensor v_extend, Tensor(a!) o_extend, Tensor k_buffer, "
        "Tensor v_buffer, Tensor req_to_token, Tensor req_pool_indices, Tensor seq_lens, Tensor extend_seq_lens, Tensor "
        "extend_start_loc, int max_len_extend, float sm_scale, float logit_cap) -> ()");
    m.impl("extend_attention_cpu", torch::kCPU, &extend_attention_cpu);

    // weight prepack
    m.def("convert_weight_packed(Tensor weight) -> Tensor");
    m.impl("convert_weight_packed", torch::kCPU, &convert_weight_packed);

    // quant
    m.def("per_token_quant_int8_cpu(Tensor A) -> (Tensor, Tensor)");
    m.impl("per_token_quant_int8_cpu", torch::kCPU, &per_token_quant_int8_cpu);

    // gemm
    m.def("weight_packed_linear(Tensor mat1, Tensor mat2, Tensor? bias, bool is_vnni) -> Tensor");
    m.impl("weight_packed_linear", torch::kCPU, &weight_packed_linear);

    // igemm
    m.def(
        "int8_scaled_mm_cpu(Tensor mat1, Tensor mat2, Tensor scales1, Tensor scales2, Tensor? bias, ScalarType "
        "out_dtype, bool is_vnni) -> Tensor");
    m.impl("int8_scaled_mm_cpu", torch::kCPU, &int8_scaled_mm_cpu);

    // fp8 gemm
    m.def(
        "fp8_scaled_mm_cpu(Tensor mat1, Tensor mat2, Tensor scales2, int[] block_size, Tensor? bias, ScalarType "
        "out_dtype, bool is_vnni) -> Tensor");
    m.impl("fp8_scaled_mm_cpu", torch::kCPU, &fp8_scaled_mm_cpu);

    // quant + igemm
    m.def(
        "int8_scaled_mm_with_quant(Tensor mat1, Tensor mat2, Tensor scales2, Tensor? bias, ScalarType out_dtype, bool "
        "is_vnni) -> Tensor");
    m.impl("int8_scaled_mm_with_quant", torch::kCPU, &int8_scaled_mm_with_quant);

    // bmm
    m.def("bmm_cpu(Tensor(a!) out, Tensor mat1, Tensor mat2, bool is_vnni, Tensor? scale) -> ()");
    m.impl("bmm_cpu", torch::kCPU, &bmm_cpu);

    // moe
    m.def(
        "fused_experts_cpu(Tensor hidden_states, Tensor w1, Tensor w2, Tensor topk_weights, Tensor topk_ids, bool "
        "inplace, bool use_int8_w8a8, bool use_fp8_w8a16, Tensor? w1_scale, Tensor? w2_scale, int[]? block_size, Tensor? "
        "a1_scale, Tensor? a2_scale, bool "
        "is_vnni) -> Tensor");
    m.impl("fused_experts_cpu", torch::kCPU, &fused_experts_cpu);

    // weight absorption
    m.def(
        "qkv_proj_with_rope(Tensor hidden_states, Tensor q_a_proj_weight, Tensor q_b_proj_weight, Tensor "
        "kv_a_proj_weight, Tensor w_kc, Tensor q_a_layernorm_weight, Tensor kv_a_layernorm_weight, Tensor positions, "
        "Tensor cos_sin_cache, float eps, bool use_int8_w8a8, bool use_fp8_w8a16, Tensor? q_a_proj_scale, Tensor? "
        "q_b_proj_scale, Tensor? "
        "kv_a_proj_scale, bool is_vnni, int[]? block_size) -> (Tensor, Tensor, Tensor)");
    m.impl("qkv_proj_with_rope", torch::kCPU, &qkv_proj_with_rope);
    m.def(
        "qkv_proj_with_rope_fused_weight(Tensor hidden_states, Tensor qkv_a_proj_weight, Tensor q_b_proj_weight, "
        "Tensor w_kc, Tensor q_a_layernorm_weight, Tensor kv_a_layernorm_weight, Tensor positions, "
        "Tensor cos_sin_cache, float eps, bool use_int8_w8a8, bool use_fp8_w8a16, Tensor? qkv_a_proj_scale, Tensor? "
        "q_b_proj_scale,"
        "bool is_vnni, int[]? block_size, int q_lora_rank, int kv_lora_rank,"
        "int qk_rope_head_dim) -> (Tensor, Tensor, Tensor)");
    m.impl("qkv_proj_with_rope_fused_weight", torch::kCPU, &qkv_proj_with_rope_fused_weight);

    // shared expert
    m.def(
        "shared_expert_cpu(Tensor hidden_states, Tensor w1, Tensor w2, Tensor fused_experts_out, float "
        "routed_scaling_factor, bool inplace, bool use_int8_w8a8, bool use_fp8_w8a16, Tensor? w1_scale, Tensor? "
        "w2_scale, int[]? block_size, Tensor? a1_scale, Tensor? a2_scale, bool is_vnni) -> Tensor");
    m.impl("shared_expert_cpu", torch::kCPU, &shared_expert_cpu);

    // all reduce
    m.def("initialize(int size, int rank) -> ()");
    m.def("shm_allreduce(Tensor(a!) data, int reduce_op) -> ()");
    m.impl("shm_allreduce", torch::kCPU, &shm_allreduce);
    m.def("shm_allgather(Tensor data, int dim) -> Tensor");
    m.impl("shm_allgather", torch::kCPU, &shm_allgather);

    // rope
    m.def(
        "rotary_embedding_cpu(Tensor positions, Tensor query, Tensor key, int head_size, Tensor cos_sin_cache, "
        "bool is_neox) -> (Tensor, Tensor)");
    m.impl("rotary_embedding_cpu", torch::kCPU, &rotary_embedding_cpu);

    // CPU and memory binding
    m.def("init_cpu_threads_env(str cpu_ids) -> str");
}

TORCH_LIBRARY_IMPL(sgl_kernel, CatchAll, m) {
    m.impl("init_cpu_threads_env", init_cpu_threads_env);
    m.impl("initialize", &initialize);
  }

REGISTER_EXTENSION(common_ops)
