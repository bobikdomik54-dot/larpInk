#include <torch/extension.h>
#include <tuple>
#include <vector>

std::tuple<torch::Tensor, torch::Tensor> dora_3x_fp8_fwd_cuda(
    torch::Tensor x2,
    torch::Tensor w0_fp8,
    torch::Tensor sw,
    torch::Tensor sx,
    torch::Tensor inv_sx,
    torch::Tensor la_cat,
    torch::Tensor b_cat,
    torch::Tensor m_cat,
    torch::Tensor inv_n_cat,
    double lora_scale
);

std::vector<double> dora_3x_fp8_fwd_profile_cuda(
    torch::Tensor x2,
    torch::Tensor w0_fp8,
    torch::Tensor sw,
    torch::Tensor sx,
    torch::Tensor inv_sx,
    torch::Tensor la_cat,
    torch::Tensor b_cat,
    torch::Tensor m_cat,
    torch::Tensor inv_n_cat,
    double lora_scale
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> dora_3x_fp8_bwd_cuda(
    torch::Tensor go2,
    torch::Tensor out,
    torch::Tensor x2,
    torch::Tensor w0_T_fp8,
    torch::Tensor sw,
    torch::Tensor sdy,
    torch::Tensor inv_sdy,
    torch::Tensor la_cat,
    torch::Tensor b_cat,
    torch::Tensor m_cat,
    torch::Tensor inv_n_cat,
    torch::Tensor h_a_cat,
    double lora_scale
);

std::tuple<torch::Tensor, torch::Tensor> fa_inkling_fwd_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor rel_states,
    torch::Tensor rel_proj,
    torch::Tensor sinks_2d,
    torch::Tensor doc_id,
    torch::Tensor q_n_start,
    double sm_scale,
    int64_t sliding_win
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fa_inkling_bwd_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor rel_states,
    torch::Tensor rel_proj,
    torch::Tensor sinks_2d,
    torch::Tensor doc_id,
    torch::Tensor q_n_start,
    torch::Tensor k_m_start,
    torch::Tensor k_m_end,
    torch::Tensor out,
    torch::Tensor L_buf,
    torch::Tensor grad_out,
    double sm_scale,
    int64_t sliding_win
);

std::vector<double> fa_inkling_bwd_profile_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor rel_states, torch::Tensor rel_proj, torch::Tensor sinks_2d,
    torch::Tensor doc_id, torch::Tensor q_n_start, torch::Tensor k_m_start, torch::Tensor k_m_end,
    torch::Tensor out, torch::Tensor L_buf, torch::Tensor grad_out,
    double sm_scale, int64_t sliding_win
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_route_and_quant_fp8_cuda(
    torch::Tensor x, torch::Tensor top_k_index, torch::Tensor top_k_weights,
    int64_t num_experts, double top_p, int64_t k_min
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
moe_gu_lora_swiglu_fp8_fwd_cuda(
    torch::Tensor gu, torch::Tensor z_gu, torch::Tensor B_gu, torch::Tensor A_dn,
    torch::Tensor sorted_tok, torch::Tensor sorted_e, torch::Tensor active_tokens,
    double lora_scale, bool has_lora
);

torch::Tensor moe_dn_lora_scatter_fwd_cuda(
    torch::Tensor y_exp, torch::Tensor z_dn, torch::Tensor B_dn,
    torch::Tensor inv_idx, torch::Tensor sorted_tok, torch::Tensor sorted_e,
    torch::Tensor active_tokens, torch::Tensor w_eff,
    double lora_scale, bool has_lora
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_bwd_gather_dw_quant_fp8_cuda(
    torch::Tensor dy, torch::Tensor y_exp, torch::Tensor B_dn,
    torch::Tensor sorted_tok, torch::Tensor sorted_e, torch::Tensor sorted_w,
    torch::Tensor inv_idx, torch::Tensor active_tokens,
    double lora_scale, bool has_lora
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
moe_bwd_swiglu_quant_fp8_cuda(
    torch::Tensor gu, torch::Tensor d_act, torch::Tensor dz_dn,
    torch::Tensor A_dn, torch::Tensor B_gu,
    torch::Tensor sorted_e, torch::Tensor active_tokens,
    double lora_scale, bool has_lora
);

std::tuple<torch::Tensor, torch::Tensor>
moe_bwd_scatter_dx_cuda(
    torch::Tensor dx_exp, torch::Tensor dz_gu, torch::Tensor inv_idx, bool has_lora
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dora_3x_fp8_fwd", &dora_3x_fp8_fwd_cuda, "Fused 3x DoRA FP8 Forward (H100 cuBLASLt + WMMA)");
    m.def("dora_3x_fp8_fwd_profile", &dora_3x_fp8_fwd_profile_cuda, "Fused 3x DoRA FP8 Forward Breakdown");
    m.def("dora_3x_fp8_bwd", &dora_3x_fp8_bwd_cuda, "Fused 3x DoRA FP8 Backward (H100 cuBLASLt dual-stream)");
    m.def("fa_inkling_fwd", &fa_inkling_fwd_cuda, "Inkling FA3 Hopper WGMMA Forward");
    m.def("fa_inkling_bwd", &fa_inkling_bwd_cuda, "Inkling FA3 Hopper WGMMA Backward");
    m.def("fa_inkling_bwd_profile", &fa_inkling_bwd_profile_cuda, "Inkling FA3 Backward Breakdown");
    m.def("moe_route_and_quant_fp8", &moe_route_and_quant_fp8_cuda, "MoE Top-P Routing + FP8 Quant");
    m.def("moe_gu_lora_swiglu_fp8_fwd", &moe_gu_lora_swiglu_fp8_fwd_cuda, "MoE Fused LoRA_B_gu + SoftcappedSwiGLU + FP8 Quant");
    m.def("moe_dn_lora_scatter_fwd", &moe_dn_lora_scatter_fwd_cuda, "MoE Fused LoRA_B_dn + Weighted Scatter");
    m.def("moe_bwd_gather_dw_quant_fp8", &moe_bwd_gather_dw_quant_fp8_cuda, "MoE BWD Gather + dW + FP8 Quant + dz_dn");
    m.def("moe_bwd_swiglu_quant_fp8", &moe_bwd_swiglu_quant_fp8_cuda, "MoE BWD SoftcappedSwiGLU + FP8 Quant + dz_gu");
    m.def("moe_bwd_scatter_dx", &moe_bwd_scatter_dx_cuda, "MoE BWD Scatter dx + dz_gu_tok");
}
