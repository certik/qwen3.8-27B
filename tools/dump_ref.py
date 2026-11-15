#!/usr/bin/env python
"""Capture native dense/MoE MLX intermediates and require exact agreement with
ordinary inference logits and all caches. The capture schema is below.

Captures (default: layers 0 and 3, the first linear-attn and first
full-attention layers):

  Universal:
    input_ids.bin                       i32  [Lp]
    x_after_embed.bin                   f32  [Lp, hidden]
    x_after_final_norm.bin              f32  [Lp, hidden]
    logits.bin                          f32  [Lp, vocab]
    next_id.bin                         i32  [Lp]   (argmax per row)

  Per layer li in {0, 3}:
    x_in_<li>.bin                       f32  [Lp, hidden]   (input to layer)
    h_after_input_norm_<li>.bin         f32  [Lp, hidden]
    out_attn_<li>.bin                   f32  [Lp, hidden]   (block residual contrib pre-add)
    x_after_attn_residual_<li>.bin      f32  [Lp, hidden]
    h_after_post_norm_<li>.bin          f32  [Lp, hidden]
    out_mlp_<li>.bin                    f32  [Lp, hidden]   (dense or MoE result)
    x_after_mlp_residual_<li>.bin       f32  [Lp, hidden]

  For full-attention layer (li=3):
    q_proj_<li>.bin                     f32  [Lp, n_heads * 2 * head_dim]  (Q || gate)
    k_proj_<li>.bin                     f32  [Lp, n_kv * head_dim]
    v_proj_<li>.bin                     f32  [Lp, n_kv * head_dim]
    q_after_qnorm_<li>.bin              f32  [Lp, n_heads, head_dim]
    k_after_knorm_<li>.bin              f32  [Lp, n_kv, head_dim]
    q_after_rope_<li>.bin               f32  [Lp, n_heads, head_dim]
    k_after_rope_<li>.bin               f32  [Lp, n_kv, head_dim]
    attn_<li>.bin                       f32  [Lp, n_heads, head_dim]   (SDPA output, pre-gate)
    gate_<li>.bin                       f32  [Lp, n_heads * head_dim]  (output gate logits)
    out_oproj_<li>.bin                  f32  [Lp, hidden]

  For linear-attn layer (li=0):
    qkv_<li>.bin                        f32  [Lp, conv_dim]  (in_proj_qkv)
    z_<li>.bin                          f32  [Lp, value_dim]
    a_<li>.bin                          f32  [Lp, num_v_heads]
    b_<li>.bin                          f32  [Lp, num_v_heads]
    conv_out_<li>.bin                   f32  [Lp, conv_dim]  (post-silu)
    q_lin_<li>.bin                      f32  [Lp, num_k_heads, head_k_dim]   (pre-rms)
    k_lin_<li>.bin                      f32  [Lp, num_k_heads, head_k_dim]
    v_lin_<li>.bin                      f32  [Lp, num_v_heads, head_v_dim]
    q_norm_lin_<li>.bin                 f32  [Lp, num_k_heads, head_k_dim]   (post-rms+scale)
    k_norm_lin_<li>.bin                 f32  [Lp, num_k_heads, head_k_dim]
    g_<li>.bin                          f32  [Lp, num_v_heads]
    beta_<li>.bin                       f32  [Lp, num_v_heads]
    delta_y_<li>.bin                    f32  [Lp, num_v_heads, head_v_dim]   (gated_delta out)
    delta_state_<li>.bin                f32  [num_v_heads, head_v_dim, head_k_dim]
    norm_out_<li>.bin                   f32  [Lp, num_v_heads, head_v_dim]   (after RMSNormGated)
    out_oproj_<li>.bin                  f32  [Lp, hidden]

  Dense branches capture gate_proj/up_proj/swiglu_mid/dense_down.
  For MoE branch:
    router_logits_<li>.bin              f32  [Lp, n_experts]
    router_probs_<li>.bin               f32  [Lp, n_experts]   (post-softmax)
    topk_idx_<li>.bin                   i32  [Lp, top_k]
    topk_score_<li>.bin                 f32  [Lp, top_k]
    gate_proj_<li>.bin                  f32  [Lp, top_k, moe_inter]
    up_proj_<li>.bin                    f32  [Lp, top_k, moe_inter]
    swiglu_mid_<li>.bin                 f32  [Lp, top_k, moe_inter]
    expert_down_<li>.bin                f32  [Lp, top_k, hidden]
    moe_out_<li>.bin                    f32  [Lp, hidden]   (per-expert weighted sum)
    shared_gate_proj_<li>.bin           f32  [Lp, shared_inter]
    shared_up_proj_<li>.bin             f32  [Lp, shared_inter]
    shared_swiglu_<li>.bin              f32  [Lp, shared_inter]
    shared_down_<li>.bin                f32  [Lp, hidden]
    shared_gate_<li>.bin                f32  [Lp, 1]
    shared_y_<li>.bin                   f32  [Lp, hidden]   (post sigmoid-gate)
"""
from __future__ import annotations
import argparse
import json
import os
import struct
import sys

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.activations import swiglu
from reference_support import (eos_ids, model_identity, model_vocab_size, prompt_input,
                               runtime_identity, write_json)
from pathlib import Path


MAGIC = b"LLMTNSR1"
DT_F32, DT_I32 = 0, 1


def save_bin(path: str, arr, dtype: str) -> None:
    if dtype == "f32":
        a = np.asarray(arr, dtype="<f4")
        code = DT_F32
    elif dtype == "i32":
        a = np.asarray(arr, dtype="<i4")
        code = DT_I32
    else:
        raise ValueError(dtype)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", code, a.ndim))
        for s in a.shape:
            f.write(struct.pack("<q", int(s)))
        f.write(a.tobytes())


def to_np_f32(a: mx.array) -> np.ndarray:
    mx.eval(a)
    return np.array(a.astype(mx.float32))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="artifacts/oracle")
    p.add_argument("--user", default="Hello, who are you?")
    p.add_argument("--layers", default="0,3", help="comma-separated layer indices")
    p.add_argument("--input-ids", help="JSON list or manifest with input_ids")
    p.add_argument("--decode-steps", type=int, default=16)
    p.add_argument("--teacher-ids", help="fixed teacher-forced continuation JSON list")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    layer_indices = sorted({int(x) for x in args.layers.split(",")})
    identity = model_identity(args.model, out.parent / "model-identity.json")
    manifest = {"schema": 2, "model_identity": identity, "runtime": runtime_identity(),
                "status": "in_progress", "calls": [], "capture_layers": layer_indices}
    write_json(out / "manifest.json", manifest)
    print(f"loading {args.model} ...", file=sys.stderr)
    model, tokenizer = load(args.model)
    model.eval()
    vocab_size = model_vocab_size(model, tokenizer)
    inputs = prompt_input(tokenizer, args.user, args.input_ids, vocab_size=vocab_size)
    ids = inputs["input_ids"]
    manifest.update({"input": inputs, "eos_ids": eos_ids(args.model, tokenizer),
                     "stop_policy": "fixed steps; EOS recorded, not suppressed",
                     "binary_schema": "LLMTNSR1; LE u32 dtype(0=f32,1=i32), u32 rank, i64 dimensions, data",
                     "storage": "BF16 tensors widened losslessly to F32; native dtype in tensor manifest"})
    teacher = json.loads(Path(args.teacher_ids).read_text()) if args.teacher_ids else None
    if teacher is not None and (len(teacher) < args.decode_steps or
            any(type(x) is not int or not 0 <= x < vocab_size for x in teacher)):
        raise ValueError("invalid or insufficient teacher continuation IDs")
    ordinary_cache, manual_cache = model.make_cache(), model.make_cache()
    current = ids
    generated = []
    for step in range(args.decode_steps + 1):
        call_out = out if step == 0 else out / f"decode-{step:03d}"
        call_out.mkdir(exist_ok=True)
        save_bin(call_out / "input_ids.bin", current, "i32")
        expected = model(mx.array(current)[None], cache=ordinary_cache)
        mx.eval(expected, [c.state for c in ordinary_cache])
        actual = manual_forward(model, current, manual_cache, layer_indices, call_out)
        mx.eval(actual, [c.state for c in manual_cache])
        report = {"step": step, "input_ids": current,
                  "logits": compare(actual, expected, "manual logits")}
        report["caches"] = compare_caches(manual_cache, ordinary_cache)
        save_bin(call_out / "logits.bin", to_np_f32(actual[0]), "f32")
        next_ids = np.array(mx.argmax(actual, axis=-1)[0]).astype(np.int32)
        save_bin(call_out / "next_id.bin", next_ids, "i32")
        report["argmax_per_position"] = next_ids.tolist()
        generated.append(int(next_ids[-1]))
        for li in layer_indices:
            c = manual_cache[li]
            for ci, state in enumerate(c.state):
                if state is not None:
                    save_bin(call_out / f"cache_{li}_{ci}.bin", to_np_f32(state), "f32")
        manifest["calls"].append(report)
        manifest["generated_ids"] = generated
        write_json(out / "manifest.json", manifest)
        print(f"call {step}: {len(current)} input IDs; argmax={generated[-1]}; logits/cache exact",
              file=sys.stderr, flush=True)
        current = [teacher[step] if teacher is not None and step < args.decode_steps
                   else generated[-1]]
    manifest["status"] = "passed"
    manifest["teacher_forced_ids"] = teacher
    reset_out = out / "reset"
    reset_out.mkdir()
    expected_reset_cache, actual_reset_cache = model.make_cache(), model.make_cache()
    expected_reset = model(mx.array(ids)[None], cache=expected_reset_cache)
    actual_reset = manual_forward(model, ids, actual_reset_cache, layer_indices, reset_out)
    manifest["reset_logits"] = compare(actual_reset, expected_reset, "reset logits")
    manifest["reset_cache"] = compare_caches(actual_reset_cache, expected_reset_cache)
    manifest["executed_model_forwards"] = {
        "ordinary": len(manifest["calls"]) + 1,
        "manual": len(manifest["calls"]) + 1,
        "note": "one additional full prefill per path checks new-cache reset/reuse"}
    manifest["tensor_files"] = tensor_inventory(out)
    write_json(out / "manifest.json", manifest)
    print(f"oracle passed: {len(manifest['calls'])} calls; generated IDs={generated}", flush=True)
    return 0


def compare(actual, expected, label):
    a, e = to_np_f32(actual), to_np_f32(expected)
    if a.shape != e.shape or not np.isfinite(a).all() or not np.isfinite(e).all():
        raise AssertionError(f"{label}: invalid shape/non-finite")
    d = float(np.max(np.abs(a - e)))
    if d != 0:
        raise AssertionError(f"{label}: max_abs={d}; exact agreement required")
    return {"max_abs": d, "shape": list(a.shape), "dtype": str(actual.dtype)}


def compare_caches(actual, expected):
    results = []
    for li, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if a.meta_state != e.meta_state:
            raise AssertionError(f"cache metadata mismatch layer {li}")
        for ci, (aa, ee) in enumerate(zip(a.state, e.state, strict=True)):
            if aa is None or ee is None:
                if aa is not ee:
                    raise AssertionError("cache missing")
            else:
                results.append({"layer": li, "slot": ci,
                                **compare(aa, ee, f"cache {li}:{ci}")})
    return results


def tensor_inventory(out):
    from reference_support import sha256
    result = {}
    for path in sorted(out.rglob("*.bin")):
        with path.open("rb") as f:
            f.read(8)
            dtype, rank = struct.unpack("<II", f.read(8))
            shape = struct.unpack("<" + "q" * rank, f.read(8 * rank))
        result[str(path.relative_to(out))] = {
            "dtype_code": dtype, "shape": shape, "bytes": path.stat().st_size,
            "sha256": sha256(path)}
    return result


def manual_forward(model, ids, cache, layer_indices, out):
    lm = model.language_model
    inner = lm.model
    cfg = model.args.text_config
    arr_ids = mx.array(ids)[None]   # [1, Lp]
    x = inner.embed_tokens(arr_ids)
    save_bin(f"{out}/x_after_embed.bin", to_np_f32(x[0]), "f32")

    # Build masks the same way inner.__call__ does.
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    fa_mask = create_attention_mask(x, cache[inner.fa_idx])
    ssm_mask = create_ssm_mask(x, cache[inner.ssm_idx])

    for li, layer in enumerate(inner.layers):
        capture = (li in layer_indices)

        if capture:
            save_bin(f"{out}/x_in_{li}.bin", to_np_f32(x[0]), "f32")

        h = layer.input_layernorm(x)
        if capture:
            save_bin(f"{out}/h_after_input_norm_{li}.bin",
                     to_np_f32(h[0]), "f32")

        if layer.is_linear:
            r = dump_linear_attn(layer.linear_attn, h, ssm_mask, li,
                                 out, capture, cfg, cache[li])
        else:
            r = dump_full_attn(layer.self_attn, h, fa_mask, li,
                               out, capture, cache[li])

        if capture:
            save_bin(f"{out}/out_attn_{li}.bin", to_np_f32(r[0]), "f32")

        x = x + r
        if capture:
            save_bin(f"{out}/x_after_attn_residual_{li}.bin",
                     to_np_f32(x[0]), "f32")

        h2 = layer.post_attention_layernorm(x)
        if capture:
            save_bin(f"{out}/h_after_post_norm_{li}.bin",
                     to_np_f32(h2[0]), "f32")

        r2 = (dump_moe(layer.mlp, h2, li, out, capture, cfg)
              if cfg.get("num_experts", 0) > 0 else dump_dense(layer.mlp, h2, li, out, capture))
        if capture:
            save_bin(f"{out}/out_mlp_{li}.bin", to_np_f32(r2[0]), "f32")
        x = x + r2
        save_bin(f"{out}/x_after_mlp_residual_{li}.bin", to_np_f32(x[0]), "f32")

        # Force eval after every layer to bound memory.
        mx.eval(x)

    final = inner.norm(x)
    save_bin(f"{out}/x_after_final_norm.bin", to_np_f32(final[0]), "f32")

    if lm.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(final)
    else:
        logits = lm.lm_head(final)
    return logits


def dump_full_attn(self_attn, x, mask, li, out, capture, cache=None):
    """Mirror Qwen3NextAttention.__call__ but capture intermediates."""
    import mlx.core as mx

    B, L, D = x.shape
    Nq = self_attn.num_attention_heads
    Nkv = self_attn.num_key_value_heads
    Dh = self_attn.head_dim
    scale = self_attn.scale

    qproj = self_attn.q_proj(x)             # [B, L, 2*Nq*Dh]
    if capture: save_bin(f"{out}/q_proj_{li}.bin", to_np_f32(qproj[0]), "f32")
    queries, gate = mx.split(qproj.reshape(B, L, Nq, -1), 2, axis=-1)   # each [B, L, Nq, Dh]
    gate_flat = gate.reshape(B, L, -1)
    if capture: save_bin(f"{out}/gate_{li}.bin", to_np_f32(gate_flat[0]), "f32")

    keys = self_attn.k_proj(x)
    values = self_attn.v_proj(x)
    if capture:
        save_bin(f"{out}/k_proj_{li}.bin", to_np_f32(keys[0]), "f32")
        save_bin(f"{out}/v_proj_{li}.bin", to_np_f32(values[0]), "f32")

    queries = self_attn.q_norm(queries)
    keys_r = self_attn.k_norm(keys.reshape(B, L, Nkv, -1))
    if capture:
        save_bin(f"{out}/q_after_qnorm_{li}.bin", to_np_f32(queries[0]), "f32")
        save_bin(f"{out}/k_after_knorm_{li}.bin", to_np_f32(keys_r[0]), "f32")

    queries = queries.transpose(0, 2, 1, 3)        # [B, Nq, L, Dh]
    keys_t = keys_r.transpose(0, 2, 1, 3)
    values_t = values.reshape(B, L, Nkv, -1).transpose(0, 2, 1, 3)

    offset = cache.offset if cache is not None else 0
    queries = self_attn.rope(queries, offset=offset)
    keys_t = self_attn.rope(keys_t, offset=offset)
    if capture:
        # transpose back to [L, H, D] for easy comparison with C layout
        save_bin(f"{out}/q_after_rope_{li}.bin",
                 to_np_f32(queries.transpose(0, 2, 1, 3)[0]), "f32")
        save_bin(f"{out}/k_after_rope_{li}.bin",
                 to_np_f32(keys_t.transpose(0, 2, 1, 3)[0]), "f32")

    if cache is not None:
        keys_t, values_t = cache.update_and_fetch(keys_t, values_t)
    from mlx_lm.models.base import scaled_dot_product_attention
    out_sdpa = scaled_dot_product_attention(
        queries, keys_t, values_t, cache=cache, scale=scale, mask=mask
    )
    out_sdpa = out_sdpa.transpose(0, 2, 1, 3)        # [B, L, Nq, Dh]
    if capture: save_bin(f"{out}/attn_{li}.bin", to_np_f32(out_sdpa[0]), "f32")
    out_flat = out_sdpa.reshape(B, L, -1)            # [B, L, Nq*Dh]

    gated = out_flat * mx.sigmoid(gate_flat)
    r = self_attn.o_proj(gated)
    if capture: save_bin(f"{out}/out_oproj_{li}.bin", to_np_f32(r[0]), "f32")
    return r


def dump_linear_attn(ln, x, mask, li, out, capture, cfg, cache=None):
    """Mirror GatedDeltaNet.__call__ but capture intermediates."""
    import mlx.core as mx
    import mlx.nn as nn

    B, S, _ = x.shape
    Hk = ln.num_k_heads; Hv = ln.num_v_heads
    Dk = ln.head_k_dim;  Dv = ln.head_v_dim

    qkv = ln.in_proj_qkv(x)
    z = ln.in_proj_z(x).reshape(B, S, Hv, Dv)
    b = ln.in_proj_b(x)
    a = ln.in_proj_a(x)
    if capture:
        save_bin(f"{out}/qkv_{li}.bin", to_np_f32(qkv[0]), "f32")
        save_bin(f"{out}/z_{li}.bin", to_np_f32(z[0].reshape(S, -1)), "f32")
        save_bin(f"{out}/a_{li}.bin", to_np_f32(a[0]), "f32")
        save_bin(f"{out}/b_{li}.bin", to_np_f32(b[0]), "f32")

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_state = (cache[0] if cache is not None and cache[0] is not None
                  else mx.zeros((B, ln.conv_kernel_size - 1, ln.conv_dim), dtype=x.dtype))
    state = cache[1] if cache is not None else None
    if capture:
        save_bin(f"{out}/conv_state_in_{li}.bin", to_np_f32(conv_state[0]), "f32")
        if state is not None:
            save_bin(f"{out}/delta_state_in_{li}.bin", to_np_f32(state[0]), "f32")
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    if cache is not None:
        if cache.lengths is not None:
            raise NotImplementedError("batched/padded cache lengths are not in the batch1 contract")
        cache[0] = mx.contiguous(conv_input[:, -(ln.conv_kernel_size - 1):, :])
    conv_out = nn.silu(ln.conv1d(conv_input))
    if capture:
        save_bin(f"{out}/conv_out_{li}.bin", to_np_f32(conv_out[0]), "f32")

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [ln.key_dim, 2 * ln.key_dim], -1),
            [Hk, Hk, Hv],
            [Dk, Dk, Dv],
        )
    ]
    if capture:
        save_bin(f"{out}/q_lin_{li}.bin", to_np_f32(q[0]), "f32")
        save_bin(f"{out}/k_lin_{li}.bin", to_np_f32(k[0]), "f32")
        save_bin(f"{out}/v_lin_{li}.bin", to_np_f32(v[0]), "f32")

    inv_scale = Dk ** -0.5
    qn = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    kn = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    if capture:
        save_bin(f"{out}/q_norm_lin_{li}.bin", to_np_f32(qn[0]), "f32")
        save_bin(f"{out}/k_norm_lin_{li}.bin", to_np_f32(kn[0]), "f32")

    from mlx_lm.models.gated_delta import compute_g, gated_delta_update
    beta = mx.sigmoid(b)
    g = compute_g(ln.A_log, a, ln.dt_bias)
    if capture:
        save_bin(f"{out}/g_{li}.bin", to_np_f32(g[0]), "f32")
        save_bin(f"{out}/beta_{li}.bin", to_np_f32(beta[0]), "f32")

    y, state = gated_delta_update(qn, kn, v, a, b, ln.A_log, ln.dt_bias,
                                  state=state, mask=mask, use_kernel=not ln.training)
    if cache is not None:
        cache[1] = state
        cache.advance(S)
    if capture:
        save_bin(f"{out}/delta_y_{li}.bin", to_np_f32(y[0]), "f32")
        save_bin(f"{out}/delta_state_{li}.bin", to_np_f32(state[0]), "f32")

    normed = ln.norm(y, z)
    if capture:
        save_bin(f"{out}/norm_out_{li}.bin", to_np_f32(normed[0]), "f32")
    r = ln.out_proj(normed.reshape(B, S, -1))
    if capture:
        save_bin(f"{out}/out_oproj_{li}.bin", to_np_f32(r[0]), "f32")
    return r


def dump_dense(mlp, x, li, out, capture):
    gate, up = mlp.gate_proj(x), mlp.up_proj(x)
    mid = swiglu(gate, up)
    down = mlp.down_proj(mid)
    if capture:
        for name, value in [("gate_proj", gate), ("up_proj", up),
                            ("swiglu_mid", mid), ("dense_down", down)]:
            save_bin(f"{out}/{name}_{li}.bin", to_np_f32(value[0]), "f32")
    return down


def dump_moe(mlp, x, li, out, capture, cfg):
    """Mirror Qwen3NextSparseMoeBlock.__call__ but capture intermediates."""
    import mlx.core as mx
    import mlx.nn as nn

    logits_r = mlp.gate(x)
    if capture: save_bin(f"{out}/router_logits_{li}.bin", to_np_f32(logits_r[0]), "f32")
    gates = mx.softmax(logits_r, axis=-1, precise=True)
    if capture: save_bin(f"{out}/router_probs_{li}.bin", to_np_f32(gates[0]), "f32")

    k = mlp.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if mlp.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    if capture:
        save_bin(f"{out}/topk_idx_{li}.bin", np.array(inds[0]).astype(np.int32), "i32")
        save_bin(f"{out}/topk_score_{li}.bin", to_np_f32(scores[0]), "f32")

    # Per-expert outputs (manual): we want gate_proj / up_proj / down /
    # mixed result. Reuse the SwitchGLU directly but also call its sub-
    # modules manually for instrumentation.
    B, L, D = x.shape
    H = mlp.switch_mlp.gate_proj.input_dims
    HM = mlp.switch_mlp.gate_proj.output_dims  # moe_intermediate_size
    x_e = mx.expand_dims(x, (-2, -3))           # [B, L, 1, 1, D]
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
    do_sort = inds.size >= 64
    idx = inds
    if do_sort:
        x_e, idx, inv_order = _gather_sort(x_e, inds)
    up_proj = mlp.switch_mlp.up_proj(x_e, idx, sorted_indices=do_sort)
    gate_proj = mlp.switch_mlp.gate_proj(x_e, idx, sorted_indices=do_sort)
    mid = swiglu(gate_proj, up_proj)
    down = mlp.switch_mlp.down_proj(mid, idx, sorted_indices=do_sort)
    if do_sort:
        gate_proj, up_proj, mid, down = [
            _scatter_unsort(value, inv_order, inds.shape)
            for value in [gate_proj, up_proj, mid, down]]
    # Squeeze the inserted singleton axis -3 (the "Hv" placeholder).
    gate_proj = gate_proj.squeeze(-2)
    up_proj   = up_proj.squeeze(-2)
    mid       = mid.squeeze(-2)
    down      = down.squeeze(-2)
    if capture:
        save_bin(f"{out}/gate_proj_{li}.bin", to_np_f32(gate_proj[0]), "f32")
        save_bin(f"{out}/up_proj_{li}.bin", to_np_f32(up_proj[0]), "f32")
        save_bin(f"{out}/swiglu_mid_{li}.bin", to_np_f32(mid[0]), "f32")
        save_bin(f"{out}/expert_down_{li}.bin", to_np_f32(down[0]), "f32")
    y = (down * scores[..., None]).sum(axis=-2)
    if capture: save_bin(f"{out}/moe_out_{li}.bin", to_np_f32(y[0]), "f32")

    # Shared expert (regular MLP) + gate.
    sgp = mlp.shared_expert.gate_proj(x)
    sup = mlp.shared_expert.up_proj(x)
    sm  = swiglu(sgp, sup)
    sd  = mlp.shared_expert.down_proj(sm)
    if capture:
        save_bin(f"{out}/shared_gate_proj_{li}.bin", to_np_f32(sgp[0]), "f32")
        save_bin(f"{out}/shared_up_proj_{li}.bin",   to_np_f32(sup[0]), "f32")
        save_bin(f"{out}/shared_swiglu_{li}.bin",    to_np_f32(sm[0]),  "f32")
        save_bin(f"{out}/shared_down_{li}.bin",      to_np_f32(sd[0]),  "f32")
    sg = mlp.shared_expert_gate(x)
    if capture: save_bin(f"{out}/shared_gate_{li}.bin", to_np_f32(sg[0]), "f32")
    shared_y = mx.sigmoid(sg) * sd
    if capture: save_bin(f"{out}/shared_y_{li}.bin", to_np_f32(shared_y[0]), "f32")
    return y + shared_y


if __name__ == "__main__":
    sys.exit(main())
