

# TE THD Dense Mask Feasibility Spike

## Scope

This is a feasibility spike for applying the OpenRLHF G1 arbitrary dense
attention mask through the installed Megatron/Transformer Engine packed THD
attention path. It does not change the core attention implementation.

## Environment Reviewed

- Megatron source: `/root/slime_runtime/Megatron-LM`
- Megatron core version reported by import: `0.16.0rc0`
- Megatron commit pinned by `build_conda.sh`: `3714d81d418c9f1bca4594fc35f9e8289f652862`
- Transformer Engine package: `/root/venvs/slime/lib/python3.12/site-packages/transformer_engine`
- Transformer Engine version reported by import: `2.10.0`

Default `python` in this workspace does not import Megatron or Transformer
Engine; the runtime stack is available through `/root/venvs/slime/bin/python`
with Megatron on `PYTHONPATH`.

## API / Source Findings

Megatron's TE wrapper accepts `attention_bias` and `packed_seq_params`, but for
THD it rewrites causal masks to TE's padding variants before calling
`transformer_engine.pytorch.DotProductAttention`:

```text
TEDotProductAttention.forward(..., attention_mask, attn_mask_type,
                              attention_bias=None, packed_seq_params=None)
```

For TE `qkv_format="thd"`, the installed wrapper forwards only the standard
packed sequence fields from `PackedSeqParams`: `cu_seqlens_q`,
`cu_seqlens_kv`, padded cu-seqlens, and max seqlens. There is no Megatron
`PackedSeqParams` field for an arbitrary dense per-token bias/mask.

Transformer Engine 2.10 documents `attn_mask_type="arbitrary"` as requiring an
`attention_mask` broadcastable to `[batch_size, num_heads, max_seqlen_q, max_seqlen_kv]`, but its backend selection disables FlashAttention and
FusedAttention for `arbitrary`. The same backend selection disables
UnfusedDotProductAttention for `qkv_format="thd"`.

The relevant installed TE selection logic therefore leaves no backend for THD
arbitrary masks:

- `qkv_format == "thd"` disables `UnfusedDotProductAttention`.
- `attn_mask_type == "arbitrary"` disables FlashAttention and FusedAttention.
- `core_attention_bias_type="post_scale_bias"` with THD also selects no backend
in this installation.

## Backend Probe

The following probe used `/root/venvs/slime/bin/python` and TE's public backend
selection helper:

```text
thd_padding_no_bias -> (flash=True, fused=False, unfused=False)
thd_arbitrary_no_bias -> (flash=False, fused=False, unfused=False)
thd_padding_post_scale_bias_1hss -> (flash=False, fused=False, unfused=False)
thd_padding_post_scale_bias_b1ss -> (flash=False, fused=False, unfused=False)
```

This matches the source review: packed THD padding/causal without dense bias is
supported by a fast backend, while packed THD arbitrary dense masks or dense
post-scale attention bias are not supported by the installed standard TE path.

## Conclusion

Packed THD arbitrary dense attention mask application is not feasible through
the current Megatron/Transformer Engine fast path without a lower-level TE
change, an alternate non-THD/unpacked path, or a custom fallback.

Keep the current runtime caveat:

```text
applied-via-torch-thd-fallback
```

means `--g1-megatron-ref-apply-dense-attention-mask` was effective through the
slow diagnostic torch THD fallback in `openrlhf_dense_mask_thd_attention`, not
through standard Megatron/TE THD attention.

Because the candidate TE THD path has no backend, this spike does not add a
fallback-vs-candidate golden. The existing CPU golden
`test_openrlhf_dense_mask_thd_attention_applies_arbitrary_mask` remains the
minimal coverage for the diagnostic fallback itself.

## Next Options

- Keep using `applied-via-torch-thd-fallback` only for diagnostic parity runs.
- If production performance becomes required, evaluate either a TE/cuDNN feature
request for THD arbitrary masks or a local custom kernel that accepts packed
dense block masks.
- Revisit if a newer TE release advertises a backend for `qkv_format="thd"` plus
`attn_mask_type="arbitrary"` or THD `post_scale_bias`.

下一步调优！！！！！！！！！！

**我们现在能做到“mask 语义正确”，但做法比较慢，不是 Megatron/Transformer Engine 的高速 attention kernel。**

简单拆开说：

1. **dense mask 是什么**

OpenRLHF G1 里 attention 不是普通 causal mask，而是一个更复杂的 dense attention mask。它精确规定每个 token 能不能看另一个 token。
2. **当前 Slime 怎么做**  
当前 Slime 在 `openrlhf_exact + g1_megatron_ref_apply_dense_attention_mask` 时，会用一个 **torch 手写 fallback attention** 来真正应用这个 dense mask。  
所以它的好处是：**语义精确，适合验证 parity**。  
坏处是：**慢，不是生产 fast path**。
3. **TE fast path 是什么**  
Transformer Engine / Megatron 的 THD attention 是高性能 fused kernel。理想情况是把这个 dense mask 直接交给 TE fast attention，让它又快又精确。
4. **为什么现在不行**  
我们检查了当前环境的 Megatron `0.16.0rc0` + Transformer Engine `2.10.0`：  
它支持 packed THD 的常规 causal/padding mask fast path，但**不支持 arbitrary dense mask** 这种任意形状/任意可见性的 mask。  
所以不能直接把 OpenRLHF G1 dense mask 塞进 TE fast kernel。

结论就是：

- **能验证正确性**：可以用 torch THD fallback 精确应用 dense mask。
- **不能声称是 TE fast path**：当前高速 kernel 没有这个能力。
- **这主要是性能 caveat，不是当前 smoke 闭环失败。** ref/reward parity 可以靠 fallback 证明，但如果以后要高性能生产化，就需要新 TE 版本、定制 kernel，或者换一种 attention 实现路径。

