"""Loading the published reference checkpoints into the modules of this repo.

Every trained parameter maps onto exactly one of ours; what differs is how they are named and
grouped, and this module is that mapping. It renames ``state_dict`` keys and builds no network.
Three differences are more than a rename, and equivalence needs a matching config too; both are
in docs/decisions.md. Reproduces the reference project, so its published checkpoints load.
"""

from __future__ import annotations

import torch


def _map_attention_block(old: dict, src: str, dst: str, out: dict) -> None:
    """One reference pre-norm attention block (cross or self) onto one of our AttentionBlocks."""
    for key, v in old.items():
        if not key.startswith(src):
            continue
        rest = key[len(src):]
        if rest.startswith("attn.fn.to_qkv."):
            inner = v.shape[0] // 3
            out[dst + "attn.to_q.weight"] = v[:inner]
            out[dst + "attn.to_kv.weight"] = v[inner:]
        elif rest.startswith("attn.fn."):
            out[dst + "attn." + rest[len("attn.fn."):]] = v
        elif rest.startswith("attn.norm_context."):
            out[dst + "norm_context." + rest[len("attn.norm_context."):]] = v
        elif rest.startswith("attn.norm."):
            out[dst + "norm_x." + rest[len("attn.norm."):]] = v
        elif rest.startswith("ff.norm."):
            out[dst + "norm_ff." + rest[len("ff.norm."):]] = v
        elif rest.startswith("ff.fn.net.0.0."):
            out[dst + "ff.net.0." + rest[len("ff.fn.net.0.0."):]] = v
        elif rest.startswith("ff.fn.net.1."):
            out[dst + "ff.net.2." + rest[len("ff.fn.net.1."):]] = v


def _block_indices(old: dict, prefix: str) -> range:
    idx = {int(k[len(prefix):].split(".")[0]) for k in old if k.startswith(prefix)}
    return range(max(idx) + 1) if idx else range(0)


def _convert_autoencoder(old: dict, src: str, dst: str, field_embedding_renames: dict,
                         pos_embed_infix: str = "embed.") -> dict:
    """The autoencoder mapping the datasets share; each passes in its own rename table.

    ``pos_embed_infix``: the microstructures field embedding (lang-ok) wraps its positional embedding (it
    has a periodic option), beam3d uses ``ContinuousSincosEmbed`` directly.

    Parameters with no counterpart are dropped on purpose: the encoder of the reference
    ``conditioning_proj``, which the first stage never calls, and the loss modules'
    normalization buffers.
    """
    out: dict = {}
    renames = dict(field_embedding_renames)
    renames.update({
        "encoder.pos_embed.omega": f"field_embedding.pos_embed.{pos_embed_infix}omega",
        "encoder.mlp.": "field_embedding.mix.",
        "encoder.latents": "encoder.latents",
        "decoder.pos_embed.omega": f"query_embedding.pos_embed.{pos_embed_infix}omega",
        "decoder.query_mlp.1.": "query_embedding.proj.1.",
        "decoder.output_layers.displacement.": "decoder.heads.y.",
        "quant.": "to_latent.",        # a plain linear bottleneck; the name is the old one
        "post_quant.": "from_latent.",
    })
    for key, v in old.items():
        if not key.startswith(src):
            continue
        rest = key[len(src):]
        for old_name, new_name in renames.items():
            if rest.startswith(old_name):
                out[dst + new_name + rest[len(old_name):]] = v
                break

    for i in _block_indices(old, src + "encoder.cross_attn_blocks."):
        _map_attention_block(old, f"{src}encoder.cross_attn_blocks.{i}.",
                             f"{dst}encoder.cross.{i}.", out)
    for i in _block_indices(old, src + "encoder.blocks_attn."):
        _map_attention_block(old, f"{src}encoder.blocks_attn.{i}.",
                             f"{dst}encoder.blocks.{i}.", out)
    for i in _block_indices(old, src + "decoder.self_attn_blocks."):
        _map_attention_block(old, f"{src}decoder.self_attn_blocks.{i}.",
                             f"{dst}decoder.blocks.{i}.", out)
    for i in _block_indices(old, src + "decoder.cross_attn_blocks."):
        _map_attention_block(old, f"{src}decoder.cross_attn_blocks.{i}.",
                             f"{dst}decoder.cross.{i}.", out)
    _map_attention_block(old, src + "decoder.output_block.", dst + "decoder.readout.", out)
    return out


def convert_first_stage(old: dict, src: str = "backbone.", dst: str = "backbone.",
                        num_wallpaper_groups: int = 17) -> dict:
    """A reference microstructures autoencoder onto :class:`PerceiverAutoencoder` keys."""
    out = _convert_autoencoder(old, src, dst, field_embedding_renames={
        "net_merge.": "field_embedding.y_merge.",
        "encoder.type_embed.weight": "field_embedding.node_embed.weight",
    })

    geom = old.get(src + "encoder.geom_embed.weight")
    if geom is not None:
        used = geom[:num_wallpaper_groups]
        out[dst + "field_embedding.wallpaper_embed.weight"] = \
            torch.cat([used.mean(0, keepdim=True), used])
    return out


def convert_first_stage_beam3d(old: dict, src: str = "backbone.", dst: str = "backbone.") -> dict:
    """A reference beam3d autoencoder onto :class:`PerceiverAutoencoder` keys.

    Needs :class:`~bifurcation.models.composites.beam3d.Beam3dFieldEmbedding`, whose node and
    edge projections stay separate as in the reference. The two input columns of the element
    projection are swapped when converting, because the reference feeds ``[L, C]`` and we read
    ``[C, L]``; the measurement that shows it matters is in docs/decisions.md.
    """
    out = _convert_autoencoder(old, src, dst, field_embedding_renames={
        "net_merge.": "field_embedding.y_merge.",
        "encoder.node_attr_proj.": "field_embedding.node_proj.",
        "encoder.edge_token_proj.": "field_embedding.edge_proj.",
    }, pos_embed_infix="")
    w = dst + "field_embedding.edge_proj.weight"
    if w in out:  # [dim, 2]: swap the L and C input columns to match our [C, L] order
        out[w] = out[w][:, [1, 0]].contiguous()
    return out


def convert_reference_first_stage(old: dict, src: str = "backbone.", dst: str = "backbone.") -> dict:
    """Any reference first stage, reading the dataset off the own keys of the checkpoint.

    This is what lets the second stage take a reference autoencoder checkpoint as its first stage
    directly -- see ``LatentFlowMatching(first_stage_reference=True)``.
    """
    if src + "encoder.node_attr_proj.weight" in old:
        return convert_first_stage_beam3d(old, src, dst)
    if src + "encoder.type_embed.weight" in old:
        return convert_first_stage(old, src, dst)
    raise ValueError(
        "not a reference first-stage checkpoint: expected the encoder of the reference "
        f"{src}encoder.node_attr_proj.weight (beam3d) or {src}encoder.type_embed.weight "
        "(microstructures). For a checkpoint trained in this repo, pass "
        "first_stage_reference=false.")


def convert_conditioner(old: dict, src: str = "force_conditioner.", dst: str = "conditioner.") -> dict:
    """The reference ``ForceConditioner`` onto our ``UConditioner``."""
    out: dict = {}
    for key, v in old.items():
        if key.startswith(src):
            out[dst + key[len(src):].replace("force_mlp.net.", "net.").replace(".proj.", ".")] = v
    return out


def convert_approximator(old: dict, src: str = "backbone.", dst: str = "backbone.") -> dict:
    """The reference ``latent_flux.Flux`` onto our ``SiTApproximator``."""
    renames = {
        "x_in.": "x_in.",
        "cond_to_emb.": "cond_in.",
        "mask_to_emb.": "mask_in.",
        "time_in.in_layer.": "time_in.net.0.",
        "time_in.out_layer.": "time_in.net.2.",
        "vector_in.in_layer.": "y_in.net.0.",
        "vector_in.out_layer.": "y_in.net.2.",
        "traj_time_in.in_layer.": "rollout_time_in.net.0.",
        "traj_time_in.out_layer.": "rollout_time_in.net.2.",
        "context_to_emb.": "context_in.",
        "context_gate.": "context_gate.",
        "final_layer.adaLN_modulation.1.": "final.adaLN.1.",
        "final_layer.linear.": "final.linear.",
    }
    for i in _block_indices(old, src + "single_blocks."):
        for name in ("adaLN_modulation.1.", "qkv.", "proj.", "mlp.0.", "mlp.2."):
            new_name = name.replace("adaLN_modulation.1.", "adaLN.1.")
            renames[f"single_blocks.{i}.{name}"] = f"blocks.{i}.{new_name}"

    out: dict = {}
    for key, v in old.items():
        if not key.startswith(src):
            continue
        rest = key[len(src):]
        for old_name, new_name in renames.items():
            if rest.startswith(old_name):
                out[dst + new_name + rest[len(old_name):]] = v
                break
    return out


def convert_second_stage(old: dict, convert_first_stage_fn=convert_first_stage) -> dict:
    """A reference second stage onto :class:`LatentFlowMatching`.

    The reference checkpoints embed their frozen autoencoder (``first_stage_model.backbone.*``),
    so the converted checkpoint is self-contained: load it with ``model.first_stage_ckpt=null``.
    """
    out = convert_approximator(old, src="backbone.", dst="backbone.")
    out |= convert_conditioner(old, src="force_conditioner.", dst="conditioner.")
    out |= convert_first_stage_fn(old, src="first_stage_model.backbone.", dst="ae.")
    return out


def convert_geotdm(old: dict, src: str = "model.", dst: str = "backbone.") -> dict:
    """The published GeoTDM baseline onto :class:`~bifurcation.models.geotdm.GeoTDMReference`.

    Its lightning module keeps the network under ``model``, ours under ``backbone``, and neither
    holds anything else, so the whole mapping is that one prefix. The network itself is the
    published one, so the parameter names inside it already agree.
    """
    return {dst + key[len(src):]: v for key, v in old.items() if key.startswith(src)}


def convert_reference_checkpoint(old: dict) -> dict:
    """Auto-dispatch a published reference checkpoint (first- or second-stage) onto our keys.

    A second-stage checkpoint embeds its frozen autoencoder under ``first_stage_model.*``; a
    first-stage one is the autoencoder alone, and the dataset is detected from the keys of the
    encoder. The GeoTDM baseline is neither. ``eval.py`` calls this, so a reference checkpoint
    evaluates in memory with no offline conversion step.
    """
    if any(k.startswith("model.s_modules.") for k in old):
        return convert_geotdm(old)
    if any(k.startswith("first_stage_model.") for k in old):
        return convert_second_stage(old, convert_first_stage_fn=convert_reference_first_stage)
    return convert_reference_first_stage(old, src="backbone.", dst="backbone.")
