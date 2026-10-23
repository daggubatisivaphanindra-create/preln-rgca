#!/usr/bin/env python3
"""
Executable Unit and Integration Test Suite for PreLN-RGCA (v1.0.0).
"""
import os, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    PAD_POS_IDX, num_pos, EMBED_DIM, deterministic_lemmatize_and_tag,
    build_all_datasets, build_vocabulary, encode_classification_items,
    strict_frame_split, fit_training_only_svd_embeddings, compute_verb_to_subj_alignment
)
from src.model import PreLN_RGCA_Network, softmax

def run_all_unit_tests():
    on99_items, ko05_items, blimp_pairs, cola_items, ext_items, on99_oov_items, _ = build_all_datasets()
    vocab, word2idx = build_vocabulary(on99_items, ko05_items, blimp_pairs, cola_items, ext_items, on99_oov_items)
    vocab_size = len(vocab)
    assert vocab_size == 186, f"Expected vocab_size=186, got {vocab_size}"

    # Surface uniqueness & deterministic tag agreement
    for ds_name, items in [("ON99", on99_items), ("KO05", ko05_items), ("CoLA", cola_items), ("External", ext_items)]:
        seen = {}
        for it in items:
            s_str = " ".join(it["surface_words"])
            if s_str in seen:
                assert seen[s_str] == it["label"], f"Conflicting label on {ds_name}: {s_str}"
            seen[s_str] = it["label"]
            lem_check, tag_check = deterministic_lemmatize_and_tag(it["surface_words"])
            assert lem_check == it["words"], f"Lemma mismatch on {ds_name}"
            assert tag_check == it["tags"], f"Tag mismatch on {ds_name}"

    for p in blimp_pairs:
        assert p["good_surface_words"] != p["bad_surface_words"], f"Identical pair in BLiMP: {p['pair_id']}"

    enc_on99 = encode_classification_items(on99_items, word2idx, 101)
    seeds = [42, 101, 2024, 7, 13, 99, 123, 456, 789, 2026]
    for s in seeds:
        tr, va, te, _ = strict_frame_split(enc_on99["frame_ids"], 10, 2, s)
        f_tr = set(enc_on99["frame_ids"][tr])
        f_va = set(enc_on99["frame_ids"][va])
        f_te = set(enc_on99["frame_ids"][te])
        assert len(f_tr & f_va) == 0 and len(f_tr & f_te) == 0 and len(f_va & f_te) == 0

    tr0, _, _, _ = strict_frame_split(enc_on99["frame_ids"], 10, 2, 42)
    W_sem0 = fit_training_only_svd_embeddings(enc_on99, tr0, vocab_size=vocab_size, embed_dim=EMBED_DIM, seed=42, task_type="on99")
    net4 = PreLN_RGCA_Network(W_sem0, vocab_size=186, num_pos=12, embed_dim=24, num_classes=4, mode="full_preln_rgca", seed=42)
    assert net4.count_parameters() == {"trainable": 10252, "frozen": 4464, "total": 14716}

    # PAD invariance
    B_t = 4
    Xw_12, Xt_12, Xtoh_12, mask_12 = enc_on99["Xw"][:B_t], enc_on99["Xt"][:B_t], enc_on99["Xt_oh"][:B_t], enc_on99["mask"][:B_t]
    A_12, Pf_12, Pa_12 = compute_verb_to_subj_alignment(Xt_12, mask_12)
    l_12, _ = net4.forward(Xw_12, Xt_12, Xtoh_12, mask_12, A_struct=A_12, P_fix=Pf_12, P_adj=Pa_12)

    Xw_16 = np.pad(Xw_12, ((0, 0), (0, 4)), constant_values=0)
    Xt_16 = np.pad(Xt_12, ((0, 0), (0, 4)), constant_values=PAD_POS_IDX)
    Xtoh_16 = np.zeros((B_t, 16, num_pos))
    Xtoh_16[:, :12, :] = Xtoh_12
    Xtoh_16[:, 12:, PAD_POS_IDX] = 1.0
    mask_16 = np.pad(mask_12, ((0, 0), (0, 4)), constant_values=0.0)
    A_16, Pf_16, Pa_16 = compute_verb_to_subj_alignment(Xt_16, mask_16)
    l_16, _ = net4.forward(Xw_16, Xt_16, Xtoh_16, mask_16, A_struct=A_16, P_fix=Pf_16, P_adj=Pa_16)
    pad_diff = float(np.max(np.abs(l_12 - l_16)))
    assert pad_diff < 1e-12, f"PAD invariance failed: {pad_diff}"

    # Gradient check Pre-LN and Post-LN
    y_t = enc_on99["y"][:B_t]
    for check_mode in ["full_preln_rgca", "ablation_post_ln"]:
        net_g = PreLN_RGCA_Network(W_sem0, vocab_size=186, num_pos=12, embed_dim=24, num_classes=4, mode=check_mode, seed=42)
        l_g, c_g = net_g.forward(Xw_12, Xt_12, Xtoh_12, mask_12, A_struct=A_12, P_fix=Pf_12, P_adj=Pa_12)
        p_g = softmax(l_g, axis=-1)
        dl = p_g.copy(); dl[np.arange(B_t), y_t] -= 1.0; dl /= B_t
        grads = net_g.backward(dl, c_g)
        eps = 1e-5
        orig = net_g.params["W_qc_sem"][0, 0]
        net_g.params["W_qc_sem"][0, 0] = orig + eps
        lp, _ = net_g.forward(Xw_12, Xt_12, Xtoh_12, mask_12, A_struct=A_12, P_fix=Pf_12, P_adj=Pa_12)
        lossp = -np.mean(np.log(softmax(lp)[np.arange(B_t), y_t]))
        net_g.params["W_qc_sem"][0, 0] = orig - eps
        lm, _ = net_g.forward(Xw_12, Xt_12, Xtoh_12, mask_12, A_struct=A_12, P_fix=Pf_12, P_adj=Pa_12)
        lossm = -np.mean(np.log(softmax(lm)[np.arange(B_t), y_t]))
        net_g.params["W_qc_sem"][0, 0] = orig
        fd = (lossp - lossm) / (2 * eps)
        an = grads["W_qc_sem"][0, 0]
        rel_err = abs(fd - an) / max(1e-8, abs(fd) + abs(an))
        assert rel_err < 5e-5, f"Gradient check failed on {check_mode}: {rel_err}"

    print("[PASS] All unit and integration tests passed.")

if __name__ == "__main__":
    run_all_unit_tests()
