"""
Pre-LayerNorm Residual Gated Cross-Stream Architecture (PreLN-RGCA).
Implements strictly unconfounded single-mechanism ablations where every ablation
changes ONLY the named architectural mechanism while holding initialization scale
(init_mult = 1.0), alignment matrix (P_cross = A_struct except in ablation_fixed_align
where P_cross = P_fix), and training settings 100% constant.
"""
import numpy as np
from .data import PAD_POS_IDX, compute_verb_to_subj_alignment

def softmax(x, axis=-1):
    ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return ex / np.sum(ex, axis=axis, keepdims=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

def masked_layer_norm_fwd(x, mask, eps=1e-5):
    m_3d = mask[:, :, None]
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)
    x_norm = (x - mean) * std_inv * m_3d
    return x_norm, (x, x_norm, std_inv, mean, m_3d)

def masked_layer_norm_bwd(dout, cache):
    x, x_norm, std_inv, mean, m_3d = cache
    D = x.shape[-1]
    dout = dout * m_3d
    x_mu = (x - mean) * m_3d
    dvar = np.sum(dout * x_mu, axis=-1, keepdims=True) * (-0.5) * (std_inv ** 3)
    dmean = np.sum(dout * (-std_inv), axis=-1, keepdims=True)
    return (dout * std_inv + dvar * (2.0 * x_mu / D) + (dmean / D)) * m_3d

def bmm_2d(X_3d, W_2d):
    B, T, D_in = X_3d.shape
    return np.dot(X_3d.reshape(B * T, D_in), W_2d).reshape(B, T, -1)

def bmm_grad_W(X_3d, dY_3d):
    B, T, D_in = X_3d.shape
    D_out = dY_3d.shape[-1]
    return np.dot(X_3d.reshape(B * T, D_in).T, dY_3d.reshape(B * T, D_out))

def batch_pos_mix(A_3d, X_3d):
    return np.matmul(A_3d, X_3d)

class PreLN_RGCA_Network:
    def __init__(self, W_sem_init, vocab_size=186, num_pos=12, embed_dim=24, num_classes=4, mode="full_preln_rgca", seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.num_pos = num_pos
        self.D = embed_dim
        self.C = num_classes
        self.mode = mode

        scale = np.sqrt(2.0 / (embed_dim * 2))
        self.params = {
            "W_sem": W_sem_init.copy(),
            "W_syn": np.random.randn(num_pos, embed_dim) * 0.12,
            "W_q_sem": np.random.randn(embed_dim, embed_dim) * scale,
            "W_k_sem": np.random.randn(embed_dim, embed_dim) * scale,
            "W_o_sem": np.random.randn(embed_dim, embed_dim) * scale,
            "W_q_syn": np.random.randn(embed_dim, embed_dim) * scale,
            "W_k_syn": np.random.randn(embed_dim, embed_dim) * scale,
            "W_o_syn": np.random.randn(embed_dim, embed_dim) * scale,
            "W_qc_sem": np.random.randn(embed_dim, embed_dim) * scale,
            "W_kc_syn": np.random.randn(embed_dim, embed_dim) * scale,
            "W_qc_syn": np.random.randn(embed_dim, embed_dim) * scale,
            "W_kc_sem": np.random.randn(embed_dim, embed_dim) * scale,
            "W_g_sem": np.random.randn(embed_dim * 2, embed_dim) * scale,
            "b_g_sem": np.zeros(embed_dim) - 1.2,
            "W_g_syn": np.random.randn(embed_dim * 2, embed_dim) * scale,
            "b_g_syn": np.zeros(embed_dim) - 1.2,
            "W_proj": np.random.randn(embed_dim * 3, embed_dim) * scale,
            "b_proj": np.zeros(embed_dim),
            "W_cls": np.random.randn(embed_dim, num_classes) * scale,
            "b_cls": np.zeros(num_classes)
        }
        self.params["W_syn"][PAD_POS_IDX] = 0.0

    def count_parameters(self):
        trainable = sum(v.size for k, v in self.params.items() if k != "W_sem")
        frozen = self.params["W_sem"].size
        return {"trainable": int(trainable), "frozen": int(frozen), "total": int(trainable + frozen)}

    def forward(self, Xw, Xt, Xt_oh, mask, A_struct=None, P_fix=None, P_adj=None, Csem=None, Csyn=None, mask_syn=False, mask_sem=False, disable_ca=False):
        p = self.params
        B, T = Xw.shape
        D = self.D
        m_3d = mask[:, :, None]
        lengths = np.maximum(1.0, np.sum(mask, axis=1, keepdims=True))

        if A_struct is None or P_fix is None or P_adj is None:
            A_struct, P_fix, P_adj = compute_verb_to_subj_alignment(Xt, mask)

        E_sem = p["W_sem"][Xw].copy() * m_3d
        E_syn = p["W_syn"][Xt].copy() * m_3d
        if Csem is not None: E_sem = (E_sem + Csem) * m_3d
        if Csyn is not None: E_syn = (E_syn + Csyn) * m_3d

        if mask_sem: E_sem.fill(0.0)
        if mask_syn: E_syn.fill(0.0)

        cache = {
            "Xw": Xw, "Xt": Xt, "Xt_oh": Xt_oh, "mask": mask, "m_3d": m_3d, "lengths": lengths,
            "P_adj": P_adj, "P_fix": P_fix, "A_s": A_struct,
            "mask_sem": mask_sem, "mask_syn": mask_syn, "disable_ca": disable_ca
        }

        if self.mode == "control_linear_sanity":
            H_uni = (E_sem + E_syn) * m_3d
            pool_uni = np.sum(H_uni, axis=1) / lengths
            joint_vec = np.concatenate([pool_uni, pool_uni, np.zeros_like(pool_uni)], axis=-1)
            pre_act = np.dot(joint_vec, p["W_proj"]) + p["b_proj"]
            logits = np.dot(pre_act, p["W_cls"]) + p["b_cls"]
            cache.update({"joint_vec": joint_vec, "pre_act": pre_act})
            return logits, cache

        if self.mode == "control_single_stream":
            H_in = (E_sem + E_syn) * m_3d
            H_n, ln_c = masked_layer_norm_fwd(H_in, mask)
            ctx_H = batch_pos_mix(P_adj, H_n) * m_3d
            Q = np.tanh(bmm_2d(H_n, p["W_q_sem"])) * m_3d
            K = np.tanh(bmm_2d(ctx_H, p["W_k_sem"])) * m_3d
            S = Q * K * m_3d
            H_sa = (H_in + 0.25 * bmm_2d(S, p["W_o_sem"])) * m_3d
            pool = np.sum(H_sa, axis=1) / lengths
            joint_vec = np.concatenate([pool, pool, np.zeros_like(pool)], axis=-1)
            pre_act = np.dot(joint_vec, p["W_proj"]) + p["b_proj"]
            fused = np.maximum(0, pre_act)
            logits = np.dot(fused, p["W_cls"]) + p["b_cls"]
            cache.update({
                "H_in": H_in, "H_n": H_n, "ln_c": ln_c, "ctx_H": ctx_H,
                "Q": Q, "K": K, "S": S, "joint_vec": joint_vec, "pre_act": pre_act, "fused": fused
            })
            return logits, cache

        use_preln = (self.mode != "ablation_post_ln")
        if use_preln:
            H_sem_n, ln_sem1 = masked_layer_norm_fwd(E_sem, mask)
            H_syn_n, ln_syn1 = masked_layer_norm_fwd(E_syn, mask)
        else:
            H_sem_n, ln_sem1 = E_sem, None
            H_syn_n, ln_syn1 = E_syn, None

        ctx_sem1 = batch_pos_mix(P_adj, H_sem_n) * m_3d
        Q_sem = np.tanh(bmm_2d(H_sem_n, p["W_q_sem"])) * m_3d
        K_sem = np.tanh(bmm_2d(ctx_sem1, p["W_k_sem"])) * m_3d
        S_sem = (Q_sem * K_sem) * m_3d
        H_sem_sa = (H_sem_n + 0.25 * bmm_2d(S_sem, p["W_o_sem"])) * m_3d

        ctx_syn1 = batch_pos_mix(P_adj, H_syn_n) * m_3d
        Q_syn = np.tanh(bmm_2d(H_syn_n, p["W_q_syn"])) * m_3d
        K_syn = np.tanh(bmm_2d(ctx_syn1, p["W_k_syn"])) * m_3d
        S_syn = (Q_syn * K_syn) * m_3d
        H_syn_sa = (H_syn_n + 0.25 * bmm_2d(S_syn, p["W_o_syn"])) * m_3d

        if use_preln:
            H_sem_sa_n, ln_sem2 = masked_layer_norm_fwd(H_sem_sa, mask)
            H_syn_sa_n, ln_syn2 = masked_layer_norm_fwd(H_syn_sa, mask)
        else:
            H_sem_sa_n, ln_sem2 = H_sem_sa, None
            H_syn_sa_n, ln_syn2 = H_syn_sa, None

        P_cross = P_fix if (self.mode == "ablation_fixed_align") else A_struct

        ctx_sem_subj = batch_pos_mix(P_cross, H_sem_sa_n) * m_3d
        ctx_syn_verb = H_syn_sa_n

        if self.mode == "ablation_no_cross" or disable_ca:
            QC_sem = KC_syn = C_sem = np.zeros_like(H_sem_sa_n)
            QC_syn = KC_sem = C_syn = np.zeros_like(H_syn_sa_n)
        else:
            QC_sem = np.tanh(bmm_2d(ctx_sem_subj, p["W_qc_sem"])) * m_3d
            KC_syn = np.tanh(bmm_2d(ctx_syn_verb, p["W_kc_syn"])) * m_3d
            C_sem = KC_syn

            QC_syn = np.tanh(bmm_2d(ctx_syn_verb, p["W_qc_syn"])) * m_3d
            KC_sem = np.tanh(bmm_2d(ctx_sem_subj, p["W_kc_sem"])) * m_3d
            C_syn = KC_sem

        gate_in_sem = np.concatenate([H_sem_sa_n, C_sem], axis=-1) * m_3d
        gate_in_syn = np.concatenate([H_syn_sa_n, C_syn], axis=-1) * m_3d

        if self.mode == "ablation_no_gate":
            G_sem = np.ones_like(H_sem_sa_n) * 0.50 * m_3d
            G_syn = np.ones_like(H_syn_sa_n) * 0.50 * m_3d
        else:
            G_sem = sigmoid(bmm_2d(gate_in_sem, p["W_g_sem"]) + p["b_g_sem"]) * m_3d
            G_syn = sigmoid(bmm_2d(gate_in_syn, p["W_g_syn"]) + p["b_g_syn"]) * m_3d

        H_sem_res2 = (H_sem_sa + G_sem * C_sem) * m_3d
        H_syn_res2 = (H_syn_sa + G_syn * C_syn) * m_3d

        if use_preln:
            H_sem_gated, ln_sem_post = H_sem_res2, None
            H_syn_gated, ln_syn_post = H_syn_res2, None
        else:
            H_sem_gated, ln_sem_post = masked_layer_norm_fwd(H_sem_res2, mask)
            H_syn_gated, ln_syn_post = masked_layer_norm_fwd(H_syn_res2, mask)

        pool_sem = np.sum(H_sem_gated, axis=1) / lengths
        pool_syn = np.sum(H_syn_gated, axis=1) / lengths

        if self.mode in ["ablation_no_cross", "ablation_no_interlock"] or disable_ca:
            interlock = np.zeros_like(pool_sem)
        else:
            interlock = np.sum((G_sem * QC_sem) * (G_syn * QC_syn) * m_3d, axis=1) / lengths

        joint_vec = np.concatenate([pool_sem, pool_syn, interlock], axis=-1)
        pre_act = np.dot(joint_vec, p["W_proj"]) + p["b_proj"]
        fused = np.maximum(0, pre_act)
        logits = np.dot(fused, p["W_cls"]) + p["b_cls"]

        cache.update({
            "P_cross": P_cross, "use_preln": use_preln,
            "H_sem_n": H_sem_n, "H_syn_n": H_syn_n, "ln_sem1": ln_sem1, "ln_syn1": ln_syn1,
            "ctx_sem1": ctx_sem1, "Q_sem": Q_sem, "K_sem": K_sem, "S_sem": S_sem,
            "ctx_syn1": ctx_syn1, "Q_syn": Q_syn, "K_syn": K_syn, "S_syn": S_syn,
            "H_sem_sa": H_sem_sa, "H_syn_sa": H_syn_sa,
            "H_sem_sa_n": H_sem_sa_n, "H_syn_sa_n": H_syn_sa_n, "ln_sem2": ln_sem2, "ln_syn2": ln_syn2,
            "ctx_sem_subj": ctx_sem_subj, "ctx_syn_verb": ctx_syn_verb,
            "QC_sem": QC_sem, "KC_syn": KC_syn, "C_sem": C_sem,
            "QC_syn": QC_syn, "KC_sem": KC_sem, "C_syn": C_syn,
            "gate_in_sem": gate_in_sem, "gate_in_syn": gate_in_syn,
            "G_sem": G_sem, "G_syn": G_syn,
            "H_sem_gated": H_sem_gated, "H_syn_gated": H_syn_gated,
            "ln_sem_post": ln_sem_post, "ln_syn_post": ln_syn_post,
            "joint_vec": joint_vec, "pre_act": pre_act, "fused": fused
        })
        return logits, cache

    def backward(self, dlogits, cache):
        p = self.params
        B, T = cache["Xw"].shape
        D = self.D
        Xt_oh = cache["Xt_oh"]
        m_3d = cache["m_3d"]
        lengths = cache["lengths"][:, :, None]
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        if self.mode == "control_linear_sanity":
            grads["W_cls"] = np.dot(cache["pre_act"].T, dlogits)
            grads["b_cls"] = np.sum(dlogits, axis=0)
            dpre = np.dot(dlogits, p["W_cls"].T)
            grads["W_proj"] = np.dot(cache["joint_vec"].T, dpre)
            grads["b_proj"] = np.sum(dpre, axis=0)
            djoint = np.dot(dpre, p["W_proj"].T)
            dE_syn = (np.repeat((djoint[:, :D] + djoint[:, D:2*D])[:, None, :], T, axis=1) / lengths) * m_3d
            grads["W_syn"] = bmm_grad_W(Xt_oh, dE_syn)
            grads["W_syn"][PAD_POS_IDX] = 0.0
            return grads

        if self.mode == "control_single_stream":
            fused = cache["fused"]
            grads["W_cls"] = np.dot(fused.T, dlogits)
            grads["b_cls"] = np.sum(dlogits, axis=0)
            dfused = np.dot(dlogits, p["W_cls"].T)
            dpre = dfused * (cache["pre_act"] > 0)
            grads["W_proj"] = np.dot(cache["joint_vec"].T, dpre)
            grads["b_proj"] = np.sum(dpre, axis=0)
            djoint = np.dot(dpre, p["W_proj"].T)
            dpool = djoint[:, :D] + djoint[:, D:2*D]
            dH_sa = (np.repeat(dpool[:, None, :], T, axis=1) / lengths) * m_3d

            grads["W_o_sem"] = 0.25 * bmm_grad_W(cache["S"], dH_sa)
            dS = 0.25 * bmm_2d(dH_sa, p["W_o_sem"].T) * m_3d
            dQ_raw = dS * cache["K"] * (1.0 - cache["Q"] ** 2) * m_3d
            dK_raw = dS * cache["Q"] * (1.0 - cache["K"] ** 2) * m_3d
            grads["W_q_sem"] = bmm_grad_W(cache["H_n"], dQ_raw)
            grads["W_k_sem"] = bmm_grad_W(cache["ctx_H"], dK_raw)
            dH_n = bmm_2d(dQ_raw, p["W_q_sem"].T) + batch_pos_mix(cache["P_adj"].transpose(0, 2, 1), bmm_2d(dK_raw, p["W_k_sem"].T))
            dH_in = dH_sa + masked_layer_norm_bwd(dH_n, cache["ln_c"])
            grads["W_syn"] = bmm_grad_W(Xt_oh, dH_in * m_3d)
            grads["W_syn"][PAD_POS_IDX] = 0.0
            return grads

        fused = cache["fused"]
        grads["W_cls"] = np.dot(fused.T, dlogits)
        grads["b_cls"] = np.sum(dlogits, axis=0)
        dfused = np.dot(dlogits, p["W_cls"].T)
        dpre_act = dfused * (cache["pre_act"] > 0)
        grads["W_proj"] = np.dot(cache["joint_vec"].T, dpre_act)
        grads["b_proj"] = np.sum(dpre_act, axis=0)

        djoint = np.dot(dpre_act, p["W_proj"].T)
        dpool_sem = djoint[:, :D]
        dpool_syn = djoint[:, D:2*D]
        dinterlock = djoint[:, 2*D:]

        dH_sem_gated = (np.repeat(dpool_sem[:, None, :], T, axis=1) / lengths) * m_3d
        dH_syn_gated = (np.repeat(dpool_syn[:, None, :], T, axis=1) / lengths) * m_3d

        if cache["use_preln"]:
            dH_sem_res2 = dH_sem_gated
            dH_syn_res2 = dH_syn_gated
        else:
            dH_sem_res2 = masked_layer_norm_bwd(dH_sem_gated, cache["ln_sem_post"])
            dH_syn_res2 = masked_layer_norm_bwd(dH_syn_gated, cache["ln_syn_post"])

        G_sem, G_syn = cache["G_sem"], cache["G_syn"]
        C_sem, C_syn = cache["C_sem"], cache["C_syn"]
        QC_sem, QC_syn = cache["QC_sem"], cache["QC_syn"]

        dG_sem = dH_sem_res2 * C_sem * m_3d
        dC_sem = dH_sem_res2 * G_sem * m_3d
        dG_syn = dH_syn_res2 * C_syn * m_3d
        dC_syn = dH_syn_res2 * G_syn * m_3d

        dQC_sem = np.zeros_like(QC_sem)
        dQC_syn = np.zeros_like(QC_syn)
        if not (self.mode in ["ablation_no_cross", "ablation_no_interlock"] or cache["disable_ca"]):
            dinter_rep = (np.repeat(dinterlock[:, None, :], T, axis=1) / lengths) * m_3d
            dG_sem += dinter_rep * QC_sem * (G_syn * QC_syn) * m_3d
            dQC_sem += dinter_rep * G_sem * (G_syn * QC_syn) * m_3d
            dG_syn += dinter_rep * QC_syn * (G_sem * QC_sem) * m_3d
            dQC_syn += dinter_rep * G_syn * (G_sem * QC_sem) * m_3d

        dH_sem_sa = dH_sem_res2.copy()
        dH_syn_sa = dH_syn_res2.copy()
        dH_sem_sa_n = np.zeros_like(dH_sem_sa)
        dH_syn_sa_n = np.zeros_like(dH_syn_sa)

        if self.mode != "ablation_no_gate":
            dz_g_sem = dG_sem * G_sem * (1.0 - G_sem) * m_3d
            dz_g_syn = dG_syn * G_syn * (1.0 - G_syn) * m_3d
            grads["W_g_sem"] = bmm_grad_W(cache["gate_in_sem"], dz_g_sem)
            grads["b_g_sem"] = np.sum(dz_g_sem, axis=(0, 1))
            grads["W_g_syn"] = bmm_grad_W(cache["gate_in_syn"], dz_g_syn)
            grads["b_g_syn"] = np.sum(dz_g_syn, axis=(0, 1))

            dgate_sem = bmm_2d(dz_g_sem, p["W_g_sem"].T) * m_3d
            dgate_syn = bmm_2d(dz_g_syn, p["W_g_syn"].T) * m_3d
            dH_sem_sa_n += dgate_sem[:, :, :D]
            dC_sem += dgate_sem[:, :, D:]
            dH_syn_sa_n += dgate_syn[:, :, :D]
            dC_syn += dgate_syn[:, :, D:]

        if not (self.mode == "ablation_no_cross" or cache["disable_ca"]):
            P_cross = cache["P_cross"]
            P_cross_T = P_cross.transpose(0, 2, 1)

            dQC_sem_raw = dQC_sem * (1.0 - QC_sem ** 2) * m_3d
            dKC_syn_raw = dC_sem * (1.0 - cache["KC_syn"] ** 2) * m_3d
            grads["W_qc_sem"] = bmm_grad_W(cache["ctx_sem_subj"], dQC_sem_raw)
            grads["W_kc_syn"] = bmm_grad_W(cache["ctx_syn_verb"], dKC_syn_raw)
            dH_sem_sa_n += batch_pos_mix(P_cross_T, bmm_2d(dQC_sem_raw, p["W_qc_sem"].T)) * m_3d
            dH_syn_sa_n += bmm_2d(dKC_syn_raw, p["W_kc_syn"].T) * m_3d

            dQC_syn_raw = dQC_syn * (1.0 - QC_syn ** 2) * m_3d
            dKC_sem_raw = dC_syn * (1.0 - cache["KC_sem"] ** 2) * m_3d
            grads["W_qc_syn"] = bmm_grad_W(cache["ctx_syn_verb"], dQC_syn_raw)
            grads["W_kc_sem"] = bmm_grad_W(cache["ctx_sem_subj"], dKC_sem_raw)
            dH_syn_sa_n += bmm_2d(dQC_syn_raw, p["W_qc_syn"].T) * m_3d
            dH_sem_sa_n += batch_pos_mix(P_cross_T, bmm_2d(dKC_sem_raw, p["W_kc_sem"].T)) * m_3d

        if cache["use_preln"]:
            dH_sem_sa += masked_layer_norm_bwd(dH_sem_sa_n, cache["ln_sem2"])
            dH_syn_sa += masked_layer_norm_bwd(dH_syn_sa_n, cache["ln_syn2"])
        else:
            dH_sem_sa += dH_sem_sa_n
            dH_syn_sa += dH_syn_sa_n

        P_adj_T = cache["P_adj"].transpose(0, 2, 1)
        grads["W_o_sem"] = 0.25 * bmm_grad_W(cache["S_sem"], dH_sem_sa)
        dS_sem = 0.25 * bmm_2d(dH_sem_sa, p["W_o_sem"].T) * m_3d
        dQ_sem_raw = dS_sem * cache["K_sem"] * (1.0 - cache["Q_sem"] ** 2) * m_3d
        dK_sem_raw = dS_sem * cache["Q_sem"] * (1.0 - cache["K_sem"] ** 2) * m_3d
        grads["W_q_sem"] = bmm_grad_W(cache["H_sem_n"], dQ_sem_raw)
        grads["W_k_sem"] = bmm_grad_W(cache["ctx_sem1"], dK_sem_raw)

        grads["W_o_syn"] = 0.25 * bmm_grad_W(cache["S_syn"], dH_syn_sa)
        dS_syn = 0.25 * bmm_2d(dH_syn_sa, p["W_o_syn"].T) * m_3d
        dQ_syn_raw = dS_syn * cache["K_syn"] * (1.0 - cache["Q_syn"] ** 2) * m_3d
        dK_syn_raw = dS_syn * cache["Q_syn"] * (1.0 - cache["K_syn"] ** 2) * m_3d
        grads["W_q_syn"] = bmm_grad_W(cache["H_syn_n"], dQ_syn_raw)
        grads["W_k_syn"] = bmm_grad_W(cache["ctx_syn1"], dK_syn_raw)
        dH_syn_n = dH_syn_sa + bmm_2d(dQ_syn_raw, p["W_q_syn"].T) + batch_pos_mix(P_adj_T, bmm_2d(dK_syn_raw, p["W_k_syn"].T))
        dE_syn = (masked_layer_norm_bwd(dH_syn_n, cache["ln_syn1"]) if cache["ln_syn1"] else dH_syn_n) * m_3d

        if not cache["mask_syn"]:
            grads["W_syn"] = bmm_grad_W(Xt_oh, dE_syn)
            grads["W_syn"][PAD_POS_IDX] = 0.0
        return grads
