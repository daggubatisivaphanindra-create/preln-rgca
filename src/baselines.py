"""
Transparent Symbolic Rule-Based Baseline (Step 5 of Audit Implementation Plan).
Evaluates deterministic auxiliary-verb morphosyntactic compatibility and subject-noun
animacy compatibility under:
  1. Oracle Lexicon (full animacy inventory known a priori)
  2. Training-Split Induced Lexicon (animacy induced strictly from grammatical training items)
"""
import numpy as np
from .data import pos2idx, MAX_LEN

class RuleBasedSymbolicBaseline:
    def __init__(self, mode="train_induced"):
        self.mode = mode
        self.anim_set = set()
        self.inanim_set = set()

    def fit(self, enc, tr_idx, task_type="on99", oracle_anim_ids=None, oracle_inanim_ids=None):
        if self.mode == "oracle" and oracle_anim_ids is not None:
            self.anim_set = set(oracle_anim_ids)
            self.inanim_set = set(oracle_inanim_ids)
            return self

        self.anim_set = set()
        self.inanim_set = set()
        Xw_tr = enc["Xw"][tr_idx]
        Xt_tr = enc["Xt"][tr_idx]
        mask_tr = enc["mask"][tr_idx]
        y_tr = enc["y"][tr_idx]

        noun_id = pos2idx["NOUN"]
        sconj_id = pos2idx["SCONJ"]
        ving_id = pos2idx["VERB_ING"]
        ved_id = pos2idx["VERB_ED"]

        for b in range(len(tr_idx)):
            lbl = y_tr[b]
            is_gram = (lbl in [0, 1]) if task_type in ["on99", "external", "authentic"] else ((lbl == 0) if task_type == "ko05" else (lbl == 1))
            if not is_gram:
                continue
            first_noun = -1
            for t in range(MAX_LEN):
                if mask_tr[b, t] == 0.0:
                    continue
                tg = Xt_tr[b, t]
                if tg == sconj_id:
                    first_noun = -1
                elif tg == noun_id and first_noun == -1:
                    first_noun = Xw_tr[b, t]
                elif tg == ving_id and first_noun != -1:
                    self.anim_set.add(first_noun)
                elif tg == ved_id and first_noun != -1:
                    self.inanim_set.add(first_noun)
        return self

    def _check_sentence(self, Xw_row, Xt_row, mask_row, rng):
        noun_id = pos2idx["NOUN"]
        sconj_id = pos2idx["SCONJ"]
        mod_id = pos2idx["AUX_MODAL"]
        ving_id = pos2idx["VERB_ING"]
        ved_id = pos2idx["VERB_ED"]

        syn_err = False
        sem_err = False
        unknown_noun = False
        last_noun_w = -1

        for t in range(MAX_LEN):
            if mask_row[t] == 0.0:
                continue
            tg = Xt_row[t]
            if tg == sconj_id:
                last_noun_w = -1
            elif tg == noun_id:
                last_noun_w = Xw_row[t]
            elif tg in [ving_id, ved_id]:
                if t > 0 and Xt_row[t - 1] == mod_id:
                    syn_err = True
                if last_noun_w != -1:
                    if last_noun_w in self.anim_set:
                        if tg == ved_id:
                            sem_err = True
                    elif last_noun_w in self.inanim_set:
                        if tg == ving_id:
                            sem_err = True
                    else:
                        unknown_noun = True

        if unknown_noun and not sem_err:
            sem_err = bool(rng.randint(0, 2))
        return syn_err, sem_err

    def predict(self, enc, te_idx, num_classes=4, task_type="on99", seed=42):
        rng = np.random.RandomState(seed + 777)
        preds = np.zeros(len(te_idx), dtype=np.int32)
        for idx_local, b in enumerate(te_idx):
            syn_err, sem_err = self._check_sentence(enc["Xw"][b], enc["Xt"][b], enc["mask"][b], rng)
            if task_type in ["on99", "external", "authentic"] and num_classes == 4:
                if not syn_err and not sem_err: preds[idx_local] = 0
                elif syn_err and not sem_err:   preds[idx_local] = 1
                elif not syn_err and sem_err:   preds[idx_local] = 2
                else:                           preds[idx_local] = 3
            elif task_type == "ko05" and num_classes == 3:
                if not syn_err and not sem_err: preds[idx_local] = 0
                elif not syn_err and sem_err:   preds[idx_local] = 1
                else:                           preds[idx_local] = 2
            elif task_type == "cola" and num_classes == 2:
                preds[idx_local] = 1 if (not syn_err and not sem_err) else 0
        return preds

    def predict_blimp_pairs(self, enc_blimp, te_p, seed=42):
        rng = np.random.RandomState(seed + 888)
        eg, eb = enc_blimp["good"], enc_blimp["bad"]
        correct = np.zeros(len(te_p), dtype=np.float64)
        for i, p_idx in enumerate(te_p):
            syn_g, sem_g = self._check_sentence(eg["Xw"][p_idx], eg["Xt"][p_idx], eg["mask"][p_idx], rng)
            syn_b, sem_b = self._check_sentence(eb["Xw"][p_idx], eb["Xt"][p_idx], eb["mask"][p_idx], rng)
            score_g = 1.0 if (not syn_g and not sem_g) else 0.0
            score_b = 1.0 if (not syn_b and not sem_b) else 0.0
            if score_g > score_b:
                correct[i] = 1.0
            elif score_g == score_b:
                correct[i] = float(rng.randint(0, 2))
        return correct

class AlignmentMatchedSymbolicBaseline:
    """
    Alignment-Matched Symbolic Baseline:
    Directly uses the exact same clause-local head-subject syntactic alignment matrix
    A_struct as PreLN-RGCA to evaluate whether the proposed neural interaction provides
    benefits beyond the hand-engineered structural alignment alone.
    At each predicate verb t, it retrieves the aligned subject noun tau = argmax_k A_struct[b, t, k],
    checking auxiliary compatibility and subject animacy against verbal active/passive voice.
    """
    def __init__(self, mode="train_induced"):
        self.mode = mode
        self.anim_set = set()
        self.inanim_set = set()

    def fit(self, enc, tr_idx, task_type="on99", oracle_anim_ids=None, oracle_inanim_ids=None):
        if self.mode == "oracle" and oracle_anim_ids is not None:
            self.anim_set = set(oracle_anim_ids)
            self.inanim_set = set(oracle_inanim_ids)
            return self

        self.anim_set = set()
        self.inanim_set = set()
        Xw_tr = enc["Xw"][tr_idx]
        Xt_tr = enc["Xt"][tr_idx]
        mask_tr = enc["mask"][tr_idx]
        A_tr = enc["A_struct"][tr_idx]
        y_tr = enc["y"][tr_idx]

        noun_id = pos2idx["NOUN"]
        ving_id = pos2idx["VERB_ING"]
        ved_id = pos2idx["VERB_ED"]

        for b in range(len(tr_idx)):
            lbl = y_tr[b]
            is_gram = (lbl in [0, 1]) if task_type in ["on99", "external", "authentic"] else ((lbl == 0) if task_type == "ko05" else (lbl == 1))
            if not is_gram:
                continue
            for t in range(MAX_LEN):
                if mask_tr[b, t] == 0.0:
                    continue
                tg = Xt_tr[b, t]
                if tg in [ving_id, ved_id]:
                    subj_pos = int(np.argmax(A_tr[b, t]))
                    if Xt_tr[b, subj_pos] == noun_id:
                        w_subj = Xw_tr[b, subj_pos]
                        if tg == ving_id:
                            self.anim_set.add(w_subj)
                        elif tg == ved_id:
                            self.inanim_set.add(w_subj)
        return self

    def _check_sentence(self, Xw_row, Xt_row, mask_row, A_row, rng):
        noun_id = pos2idx["NOUN"]
        mod_id = pos2idx["AUX_MODAL"]
        ving_id = pos2idx["VERB_ING"]
        ved_id = pos2idx["VERB_ED"]

        syn_err = False
        sem_err = False
        unknown_noun = False

        for t in range(MAX_LEN):
            if mask_row[t] == 0.0:
                continue
            tg = Xt_row[t]
            if tg in [ving_id, ved_id]:
                if t > 0 and Xt_row[t - 1] == mod_id:
                    syn_err = True
                subj_pos = int(np.argmax(A_row[t]))
                if Xt_row[subj_pos] == noun_id:
                    w_subj = Xw_row[subj_pos]
                    if w_subj in self.anim_set:
                        if tg == ved_id:
                            sem_err = True
                    elif w_subj in self.inanim_set:
                        if tg == ving_id:
                            sem_err = True
                    else:
                        unknown_noun = True

        if unknown_noun and not sem_err:
            sem_err = bool(rng.randint(0, 2))
        return syn_err, sem_err

    def predict(self, enc, te_idx, num_classes=4, task_type="on99", seed=42):
        rng = np.random.RandomState(seed + 999)
        preds = np.zeros(len(te_idx), dtype=np.int32)
        Xw_te = enc["Xw"][te_idx]
        Xt_te = enc["Xt"][te_idx]
        mask_te = enc["mask"][te_idx]
        A_te = enc["A_struct"][te_idx]

        for idx_local in range(len(te_idx)):
            syn_err, sem_err = self._check_sentence(Xw_te[idx_local], Xt_te[idx_local], mask_te[idx_local], A_te[idx_local], rng)
            if task_type in ["on99", "external", "authentic"] and num_classes == 4:
                if not syn_err and not sem_err: preds[idx_local] = 0
                elif syn_err and not sem_err:   preds[idx_local] = 1
                elif not syn_err and sem_err:   preds[idx_local] = 2
                else:                           preds[idx_local] = 3
            elif task_type == "ko05" and num_classes == 3:
                if not syn_err and not sem_err: preds[idx_local] = 0
                elif not syn_err and sem_err:   preds[idx_local] = 1
                else:                           preds[idx_local] = 2
            elif task_type == "cola" and num_classes == 2:
                preds[idx_local] = 1 if (not syn_err and not sem_err) else 0

        return preds

    def predict_blimp_pairs(self, enc_blimp, te_p, seed=42):
        rng = np.random.RandomState(seed + 888)
        eg, eb = enc_blimp["good"], enc_blimp["bad"]
        correct = np.zeros(len(te_p), dtype=np.float64)
        for i, p_idx in enumerate(te_p):
            syn_g, sem_g = self._check_sentence(eg["Xw"][p_idx], eg["Xt"][p_idx], eg["mask"][p_idx], eg["A_struct"][p_idx], rng)
            syn_b, sem_b = self._check_sentence(eb["Xw"][p_idx], eb["Xt"][p_idx], eb["mask"][p_idx], eb["A_struct"][p_idx], rng)
            score_g = 1.0 if (not syn_g and not sem_g) else 0.0
            score_b = 1.0 if (not syn_b and not sem_b) else 0.0
            if score_g > score_b:
                correct[i] = 1.0
            elif score_g == score_b:
                correct[i] = float(rng.randint(0, 2))
        return correct
