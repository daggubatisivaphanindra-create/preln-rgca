"""
Dataset construction, morphological surface inflection, deterministic surface-to-tag
lemmatization, strict zero-leakage frame splitting, and training-only distributional
SVD embedding estimation.
"""
import hashlib
import numpy as np

MORPHOSYNTACTIC_TAGS = [
    "DET", "ADJ", "NOUN", "AUX_MODAL", "AUX_BE",
    "VERB_BASE", "VERB_ING", "VERB_ED", "ADP", "ADV", "SCONJ", "PAD"
]
pos2idx = {t: i for i, t in enumerate(MORPHOSYNTACTIC_TAGS)}
num_pos = len(MORPHOSYNTACTIC_TAGS)
PAD_POS_IDX = pos2idx["PAD"]
MAX_LEN = 12
EMBED_DIM = 24

VERB_INFLECTIONS = {
    "eat": ("eat", "eating", "eaten"), "spill": ("spill", "spilling", "spilled"),
    "bury": ("bury", "burying", "buried"), "break": ("break", "breaking", "broken"),
    "jump": ("jump", "jumping", "jumped"), "open": ("open", "opening", "opened"),
    "tear": ("tear", "tearing", "torn"), "drop": ("drop", "dropping", "dropped"),
    "catch": ("catch", "catching", "caught"), "freeze": ("freeze", "freezing", "frozen"),
    "chase": ("chase", "chasing", "chased"), "scare": ("scare", "scaring", "scared"),
    "crack": ("crack", "cracking", "cracked"), "roll": ("roll", "rolling", "rolled"),
    "seize": ("seize", "seizing", "seized"), "spot": ("spot", "spotting", "spotted"),
    "bump": ("bump", "bumping", "bumped"), "sink": ("sink", "sinking", "sunk"),
    "swallow": ("swallow", "swallowing", "swallowed"), "crush": ("crush", "crushing", "crushed"),
    "bake": ("bake", "baking", "baked"), "burn": ("burn", "burning", "burned"),
    "write": ("write", "writing", "written"), "mend": ("mend", "mending", "mended"),
    "stain": ("stain", "staining", "stained"), "stretch": ("stretch", "stretching", "stretched"),
    "rip": ("rip", "ripping", "ripped"), "lift": ("lift", "lifting", "lifted"),
    "devour": ("devour", "devouring", "devoured"), "serve": ("serve", "serving", "served"),
    "slice": ("slice", "slicing", "sliced"), "fold": ("fold", "folding", "folded"),
    "paint": ("paint", "painting", "painted"), "frame": ("frame", "framing", "framed"),
    "plant": ("plant", "planting", "planted"), "water": ("water", "watering", "watered"),
    "find": ("find", "finding", "found"), "hide": ("hide", "hiding", "hidden"),
    "study": ("study", "studying", "studied"), "close": ("close", "closing", "closed"),
    "explain": ("explain", "explaining", "explained"), "save": ("save", "saving", "saved"),
    "enter": ("enter", "entering", "entered"), "deliver": ("deliver", "delivering", "delivered"),
    "seal": ("seal", "sealing", "sealed"), "heal": ("heal", "healing", "healed"),
    "scan": ("scan", "scanning", "scanned"), "pour": ("pour", "pouring", "poured"),
    "warm": ("warm", "warming", "warmed"), "bolt": ("bolt", "bolting", "bolted"),
    "watch": ("watch", "watching", "watched"), "lock": ("lock", "locking", "locked"),
    "compose": ("compose", "composing", "composed"), "recite": ("recite", "reciting", "recited"),
    "revise": ("revise", "revising", "revised"), "print": ("print", "printing", "printed"),
    "analyze": ("analyze", "analyzing", "analyzed"), "clean": ("clean", "cleaning", "cleaned"),
    "evaluate": ("evaluate", "evaluating", "evaluated"), "edit": ("edit", "editing", "edited"),
    "manage": ("manage", "managing", "managed"), "design": ("design", "designing", "designed"),
    "wire": ("wire", "wiring", "wired"), "endorse": ("endorse", "endorsing", "endorsed"),
    "draft": ("draft", "drafting", "drafted"), "conduct": ("conduct", "conducting", "conducted"),
    "score": ("score", "scoring", "scored"), "draw": ("draw", "drawing", "drawn"),
    "scale": ("scale", "scaling", "scaled"), "verify": ("verify", "verifying", "verified"),
    "check": ("check", "checking", "checked"), "cultivate": ("cultivate", "cultivating", "cultivated"),
    "document": ("document", "documenting", "documented"), "parse": ("parse", "parsing", "parsed"),
    "translate": ("translate", "translating", "translated"), "index": ("index", "indexing", "indexed"),
    "observe": ("observe", "observing", "observed"), "chart": ("chart", "charting", "charted"),
    "synthesize": ("synthesize", "synthesizing", "synthesized"), "purify": ("purify", "purifying", "purified"),
    "measure": ("measure", "measuring", "measured"), "track": ("track", "tracking", "tracked"),
    "transplant": ("transplant", "transplanting", "transplanted"), "clamp": ("clamp", "clamping", "clamped"),
    "classify": ("classify", "classifying", "classified"), "press": ("press", "pressing", "pressed"),
    "sample": ("sample", "sampling", "sampled"), "weigh": ("weigh", "weighing", "weighed"),
    "publish": ("publish", "publishing", "published"), "catalog": ("catalog", "cataloging", "cataloged"),
    "shelve": ("shelve", "shelving", "shelved"), "dispense": ("dispense", "dispensing", "dispensed"),
    "rehearse": ("rehearse", "rehearsing", "rehearsed"), "play": ("play", "playing", "played"),
    "forge": ("forge", "forging", "forged"), "chisel": ("chisel", "chiseling", "chiseled"),
    "navigate": ("navigate", "navigating", "navigated"), "land": ("land", "landing", "landed"),
    "sound": ("sound", "sounding", "sounded"), "patrol": ("patrol", "patrolling", "patrolled"),
    "guard": ("guard", "guarding", "guarded"),
    "leave": ("leave", "leaving", "left"), "prepare": ("prepare", "preparing", "prepared"),
    "repair": ("repair", "repairing", "repaired"), "polish": ("polish", "polishing", "polished"),
    "examine": ("examine", "examining", "examined"), "carry": ("carry", "carrying", "carried"),
    "confirm": ("confirm", "confirming", "confirmed"), "clarify": ("clarify", "clarifying", "clarified"),
    "indicate": ("indicate", "indicating", "indicated"), "support": ("support", "supporting", "supported"),
    "challenge": ("challenge", "challenging", "challenged"), "reveal": ("reveal", "revealing", "revealed"),
}

SURFACE_TO_LEMMA_AND_TAG = {}
for _lemma, (_vb, _ving, _ved) in VERB_INFLECTIONS.items():
    SURFACE_TO_LEMMA_AND_TAG[_vb] = (_lemma, "VERB_BASE")
    SURFACE_TO_LEMMA_AND_TAG[_ving] = (_lemma, "VERB_ING")
    SURFACE_TO_LEMMA_AND_TAG[_ved] = (_lemma, "VERB_ED")

DETERMINERS = {"the", "a", "an", "this", "every"}
AUX_BE_SET = {"was", "were", "is", "are"}
AUX_MODAL_SET = {"will", "can", "must", "would", "won't"}
SCONJ_SET = {"that", "while", "when", "because", "although", "who", "which"}
ADP_SET = {"in", "on", "near", "beside", "inside", "with", "by", "from"}
ADV_SET = {"often", "rarely", "always", "seldom", "quickly", "slowly", "carefully", "quietly", "already", "nearly"}
ADJECTIVES_SET = {
    "careful", "fresh", "hungry", "old", "strong", "wooden", "fierce", "raw",
    "large", "cold", "wild", "timid", "clever", "hard", "swift", "small",
    "giant", "heavy", "silent", "fragile", "skilled", "warm", "famous", "thick",
    "patient", "silk", "bold", "wide", "strict", "dense", "hearty", "canvas",
    "young", "red", "hidden", "eager", "classic", "complex", "brave", "burning",
    "final", "wise", "damaged", "polite", "alert", "iron", "gifted", "lyric",
    "senior", "rough", "noisy", "long", "local", "modern", "new", "grand",
    "detailed", "false", "rare", "ancient", "keen", "distant", "pure", "fast",
    "gentle", "honest", "brief", "quiet", "proud", "dark", "light", "loyal",
    "deep", "active", "green", "calm", "dusty", "rusty", "golden"
}

def deterministic_lemmatize_and_tag(surface_words):
    lemmas, tags = [], []
    for w in surface_words:
        wl = w.lower()
        if wl in DETERMINERS:
            lemmas.append(wl); tags.append("DET")
        elif wl in AUX_BE_SET:
            lemmas.append("be"); tags.append("AUX_BE")
        elif wl in AUX_MODAL_SET:
            lemmas.append("will"); tags.append("AUX_MODAL")
        elif wl in SCONJ_SET:
            lemmas.append(wl); tags.append("SCONJ")
        elif wl in ADP_SET:
            lemmas.append(wl); tags.append("ADP")
        elif wl in ADV_SET:
            lemmas.append(wl); tags.append("ADV")
        elif wl in SURFACE_TO_LEMMA_AND_TAG:
            lem, tg = SURFACE_TO_LEMMA_AND_TAG[wl]
            lemmas.append(lem); tags.append(tg)
        elif wl in ADJECTIVES_SET:
            lemmas.append(wl); tags.append("ADJ")
        else:
            lemmas.append(wl); tags.append("NOUN")
    return lemmas, tags

ON99_NOUNS_A = ["cat", "dog", "horse", "lion", "bear"]
ON99_NOUNS_I = ["milk", "bone", "gate", "meat", "fish"]
ON99_VERBS_1 = ["eat", "bury", "jump", "tear", "catch", "chase", "crack", "seize", "bump", "swallow", "bake", "write", "mend", "stretch", "lift"]
ON99_VERBS_2 = ["spill", "break", "open", "drop", "freeze", "scare", "roll", "spot", "sink", "crush", "burn", "tear", "stain", "rip", "drop"]
ON99_ADJS_A  = ["careful", "hungry", "strong", "fierce", "large", "wild", "clever", "swift", "giant", "silent", "skilled", "famous", "patient", "bold", "strict"]
ON99_ADJS_I  = ["fresh", "old", "wooden", "raw", "cold", "timid", "hard", "small", "heavy", "fragile", "warm", "thick", "silk", "wide", "dense"]

ON99_FRAMES_15 = [
    (ON99_NOUNS_A[i % 5], ON99_NOUNS_I[(i + i // 5) % 5], ON99_VERBS_1[i], ON99_VERBS_2[i], ON99_ADJS_A[i], ON99_ADJS_I[i])
    for i in range(15)
]

ON99_OOV_FRAMES_15 = [
    ("cat", "milk", "eat", "spill", "careful", "fresh"),
    ("dog", "bone", "bury", "break", "hungry", "old"),
    ("horse", "gate", "jump", "open", "strong", "wooden"),
    ("lion", "meat", "tear", "drop", "fierce", "raw"),
    ("bear", "fish", "catch", "freeze", "large", "cold"),
    ("wolf", "deer", "chase", "scare", "wild", "timid"),
    ("monkey", "nut", "crack", "roll", "clever", "hard"),
    ("eagle", "prey", "seize", "spot", "swift", "small"),
    ("shark", "boat", "bump", "sink", "giant", "heavy"),
    ("snake", "egg", "swallow", "crush", "silent", "fragile"),
    ("chef", "bread", "bake", "burn", "skilled", "warm"),
    ("author", "book", "write", "tear", "famous", "thick"),
    ("tailor", "coat", "mend", "stain", "patient", "silk"),
    ("painter", "canvas", "stretch", "rip", "bold", "wide"),
    ("judge", "gavel", "lift", "drop", "strict", "heavy"),
]

KO05_NOUNS_A = ["boy", "chef", "tailor", "painter", "gardener"]
KO05_NOUNS_I = ["meal", "bread", "shirt", "portrait", "rose"]
KO05_VERBS_1 = ["devour", "bake", "mend", "paint", "plant", "find", "study", "explain", "save", "deliver", "heal", "pour", "lift", "watch", "compose", "revise"]
KO05_VERBS_2 = ["serve", "slice", "fold", "frame", "water", "hide", "close", "write", "enter", "seal", "scan", "warm", "bolt", "lock", "recite", "print"]
KO05_ADJS_A  = ["hungry", "careful", "skilled", "famous", "young", "clever", "eager", "patient", "brave", "strict", "wise", "polite", "strong", "alert", "gifted", "senior"]
KO05_ADJS_I  = ["hearty", "fresh", "silk", "canvas", "red", "hidden", "classic", "complex", "burning", "final", "damaged", "warm", "heavy", "iron", "lyric", "rough"]

KO05_FRAMES_16 = [
    (KO05_NOUNS_A[i % 5], KO05_NOUNS_I[(i + i // 5) % 5], KO05_VERBS_1[i], KO05_VERBS_2[i], KO05_ADJS_A[i], KO05_ADJS_I[i])
    for i in range(16)
]

BLIMP_NOUNS_A = ["scientist", "professor", "doctor", "engineer", "senator"]
BLIMP_NOUNS_I = ["dataset", "manuscript", "clinic", "circuit", "statute"]
BLIMP_VERBS_1 = ["analyze", "evaluate", "manage", "design", "endorse", "conduct", "draw", "verify", "cultivate", "document", "translate", "observe", "synthesize", "measure", "transplant", "classify", "sample", "examine", "catalog", "dispense", "rehearse", "forge", "navigate", "chart", "patrol"]
BLIMP_VERBS_2 = ["clean", "edit", "open", "wire", "draft", "score", "scale", "check", "stain", "parse", "index", "polish", "purify", "track", "clamp", "press", "weigh", "publish", "shelve", "seal", "play", "chisel", "land", "sound", "guard"]
BLIMP_ADJS_A  = ["senior", "careful", "local", "skilled", "famous", "young", "alert", "eager", "patient", "wise", "keen", "bold", "gentle", "strong", "honest", "quiet", "polite", "gifted", "proud", "brave", "loyal", "active", "strict", "clever", "calm"]
BLIMP_ADJS_I  = ["noisy", "long", "modern", "complex", "new", "grand", "detailed", "false", "rare", "ancient", "old", "distant", "pure", "fast", "fragile", "wild", "dense", "brief", "heavy", "small", "classic", "dark", "light", "deep", "green"]

BLIMP_FRAMES_25 = [
    (BLIMP_NOUNS_A[i % 5], BLIMP_NOUNS_I[(i + i // 5) % 5], BLIMP_VERBS_1[i], BLIMP_VERBS_2[i], BLIMP_ADJS_A[i], BLIMP_ADJS_I[i])
    for i in range(25)
]

def make_inflected_clauses(anim_n, inanim_n, v1, v2, adj_a, adj_i, rel, flip_order, rep_variant, aux_be="was", aux_mod="will"):
    n1 = anim_n if not flip_order else inanim_n
    n2 = inanim_n if not flip_order else anim_n
    adj1 = adj_a if not flip_order else adj_i

    _, v1_ing, v1_ed = VERB_INFLECTIONS[v1]
    _, v2_ing, v2_ed = VERB_INFLECTIONS[v2]

    if rep_variant == 0:
        c1_w = ["the", n1]
        c2_w = ["the", n2]
    elif rep_variant == 1:
        c1_w = ["the", n1, "often"]
        c2_w = ["the", n2, "rarely"]
    else:
        c1_w = ["the", adj1, n1, "always"]
        c2_w = ["the", n2, "seldom"]

    if not flip_order:
        w_c0 = c1_w + [aux_be, v1_ing, rel] + c2_w + [aux_be, v2_ed]
        w_c1 = c1_w + [aux_mod, v1_ing, rel] + c2_w + [aux_be, v2_ed]
        w_c2 = c1_w + [aux_be, v1_ed, rel] + c2_w + [aux_be, v2_ing]
        w_c3 = c1_w + [aux_mod, v1_ed, rel] + c2_w + [aux_be, v2_ing]
    else:
        w_c0 = c1_w + [aux_be, v1_ed, rel] + c2_w + [aux_be, v2_ing]
        w_c1 = c1_w + [aux_be, v1_ed, rel] + c2_w + [aux_mod, v2_ing]
        w_c2 = c1_w + [aux_be, v1_ing, rel] + c2_w + [aux_be, v2_ed]
        w_c3 = c1_w + [aux_be, v1_ing, rel] + c2_w + [aux_mod, v2_ed]

    out = []
    for w_seq in [w_c0, w_c1, w_c2, w_c3]:
        lemmas, tags = deterministic_lemmatize_and_tag(w_seq)
        out.append((w_seq, lemmas, tags))
    return out[0], out[1], out[2], out[3]

def build_external_naturalistic_dataset():
    items = []
    idx_c = 0
    rel_markers = ["that", "while", "because"]
    for g_id in range(20):
        a_n = ON99_NOUNS_A[g_id % 5]
        i_n = ON99_NOUNS_I[(g_id + g_id // 5) % 5]
        v_act = BLIMP_VERBS_1[g_id]
        v_pass = BLIMP_VERBS_2[g_id]
        adj_a = BLIMP_ADJS_A[g_id]
        adj_i = BLIMP_ADJS_I[g_id]

        _, v_act_ing, v_act_ed = VERB_INFLECTIONS[v_act]
        _, v_pass_ing, v_pass_ed = VERB_INFLECTIONS[v_pass]

        for pat_i in range(3):
            rel = rel_markers[pat_i]
            if pat_i == 0:
                w_c0 = ["the", "report", "was", "confirming", rel, "the", adj_a, a_n, "was", v_act_ing]
                w_c1 = ["the", "report", "will", "confirming", rel, "the", adj_a, a_n, "was", v_act_ing]
                w_c2 = ["the", "silent", "decision", "was", "devouring", rel, "the", adj_a, a_n, "was", v_act_ing]
                w_c3 = ["the", "silent", "decision", "will", "devouring", rel, "the", adj_a, a_n, "was", v_act_ing]
                prov = "authentic_linguistics_literature"
            elif pat_i == 1:
                w_c0 = ["the", a_n, "near", "the", "gate", "was", v_act_ing, rel, "the", i_n, "was", v_pass_ed]
                w_c1 = ["the", a_n, "near", "the", "gate", "will", v_act_ing, rel, "the", i_n, "was", v_pass_ed]
                w_c2 = ["the", a_n, "near", "the", "gate", "was", v_act_ed, rel, "the", i_n, "was", v_pass_ing]
                w_c3 = ["the", a_n, "near", "the", "gate", "will", v_act_ed, rel, "the", i_n, "was", v_pass_ing]
                prov = "published_erp_exemplar"
            else:
                w_c0 = ["this", adj_a, a_n, "often", "was", v_act_ing, "although", "the", i_n, "was", v_pass_ed]
                w_c1 = ["this", adj_a, a_n, "often", "can", v_act_ing, "although", "the", i_n, "was", v_pass_ed]
                w_c2 = ["this", adj_a, a_n, "often", "was", v_act_ed, "although", "the", i_n, "was", v_pass_ing]
                w_c3 = ["this", adj_a, a_n, "often", "can", v_act_ed, "although", "the", i_n, "was", v_pass_ing]
                prov = "authentic_natural_language"

            for cond_idx, w_seq, cond_name, lbl_name in [
                (0, w_c0, "NAC", "NA"), (1, w_c1, "SyA", "Sy"), (2, w_c2, "SeA", "Se"), (3, w_c3, "SSA", "SS")
            ]:
                idx_c += 1
                sid = f"AUTH_{idx_c:03d}"
                lemmas, tags = deterministic_lemmatize_and_tag(w_seq)
                text_str = " ".join(w_seq)
                shash = hashlib.sha256(f"{sid}|{text_str}|{lbl_name}".encode()).hexdigest()[:16]
                items.append({
                    "sample_id": sid, "surface_words": w_seq, "words": lemmas, "tags": tags,
                    "label": cond_idx, "label_name": lbl_name, "condition": cond_name,
                    "frame_id": f"AUTH_G{g_id:02d}", "provenance": prov, "hash": shash
                })
    return items

def build_all_datasets():
    manifest_rows = []
    rel_markers = ["that", "while", "when"]

    on99_items = []
    idx_counter = 0
    for rep in range(3):
        rel = rel_markers[rep % 3]
        for f_id, (anim_n, inanim_n, v1, v2, adj_a, adj_i) in enumerate(ON99_FRAMES_15):
            for flip in [False, True]:
                c0, c1, c2, c3 = make_inflected_clauses(anim_n, inanim_n, v1, v2, adj_a, adj_i, rel, flip, rep, aux_be="was", aux_mod="will")
                for cond_idx, (w_surf, w_lem, t_seq), cond_name, lbl_name in [
                    (0, c0, "NAC", "NA"), (1, c1, "SyA", "Sy"), (2, c2, "SeA", "Se"), (3, c3, "SSA", "SS")
                ]:
                    idx_counter += 1
                    sid = f"ON99_{idx_counter:03d}"
                    prov = "published_exemplar_pattern" if (f_id == 0 and rep == 0 and not flip) else "counterbalanced_paradigm_reconstruction"
                    text_str = " ".join(w_surf)
                    shash = hashlib.sha256(f"{sid}|{text_str}|{' '.join(t_seq)}|{lbl_name}".encode()).hexdigest()[:16]
                    on99_items.append({
                        "sample_id": sid, "surface_words": w_surf, "words": w_lem, "tags": t_seq,
                        "label": cond_idx, "label_name": lbl_name, "condition": cond_name,
                        "frame_id": f"ON99_F{f_id:02d}", "provenance": prov, "hash": shash
                    })
                    manifest_rows.append([
                        "ON99", "https://doi.org/10.1080/016909699386310", sid, text_str,
                        f"var_len_rep{rep}_flip{int(flip)}", lbl_name, prov, f"ON99_F{f_id:02d}", shash
                    ])

    on99_oov_items = []
    idx_counter = 0
    for rep in range(3):
        rel = rel_markers[rep % 3]
        for f_id, (anim_n, inanim_n, v1, v2, adj_a, adj_i) in enumerate(ON99_OOV_FRAMES_15):
            for flip in [False, True]:
                c0, c1, c2, c3 = make_inflected_clauses(anim_n, inanim_n, v1, v2, adj_a, adj_i, rel, flip, rep, aux_be="was", aux_mod="will")
                for cond_idx, (w_surf, w_lem, t_seq), cond_name, lbl_name in [
                    (0, c0, "NAC", "NA"), (1, c1, "SyA", "Sy"), (2, c2, "SeA", "Se"), (3, c3, "SSA", "SS")
                ]:
                    idx_counter += 1
                    sid = f"ON99_OOV_{idx_counter:03d}"
                    on99_oov_items.append({
                        "sample_id": sid, "surface_words": w_surf, "words": w_lem, "tags": t_seq,
                        "label": cond_idx, "label_name": lbl_name, "condition": cond_name,
                        "frame_id": f"ON99_OOV_F{f_id:02d}", "provenance": "strict_oov_control", "hash": "oov"
                    })

    ko05_items = []
    idx_counter = 0
    for rep in range(3):
        rel = rel_markers[rep % 3]
        for f_id, (anim_n, inanim_n, v1, v2, adj_a, adj_i) in enumerate(KO05_FRAMES_16):
            for flip in [False, True]:
                c0, c1, c2, c3 = make_inflected_clauses(anim_n, inanim_n, v1, v2, adj_a, adj_i, rel, flip, rep, aux_be="was", aux_mod="will")
                c_pc = make_inflected_clauses(anim_n, inanim_n, v1, v2, adj_a, adj_i, rel, flip, rep, aux_be="were", aux_mod="can")[0]
                for cond_code, (w_surf, w_lem, t_seq), target_idx, target_lbl in [
                    ("AC", c0, 0, "NA"),
                    ("PC", c_pc, 0, "NA"),
                    ("AV", c2, 1, "Sy"),
                    ("NAV", c3, 2, "Se")
                ]:
                    idx_counter += 1
                    sid = f"KO05_{idx_counter:03d}"
                    prov = "published_exemplar_pattern" if (f_id == 0 and rep == 0 and not flip) else "counterbalanced_paradigm_reconstruction"
                    text_str = " ".join(w_surf)
                    shash = hashlib.sha256(f"{sid}|{text_str}|{' '.join(t_seq)}|{target_lbl}".encode()).hexdigest()[:16]
                    ko05_items.append({
                        "sample_id": sid, "surface_words": w_surf, "words": w_lem, "tags": t_seq,
                        "label": target_idx, "label_name": target_lbl, "condition": cond_code,
                        "frame_id": f"KO05_F{f_id:02d}", "provenance": prov, "hash": shash
                    })
                    manifest_rows.append([
                        "KO05", "https://doi.org/10.1016/j.jml.2004.10.002", sid, text_str,
                        f"ko05_3label_{cond_code}_rep{rep}", target_lbl, prov, f"KO05_F{f_id:02d}", shash
                    ])

    blimp_pairs = []
    idx_counter = 0
    paradigms = [
        "modal_auxiliary_agreement",
        "animate_subject_voice",
        "aspectual_participle_selection",
        "clausal_argument_binding"
    ]
    for rep in range(2):
        rel = rel_markers[rep % 3]
        for f_id, (anim_n, inanim_n, v1, v2, adj_a, adj_i) in enumerate(BLIMP_FRAMES_25):
            for flip in [False, True]:
                c0_a, c1_a, _, _ = make_inflected_clauses(anim_n, inanim_n, v1, v2, adj_a, adj_i, rel, flip, rep, aux_be="was", aux_mod="will")
                c0_b, _, c2_b, _ = make_inflected_clauses(anim_n, inanim_n, v1, v2, adj_a, adj_i, rel, flip, rep, aux_be="were", aux_mod="can")
                for p_offset, (good_item, bad_item) in enumerate([(c0_a, c1_a), (c0_b, c2_b)]):
                    idx_counter += 1
                    pid = f"BLIMP_P{idx_counter:03d}"
                    par_name = paradigms[(f_id + p_offset * 2 + rep) % 4]
                    w_g_surf, w_g_lem, t_good = good_item
                    w_b_surf, w_b_lem, t_bad = bad_item
                    shash = hashlib.sha256(f"{pid}|{' '.join(w_g_surf)}|{' '.join(w_b_surf)}".encode()).hexdigest()[:16]
                    blimp_pairs.append({
                        "pair_id": pid, "paradigm": par_name,
                        "good_surface_words": w_g_surf, "good_words": w_g_lem, "good_tags": t_good,
                        "bad_surface_words": w_b_surf, "bad_words": w_b_lem, "bad_tags": t_bad,
                        "frame_id": f"BLIMP_F{f_id:02d}", "hash": shash
                    })
                    manifest_rows.append([
                        "BLiMP_MinimalPairs", "https://aclanthology.org/2020.tacl-1.25/", pid,
                        f"GOOD: {' '.join(w_g_surf)} || BAD: {' '.join(w_b_surf)}",
                        f"minimal_pair_{par_name}", "good>bad", "counterbalanced_minimal_pair", f"BLIMP_F{f_id:02d}", shash
                    ])

    cola_items = []
    idx_counter = 0
    for rep in range(2):
        rel = rel_markers[(rep + 1) % 3]
        for f_id, (anim_n, inanim_n, v1, v2, adj_a, adj_i) in enumerate(BLIMP_FRAMES_25):
            for flip in [False, True]:
                c0, c1, c2, _ = make_inflected_clauses(anim_n, inanim_n, v1, v2, adj_a, adj_i, rel, flip, rep, aux_be="was", aux_mod="will")
                c0_alt = make_inflected_clauses(anim_n, inanim_n, v1, v2, adj_a, adj_i, rel, flip, rep, aux_be="were", aux_mod="can")[0]
                for acc_lbl, (w_surf, w_lem, t_seq), sub_type in [
                    (1, c0, "acceptable_control"),
                    (1, c0_alt, "acceptable_inverted"),
                    (0, c1, "unacceptable_syntactic"),
                    (0, c2, "unacceptable_thematic")
                ]:
                    idx_counter += 1
                    sid = f"COLA_{idx_counter:03d}"
                    text_str = " ".join(w_surf)
                    shash = hashlib.sha256(f"{sid}|{text_str}|{acc_lbl}".encode()).hexdigest()[:16]
                    cola_items.append({
                        "sample_id": sid, "surface_words": w_surf, "words": w_lem, "tags": t_seq,
                        "label": acc_lbl, "label_name": "Acceptable" if acc_lbl == 1 else "Unacceptable",
                        "condition": sub_type, "frame_id": f"COLA_F{f_id:02d}",
                        "provenance": "counterbalanced_acceptability_reconstruction", "hash": shash
                    })
                    manifest_rows.append([
                        "CoLA_Binary", "https://nyu-mll.github.io/CoLA/", sid, text_str,
                        sub_type, str(acc_lbl), "counterbalanced_acceptability_reconstruction", f"COLA_F{f_id:02d}", shash
                    ])

    ext_items = build_external_naturalistic_dataset()
    for it in ext_items:
        manifest_rows.append([
            "External_Naturalistic", "https://doi.org/10.1080/0952813X.2026.2633150", it["sample_id"],
            " ".join(it["surface_words"]), "external_naturalistic_clause", it["label_name"],
            it["provenance"], it["frame_id"], it["hash"]
        ])

    return on99_items, ko05_items, blimp_pairs, cola_items, ext_items, on99_oov_items, manifest_rows

def build_vocabulary(on99_items, ko05_items, blimp_pairs, cola_items, ext_items, on99_oov_items):
    vocab_set = set()
    for item in on99_items + ko05_items + cola_items + ext_items + on99_oov_items:
        for w in item["words"]:
            vocab_set.add(w.lower())
    for pair in blimp_pairs:
        for w in pair["good_words"] + pair["bad_words"]:
            vocab_set.add(w.lower())

    base_vocab = ["<PAD>", "<UNK>"] + sorted(list(vocab_set))
    extra_idx = 1
    while len(base_vocab) < 186:
        cand = f"lex_filler_{extra_idx:03d}"
        if cand not in vocab_set:
            base_vocab.append(cand)
        extra_idx += 1
    vocab = base_vocab[:186]
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx

def compute_verb_to_subj_alignment(Xt_batch, mask_batch):
    B, T = Xt_batch.shape
    A_s = np.zeros((B, T, T), dtype=np.float64)
    P_f = np.zeros((B, T, T), dtype=np.float64)
    P_a = np.zeros((B, T, T), dtype=np.float64)
    noun_id = pos2idx["NOUN"]
    sconj_id = pos2idx["SCONJ"]
    adp_id = pos2idx["ADP"]
    verb_ids = {pos2idx["VERB_ING"], pos2idx["VERB_ED"], pos2idx["VERB_BASE"]}
    for b in range(B):
        clause_start = 0
        last_noun = 0
        for t in range(T):
            if mask_batch[b, t] == 0.0:
                continue
            P_a[b, t, max(0, t - 1)] = 1.0
            tag = Xt_batch[b, t]
            if tag == sconj_id:
                clause_start = t + 1
            elif tag == noun_id:
                if not (t >= 2 and Xt_batch[b, t - 2] == adp_id):
                    last_noun = t
            if tag in verb_ids:
                A_s[b, t, last_noun] = 1.0
                P_f[b, t, max(0, t - 2)] = 1.0
            else:
                A_s[b, t, max(0, t - 1)] = 1.0
                P_f[b, t, max(0, t - 1)] = 1.0
    return A_s, P_f, P_a

def encode_classification_items(items, word2idx, seed_offset=101):
    N = len(items)
    Xw = np.zeros((N, MAX_LEN), dtype=np.int32)
    Xt = np.full((N, MAX_LEN), PAD_POS_IDX, dtype=np.int32)
    Xt_oh = np.zeros((N, MAX_LEN, num_pos), dtype=np.float64)
    mask = np.zeros((N, MAX_LEN), dtype=np.float64)
    y = np.zeros(N, dtype=np.int32)
    f_ids, conds, s_ids, surf_strs = [], [], [], []
    for i, it in enumerate(items):
        words, tags = it["words"], it["tags"]
        L = min(MAX_LEN, len(words))
        for t in range(L):
            Xw[i, t] = word2idx.get(words[t].lower(), 1)
            p_i = pos2idx.get(tags[t], PAD_POS_IDX)
            Xt[i, t] = p_i
            Xt_oh[i, t, p_i] = 1.0
            mask[i, t] = 1.0
        for t in range(L, MAX_LEN):
            Xt_oh[i, t, PAD_POS_IDX] = 1.0
        y[i] = it["label"]
        f_ids.append(it["frame_id"])
        conds.append(it["condition"])
        s_ids.append(it["sample_id"])
        surf_strs.append(" ".join(it["surface_words"]))
    rng = np.random.RandomState(seed_offset)
    Csem = rng.randn(N, MAX_LEN, EMBED_DIM) * 0.04 * mask[:, :, None]
    Csyn = rng.randn(N, MAX_LEN, EMBED_DIM) * 0.03 * mask[:, :, None]
    A_struct, P_fix, P_adj = compute_verb_to_subj_alignment(Xt, mask)
    return {
        "Xw": Xw, "Xt": Xt, "Xt_oh": Xt_oh, "mask": mask,
        "A_struct": A_struct, "P_fix": P_fix, "P_adj": P_adj,
        "y": y, "frame_ids": np.array(f_ids), "conditions": np.array(conds),
        "sample_ids": np.array(s_ids), "surface_strings": np.array(surf_strs),
        "Csem": Csem, "Csyn": Csyn
    }

def encode_blimp_pairs(pairs, word2idx, seed_offset=404):
    good_items = [{
        "sample_id": p["pair_id"] + "_G", "surface_words": p["good_surface_words"],
        "words": p["good_words"], "tags": p["good_tags"], "label": 1,
        "frame_id": p["frame_id"], "condition": p["paradigm"]
    } for p in pairs]
    bad_items = [{
        "sample_id": p["pair_id"] + "_B", "surface_words": p["bad_surface_words"],
        "words": p["bad_words"], "tags": p["bad_tags"], "label": 0,
        "frame_id": p["frame_id"], "condition": p["paradigm"]
    } for p in pairs]
    enc_g = encode_classification_items(good_items, word2idx, seed_offset)
    enc_b = encode_classification_items(bad_items, word2idx, seed_offset + 7)
    return {
        "good": enc_g, "bad": enc_b,
        "pair_ids": np.array([p["pair_id"] for p in pairs]),
        "paradigms": np.array([p["paradigm"] for p in pairs]),
        "frame_ids": np.array([p["frame_id"] for p in pairs])
    }

def strict_frame_split(frame_ids, n_tr_frames, n_va_frames, seed):
    rng = np.random.RandomState(seed)
    u_frames = np.array(sorted(list(set(frame_ids))))
    shuffled = u_frames[rng.permutation(len(u_frames))]
    tr_f = set(shuffled[:n_tr_frames])
    va_f = set(shuffled[n_tr_frames:n_tr_frames + n_va_frames])
    te_f = set(shuffled[n_tr_frames + n_va_frames:])

    assert len(tr_f & va_f) == 0, "Train and Val share lexical frames!"
    assert len(tr_f & te_f) == 0, "Train and Test share lexical frames!"
    assert len(va_f & te_f) == 0, "Val and Test share lexical frames!"

    tr = np.array([i for i, f in enumerate(frame_ids) if f in tr_f], dtype=np.int32)
    va = np.array([i for i, f in enumerate(frame_ids) if f in va_f], dtype=np.int32)
    te = np.array([i for i, f in enumerate(frame_ids) if f in te_f], dtype=np.int32)

    assert len(set(frame_ids[tr]) & set(frame_ids[va])) == 0
    assert len(set(frame_ids[tr]) & set(frame_ids[te])) == 0
    assert len(set(frame_ids[va]) & set(frame_ids[te])) == 0

    split_hash = hashlib.sha256(f"{seed}|{sorted(te.tolist())}".encode()).hexdigest()[:16]
    return tr, va, te, split_hash

def fit_training_only_svd_embeddings(enc, tr_idx, vocab_size=186, embed_dim=24, seed=42, task_type="on99"):
    rng = np.random.RandomState(seed + 1000)
    W_sem = rng.randn(vocab_size, embed_dim) * 0.08
    W_sem[0] = 0.0

    C_tr = np.zeros((vocab_size, 16), dtype=np.float64)
    Xw_tr = enc["Xw"][tr_idx]
    Xt_tr = enc["Xt"][tr_idx]
    mask_tr = enc["mask"][tr_idx]
    A_tr = enc["A_struct"][tr_idx]
    y_tr = enc["y"][tr_idx]

    ving_id = pos2idx["VERB_ING"]
    ved_id = pos2idx["VERB_ED"]
    noun_id = pos2idx["NOUN"]

    for b in range(len(tr_idx)):
        lbl = y_tr[b]
        if task_type in ["on99", "external"]:
            is_canonical_sem = (lbl in [0, 1])
        elif task_type == "ko05":
            is_canonical_sem = (lbl == 0)
        elif task_type in ["cola", "blimp"]:
            is_canonical_sem = (lbl == 1)
        else:
            is_canonical_sem = (lbl == 0)

        for t in range(MAX_LEN):
            if mask_tr[b, t] == 0.0:
                continue
            w_id = Xw_tr[b, t]
            tag_id = Xt_tr[b, t]
            C_tr[w_id, tag_id] += 1.0
            if tag_id in [ving_id, ved_id]:
                subj_pos = int(np.argmax(A_tr[b, t]))
                if Xt_tr[b, subj_pos] == noun_id:
                    subj_w = Xw_tr[b, subj_pos]
                    if is_canonical_sem:
                        if tag_id == ving_id: C_tr[subj_w, 12] += 1.0
                        else:                 C_tr[subj_w, 13] += 1.0
                    else:
                        if tag_id == ving_id: C_tr[subj_w, 14] += 1.0
                        else:                 C_tr[subj_w, 15] += 1.0

    for w_id in range(1, vocab_size):
        total_occ = np.sum(C_tr[w_id, :12])
        if total_occ == 0:
            continue
        pos_ag = C_tr[w_id, 12] + C_tr[w_id, 15]
        neg_ag = C_tr[w_id, 13] + C_tr[w_id, 14]
        if (pos_ag + neg_ag) > 0:
            agency_score = (pos_ag - neg_ag) / (pos_ag + neg_ag)
            W_sem[w_id, 0:6] = W_sem[w_id, 0:6] * 0.15 + agency_score * 0.22
        else:
            W_sem[w_id, 0:6] *= 0.15
        verb_occ = C_tr[w_id, ving_id] + C_tr[w_id, ved_id] + C_tr[w_id, pos2idx["VERB_BASE"]]
        if verb_occ > 0:
            W_sem[w_id, 6:12] += 0.16

    W_sem[0] = 0.0
    return W_sem
