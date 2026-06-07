import json
import numpy as np
import pandas as pd
from Bio import SeqIO, Entrez
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, mean_absolute_error,
    confusion_matrix, roc_auc_score, roc_curve, auc, balanced_accuracy_score
)
from sklearn.tree import DecisionTreeClassifier # NEW: Meta-Classifier
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import pickle
import argparse
import os

warnings.filterwarnings('ignore')
Entrez.email = "example@example.com"

# ─────────────────────────────────────────────
# GLOBAL VARIABLES
# ─────────────────────────────────────────────
nuc2idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# Active Global Parameters (Swapped dynamically by the meta-router)
pwm_logo = None
codon_vocab = {}
emissions = None
log_emissions = None
codon_log_odds_map = None  
hmm_emissions_bg = None
hmm_emissions_cds = None

# Ensemble Parameters
meta_tree = None
clade_models = {}

class GatekeeperDeployer:
    def __init__(self, model_path='ensemble_model.pkl'):
        """Loads the serialized meta-ensemble parameters."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Train first.")
            
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.meta_tree = model_data.get('meta_tree')
        self.clade_models = model_data.get('clade_models', {})
        
        if not self.clade_models:
            raise ValueError("Corrupted model file: No clade matrices found.")
            
        # Fallback to the first available clade if a specific one isn't requested
        self.default_params = list(self.clade_models.values())[0]
        
        # HMM Transition Matrix (Isolated to the class)
        self.LOG_TRANS_TOK = np.log(np.array([
            [0.99, 0.01],  # Intergenic -> Intergenic, Intergenic -> CDS
            [0.05, 0.95]   # CDS -> Intergenic (Stop), CDS -> CDS
        ]))

    # ─────────────────────────────────────────────
    # INTERNAL HELPERS (Thread-Safe)
    # ─────────────────────────────────────────────
    def _encode(self, seq_str, vocab):
        return [vocab.get(seq_str[i:i+3], vocab.get("UNK", 64))
                for i in range(0, len(seq_str)-2, 3)]
                
    def _score_promoter(self, window, pwm_logo):
        if len(window) != 50: return -999.0
        nuc2idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        return sum(pwm_logo[nuc2idx.get(c, 0), i] for i, c in enumerate(window))

    def _score_cub(self, sequence, log_odds_map):
        score, valid = 0.0, 0
        for i in range(0, len(sequence)-2, 3):
            codon = sequence[i:i+3]
            if codon in log_odds_map:
                score += log_odds_map[codon]
                valid += 1
        return score / valid if valid > 0 else 0.0

    def _extract_features(self, seq_str):
        if len(seq_str) == 0: return [0.0, 0.0, 0.0]
        l = len(seq_str)
        return [
            (seq_str.count('G') + seq_str.count('C')) / l,
            seq_str.count('CG') / max(1, l - 1),
            seq_str.count('TG') / max(1, l - 1)
        ]

    # ─────────────────────────────────────────────
    # ISOLATED DECODERS (Updated for Multi-TIS Detection)
    # ─────────────────────────────────────────────
    def _decoder_baseline(self, seq_str, tokens, params):
        path = [0] * len(tokens)
        for t, tok in enumerate(tokens):
            if tok == params['codon_vocab'].get("ATG"):
                path[t] = 1 # Mark the TIS
                # REMOVED the 'break' and 'return' here. 
                # Now it will continue scanning the rest of the sequence for more ATGs.
        return path

    def _decoder_v3(self, seq_str, tokens, params, pwm_threshold=-5.0, cub_weight=2.0):
        path = [0] * len(tokens)
        for t, tok in enumerate(tokens):
            if tok == params['codon_vocab'].get("ATG"):
                nuc_idx = t * 3
                upstream = seq_str[nuc_idx-50:nuc_idx]
                downstream = seq_str[nuc_idx+3:nuc_idx+93] 
                
                if len(upstream) == 50 and len(downstream) >= 90:
                    p_score = self._score_promoter(upstream, params['pwm_logo'])
                    # If it passes the basic PWM threshold, evaluate it
                    if p_score >= pwm_threshold:
                        cub_score = self._score_cub(downstream, params['codon_log_odds'])
                        
                        # REMOVED the max() bottleneck. 
                        # Now, if the combined context score is strong enough (e.g., > 0), 
                        # we plot it as a valid predicted TIS.
                        if p_score + (cub_score * cub_weight) > 0.0:
                            path[t] = 1
        return path

    def _decoder_hmm_tokenized(self, seq_str, tokens, params):
        n_obs = len(tokens)
        if n_obs == 0: return [0] * len(tokens)
        
        viterbi = np.full((2, n_obs), -np.inf)
        backpointer = np.zeros((2, n_obs), dtype=int)
        viterbi[0, 0] = 0.0 
        
        for t in range(1, n_obs):
            codon_str = seq_str[t*3:t*3+3]
            em_1 = params['codon_log_odds'].get(codon_str, 0.0)
            
            gate_score = -np.inf
            if codon_str == "ATG":
                upstream = seq_str[t*3-50:t*3]
                if len(upstream) == 50:
                    gate_score = self._score_promoter(upstream, params['pwm_logo'])
                    
            p00 = viterbi[0, t-1] + self.LOG_TRANS_TOK[0, 0]
            p10 = viterbi[1, t-1] + self.LOG_TRANS_TOK[1, 0]
            viterbi[0, t], backpointer[0, t] = (p00, 0) if p00 > p10 else (p10, 1)
            
            p01 = viterbi[0, t-1] + self.LOG_TRANS_TOK[0, 1] + gate_score + em_1
            p11 = viterbi[1, t-1] + self.LOG_TRANS_TOK[1, 1] + em_1
            viterbi[1, t], backpointer[1, t] = (p01, 0) if p01 > p11 else (p11, 1)
            
        curr_state = np.argmax(viterbi[:, -1])
        path = [0] * n_obs
        for t in range(n_obs-1, -1, -1):
            path[t] = curr_state
            curr_state = backpointer[curr_state, t]
            
        out_path = [0] * n_obs
        for t in range(1, n_obs):
            # A TIS is defined as transitioning from Intergenic (0) into CDS (1)
            if path[t] == 1 and path[t-1] == 0:
                out_path[t] = 1
                # REMOVED the 'break' statement here.
                # Now it will map every single valid gene start in a polycistronic sequence.
        return out_path

    # ─────────────────────────────────────────────
    # MAIN INFERENCE ROUTER
    # ─────────────────────────────────────────────
    def predict(self, raw_sequence, strategy="hmm_tokenized", force_clade=None):
        """
        Predicts gene structure using the specified strategy.
        
        Supported Strategies: "baseline", "v3", "hmm_tokenized", "meta_router"
        """
        seq_str = raw_sequence.upper().strip()
        if len(seq_str) < 53: # Absolute minimum needed for upstream + ATG
            return []

        # 1. Determine Parameters (Matrix Selection)
        if force_clade and force_clade in self.clade_models:
            model_params = self.clade_models[force_clade]
        elif strategy == "meta_router" and self.meta_tree:
            features = self._extract_features(seq_str)
            predicted_clade = self.meta_tree.predict([features])[0]
            model_params = self.clade_models.get(predicted_clade, self.default_params)
            # If using meta_router, the actual execution defaults to HMM
            strategy = "hmm_tokenized" 
        else:
            model_params = self.default_params

        # 2. Tokenize Sequence
        tokens = self._encode(seq_str, model_params['codon_vocab'])

        # 3. Route to specific algorithm
        if strategy == "baseline":
            return self._decoder_baseline(seq_str, tokens, model_params)
        elif strategy == "v3":
            return self._decoder_v3(seq_str, tokens, model_params)
        elif strategy == "hmm_tokenized":
            return self._decoder_hmm_tokenized(seq_str, tokens, model_params)
        else:
            raise ValueError(f"Unknown decoding strategy: '{strategy}'.")

# ─────────────────────────────────────────────
# HMM PARAMETERS & DECODERS (NEW)
# ─────────────────────────────────────────────

# Log Transition Matrices for the 2-State HMM (0: Intergenic, 1: CDS)
# These represent a priori expectations of sequence length.
# In a full model, these would be trained via Baum-Welch.
LOG_TRANS_TOK = np.log(np.array([
    [0.99, 0.01],  # Intergenic -> Intergenic, Intergenic -> CDS
    [0.05, 0.95]   # CDS -> Intergenic (Stop), CDS -> CDS
]))

LOG_TRANS_STRIDE1 = np.log(np.array([
    [0.999, 0.001], 
    [0.016, 0.984]  
]))

def decoder_hmm_tokenized(seq_str, tokens):
    """
    Part 1: With Codon Tokenization.
    Forces the HMM to evaluate in 3-bp reading frames. 
    Transitions from Intergenic to CDS are heavily gated by the PWM score.
    """
    n_obs = len(tokens)
    if n_obs == 0: return [0] * len(tokens)
    
    viterbi = np.full((2, n_obs), -np.inf)
    backpointer = np.zeros((2, n_obs), dtype=int)
    
    # Initialize in Intergenic state
    viterbi[0, 0] = 0.0 
    
    for t in range(1, n_obs):
        codon_str = seq_str[t*3:t*3+3]
        
        # Emission probabilities
        em_0 = 0.0 # Background null model
        em_1 = codon_log_odds_map.get(codon_str, 0.0) if codon_log_odds_map else 0.0
        
        # PWM Gatekeeper: Only allow state 0 -> 1 transition if ATG is present
        gate_score = -np.inf
        if codon_str == "ATG":
            upstream = seq_str[t*3-50:t*3]
            if len(upstream) == 50:
                # Add PWM score directly to log-probability
                gate_score = score_promoter(upstream) 
                
        # Update State 0 (Intergenic)
        p00 = viterbi[0, t-1] + LOG_TRANS_TOK[0, 0] + em_0
        p10 = viterbi[1, t-1] + LOG_TRANS_TOK[1, 0] + em_0
        if p00 > p10:
            viterbi[0, t], backpointer[0, t] = p00, 0
        else:
            viterbi[0, t], backpointer[0, t] = p10, 1
            
        # Update State 1 (CDS)
        # Transition from 0 requires the gate_score (PWM)
        p01 = viterbi[0, t-1] + LOG_TRANS_TOK[0, 1] + gate_score + em_1
        p11 = viterbi[1, t-1] + LOG_TRANS_TOK[1, 1] + em_1
        
        if p01 > p11:
            viterbi[1, t], backpointer[1, t] = p01, 0
        else:
            viterbi[1, t], backpointer[1, t] = p11, 1
            
    # Backtrack to find the optimal path
    curr_state = np.argmax(viterbi[:, -1])
    path = [0] * n_obs
    for t in range(n_obs-1, -1, -1):
        path[t] = curr_state
        curr_state = backpointer[curr_state, t]
        
    # Map back to legacy pipeline format: [0,0,1,2,2,3]
    out_path = [0] * n_obs
    for t in range(1, n_obs):
        if path[t] == 1 and path[t-1] == 0:
            out_path[t] = 1
            for i in range(t+1, n_obs): out_path[i] = 2
            break
            
    return out_path

def decoder_hmm_stride1(seq_str, tokens):
    """
    Part 2: Without Tokenization (Stride-1).
    Scans every single nucleotide offset. 
    Evaluates coding potential continuously without locking into a frame.
    """
    n_chars = len(seq_str)
    if n_chars < 50: return [0] * len(tokens)
    
    viterbi = np.full((2, n_chars), -np.inf)
    backpointer = np.zeros((2, n_chars), dtype=int)
    
    viterbi[0, 0] = 0.0
    
    for i in range(1, n_chars - 2):
        codon_sim = seq_str[i:i+3]
        
        em_0 = 0.0
        em_1 = codon_log_odds_map.get(codon_sim, 0.0) if codon_log_odds_map else 0.0
        
        gate_score = -np.inf
        if codon_sim == "ATG":
            upstream = seq_str[i-50:i]
            if len(upstream) == 50:
                gate_score = score_promoter(upstream)

        # State 0 (Intergenic)
        p00 = viterbi[0, i-1] + LOG_TRANS_STRIDE1[0, 0] + em_0
        p10 = viterbi[1, i-1] + LOG_TRANS_STRIDE1[1, 0] + em_0
        viterbi[0, i], backpointer[0, i] = (p00, 0) if p00 > p10 else (p10, 1)
        
        # State 1 (CDS)
        p01 = viterbi[0, i-1] + LOG_TRANS_STRIDE1[0, 1] + gate_score + em_1
        p11 = viterbi[1, i-1] + LOG_TRANS_STRIDE1[1, 1] + em_1
        viterbi[1, i], backpointer[1, i] = (p01, 0) if p01 > p11 else (p11, 1)

    curr_state = np.argmax(viterbi[:, -3])
    raw_path = [0] * n_chars
    for i in range(n_chars-3, -1, -1):
        raw_path[i] = curr_state
        curr_state = backpointer[curr_state, i]

    out_path = [0] * len(tokens)
    for i in range(1, n_chars):
        if raw_path[i] == 1 and raw_path[i-1] == 0:
            tok_idx = i // 3
            if tok_idx < len(out_path):
                out_path[tok_idx] = 1
                for j in range(tok_idx+1, len(out_path)): out_path[j] = 2
            break
            
    return out_path

# ─────────────────────────────────────────────
# SCORING FUNCTIONS
# ─────────────────────────────────────────────
def score_promoter(window):
    if len(window) != 50:
        return -999.0
    return sum(pwm_logo[nuc2idx.get(c, 0), i] for i, c in enumerate(window))

def score_coding_potential(sequence, log_odds_map):
    """v3: Scores based on Codon Log-Odds (CDS vs Intergenic)"""
    score = 0.0
    valid_codons = 0
    for i in range(0, len(sequence)-2, 3):
        codon = sequence[i:i+3]
        if codon in log_odds_map:
            score += log_odds_map[codon]
            valid_codons += 1
    return score / valid_codons if valid_codons > 0 else 0.0

def encode(seq_str):
    return [codon_vocab.get(seq_str[i:i+3], codon_vocab["UNK"])
            for i in range(0, len(seq_str)-2, 3)]

# ─────────────────────────────────────────────
# DATA LOADING, CACHING & TRAINING
# ─────────────────────────────────────────────
import hashlib

def load_training_sources(filepath="training_sources.json"):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def get_file_hash(filepath):
    """Generates an MD5 hash of a file to detect modifications."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return None

def get_cached_records(json_filepath, cache_filepath):
    """Hashes the JSON target and manages the local NCBI sequence cache."""
    current_hash = get_file_hash(json_filepath)
    
    if os.path.exists(cache_filepath):
        with open(cache_filepath, 'rb') as f:
            cache_data = pickle.load(f)
        # Check if the JSON file matches the state of our cached sequences
        if cache_data.get('source_hash') == current_hash:
            print(f"[✔] Valid cache found. Loading sequences from {cache_filepath}...")
            return cache_data['records']
        else:
            print(f"[!] {json_filepath} modified. Cache invalidated. Re-pulling...")
    
    print(f"Downloading records for {json_filepath} from NCBI...")
    sources = load_training_sources(json_filepath)
    records = {}
    for source in sources:
        print(f"   -> Pulling {source['name']} ({source['id']})...")
        fetch_args = {"db": "nucleotide", "id": source["id"], "rettype": "gbwithparts", "retmode": "text"}
        
        # --- NEW: DYNAMIC CODON ALIGNMENT FIX ---
        if source.get("start") and source.get("stop"):
            raw_start = int(source["start"])
            raw_stop = int(source["stop"])
            
            # Snap start down to the nearest multiple of 3
            aligned_start = raw_start - (raw_start % 3)
            # Snap stop up to ensure the length is a multiple of 3
            aligned_stop = raw_stop + ((3 - (raw_stop % 3)) % 3)
            
            fetch_args["seq_start"] = aligned_start
            fetch_args["seq_stop"] = aligned_stop
            print(f"      [i] Auto-aligned coords: {raw_start}-{raw_stop} -> {aligned_start}-{aligned_stop}")
        # ----------------------------------------
        
        try:
            with Entrez.efetch(**fetch_args) as handle:
                records[source["name"]] = SeqIO.read(handle, "genbank")
        except Exception as e:
            print(f"      [!] Fetch Failed for {source['name']}: {e}")
            
    print(f"[✔] Saving raw records to {cache_filepath}...")
    with open(cache_filepath, 'wb') as f:
        pickle.dump({'source_hash': current_hash, 'records': records}, f)
        
    return records

def train_model(filepath="training_sources.json"):
    print("1. Fetching Training Data & Building Meta-Ensemble...")
    training_sources = load_training_sources(filepath)
    records = get_cached_records(filepath, "training_cache.pkl")
    
    # Group sources by user-defined clade metadata
    clade_groups = {}
    for src in training_sources:
        clade = src.get("clade", "default")
        if clade not in clade_groups: clade_groups[clade] = []
        clade_groups[clade].append(src)
        
    final_models = {}
    dt_X = [] # Features for Decision Tree
    dt_y = [] # Labels for Decision Tree
    
    all_codons = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
    codon_vocab_val = {c: i for i, c in enumerate(all_codons)}
    codon_vocab_val["UNK"] = 64
    
    for clade, sources in clade_groups.items():
        print(f"\n--- Training Clade Pipeline: {clade.upper()} ---")
        pwm_counts = np.ones((4, 50)) * 1e-4
        cds_counts = {c: 1 for c in all_codons}
        bg_counts  = {c: 1 for c in all_codons}
        source_cds_count = 0
        bg_count = 0
        
        for source in sources:
            if source["name"] not in records: continue
            rec_train = records[source["name"]]
            offset = int(source["start"]) if source.get("start") else 0
            
            cds_intervals = []
            for f in sorted([f for f in rec_train.features if f.type == "CDS"], key=lambda x: int(x.location.start)):
                strand = f.location.strand
                tis_rel = int(f.location.start) if strand == 1 else int(f.location.end) - 1
                cds_intervals.append((int(f.location.start), int(f.location.end)))
                ws, we = (tis_rel - 150, tis_rel + 150) if strand == 1 else (tis_rel - 149, tis_rel + 151)
                
                if ws >= 0 and we <= len(rec_train.seq):
                    chunk = rec_train.seq[ws:we]
                    if strand == -1: chunk = chunk.reverse_complement()
                    seq_str = str(chunk).upper()
                    
                    if len(seq_str) == 300 and seq_str[150:153] == "ATG":
                        # Collect data for HMM
                        for i, nuc in enumerate(seq_str[100:150]):
                            if nuc in nuc2idx: pwm_counts[nuc2idx[nuc], i] += 1
                        downstream = seq_str[153:243] 
                        for i in range(0, len(downstream)-2, 3):
                            codon = downstream[i:i+3]
                            if codon in cds_counts: cds_counts[codon] += 1
                        source_cds_count += 1
                        
                        # Collect Data for Decision Tree
                        dt_X.append(extract_taxonomic_features(seq_str))
                        dt_y.append(clade)
                        
            # Intergenic extraction
            last_end = 0
            for start, end in cds_intervals:
                if int(start) > last_end + 300:
                    mid_point = last_end + ((int(start) - last_end) // 2)
                    bg_chunk = str(rec_train.seq[mid_point:mid_point+90]).upper()
                    if len(bg_chunk) == 90 and all(c in "ACGT" for c in bg_chunk):
                        for i in range(0, 90-2, 3):
                            codon = bg_chunk[i:i+3]
                            if codon in bg_counts: bg_counts[codon] += 1
                        bg_count += 1
                        
                        # Background sequences also get fed to DT
                        dt_X.append(extract_taxonomic_features(bg_chunk))
                        dt_y.append(clade)
                last_end = max(last_end, int(end))
                
        # Finalize Clade-Specific Model Matrices
        pwm = pwm_counts / pwm_counts.sum(axis=0, keepdims=True)
        pwm_logo_val = np.log(pwm / np.array([0.25]*4).reshape(4,1))
        
        tot_cds = sum(cds_counts.values())
        tot_bg = sum(bg_counts.values())
        log_p_cds = {c: np.log(count / tot_cds) for c, count in cds_counts.items()}
        log_p_bg = {c: np.log(count / tot_bg) for c, count in bg_counts.items()}
        
        final_models[clade] = {
            'pwm_logo': pwm_logo_val,
            'codon_vocab': codon_vocab_val,
            'hmm_emissions_bg': np.array([log_p_bg.get(c, -10.0) for c in all_codons] + [-10.0]),
            'hmm_emissions_cds': np.array([log_p_cds.get(c, -10.0) for c in all_codons] + [-10.0]),
            'codon_log_odds': {c: log_p_cds.get(c, -10.0) - log_p_bg.get(c, -10.0) for c in all_codons}
        }
        print(f"    [✔] Mined {source_cds_count} CDS & {bg_count} BG for {clade}.")

    print("\n2. Training Decision Tree Meta-Classifier...")
    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    if dt_X and len(set(dt_y)) > 1:
        tree_clf.fit(dt_X, dt_y)
        print(f"    [✔] Tree trained on {len(dt_X)} samples targeting {len(set(dt_y))} clades.")
    else:
        print("    [!] Insufficient clades/data for Decision Tree. Using mock fallback.")
        
    return {'meta_tree': tree_clf, 'clade_models': final_models}

def decoder_hmm(seq_str, tokens, p_start=1e-4, p_stop=1e-3, pwm_weight=1.5):
    """
    Viterbi HMM Decoder for long-range gene structure parsing.
    States: 0 = Intergenic, 1 = TIS, 2 = CDS
    """
    global hmm_emissions_bg, hmm_emissions_cds, pwm_logo, codon_vocab
    
    N = len(tokens)
    if N == 0:
        return [0] * N

    # 1. Initialize Log-Transition Matrix
    # State 0->0, 0->1, 0->2
    T_0 = [np.log(1.0 - p_start), np.log(p_start), -np.inf]
    # State 1->0, 1->1, 1->2 (Forces mandatory transition straight into CDS)
    T_1 = [-np.inf, -np.inf, 0.0]
    # State 2->0, 2->1, 2->2 (Enforces long-range continuity penalty)
    T_2 = [np.log(p_stop), -np.inf, np.log(1.0 - p_stop)]
    
    A = np.array([T_0, T_1, T_2])

    # 2. Dynamic Programming Trellis Setup
    # V[state, time]
    V = np.full((3, N), -np.inf)
    backpointer = np.zeros((3, N), dtype=int)

    # Base case initialization (Start safely in Intergenic space)
    V[0, 0] = hmm_emissions_bg[tokens[0]]
    if tokens[0] == codon_vocab.get("ATG"):
        upstream = seq_str[0:0] # Edge check
        V[1, 0] = hmm_emissions_bg[tokens[0]] + (score_promoter(upstream) * pwm_weight)

    # 3. Viterbi Forward Pass Loop
    for t in range(1, N):
        # Cache emission checks for time step t
        tok = tokens[t]
        emb_bg = hmm_emissions_bg[tok]
        emb_cds = hmm_emissions_cds[tok]

        for s_curr in range(3):
            # Compute potential transitions from all previous states
            scores = np.zeros(3)
            for s_prev in range(3):
                scores[s_prev] = V[s_prev, t-1] + A[s_prev, s_curr]

            best_prev_state = np.argmax(scores)
            backpointer[s_curr, t] = best_prev_state
            
            # Calculate Contextual State-Emission Modifiers
            if s_curr == 0:
                V[s_curr, t] = scores[best_prev_state] + emb_bg
            elif s_curr == 1:
                # Gatekeeper logic: State 1 can ONLY be claimed by an actual ATG sequence
                if tok == codon_vocab.get("ATG"):
                    nuc_idx = t * 3
                    upstream = seq_str[nuc_idx-50:nuc_idx]
                    p_score = score_promoter(upstream) if len(upstream) == 50 else -20.0
                    V[s_curr, t] = scores[best_prev_state] + emb_bg + (p_score * pwm_weight)
                else:
                    V[s_curr, t] = -np.inf # Structural layout violation
            elif s_curr == 2:
                V[s_curr, t] = scores[best_prev_state] + emb_cds

    # 4. Backtracking State Path Identification
    path = [0] * N
    best_last_state = np.argmax(V[:, N-1])
    
    # If the sequence fails completely to resolve or ends up stuck in invalid layout
    if V[best_last_state, N-1] == -np.inf:
        best_last_state = 0 

    path[N-1] = best_last_state

    for t in range(N-2, -1, -1):
        path[t] = backpointer[path[t+1], t+1]

    # Normalize output labels to match evaluation system expectations:
    # 0 = Intergenic, 1 = TIS location, 2 = CDS
    normalized_path = [0] * N
    in_cds = False
    for i in range(N):
        if path[i] == 1:
            normalized_path[i] = 1
            in_cds = True
        elif path[i] == 2 or in_cds:
            # Enforce persistence rules if lingering in structural sequence blocks
            if path[i] == 0: 
                in_cds = False # Broke out of gene via structural transition
                normalized_path[i] = 0
            else:
                normalized_path[i] = 2
        else:
            normalized_path[i] = 0

    return normalized_path

# ─────────────────────────────────────────────
# META-CLASSIFIER (DECISION TREE ROUTER)
# ─────────────────────────────────────────────
def extract_taxonomic_features(seq_str):
    """Extracts compositional features to route the sequence via Decision Tree."""
    if len(seq_str) == 0: return [0.0, 0.0, 0.0]
    length = len(seq_str)
    gc = (seq_str.count('G') + seq_str.count('C')) / length
    cg_di = seq_str.count('CG') / max(1, length - 1)
    tg_di = seq_str.count('TG') / max(1, length - 1)
    return [gc, cg_di, tg_di]

def decoder_meta_router(seq_str, tokens):
    """Uses a Decision Tree to predict clade, swaps globals, and runs HMM."""
    global pwm_logo, codon_vocab, hmm_emissions_bg, hmm_emissions_cds, codon_log_odds_map
    
    if not meta_tree or not clade_models:
        return decoder_hmm_tokenized(seq_str, tokens) # Safety fallback
        
    # 1. Predict Clade
    features = extract_taxonomic_features(seq_str)
    predicted_clade = meta_tree.predict([features])[0]
    
    # 2. Hot-Swap Globals (Hacky, but prevents rewriting all scoring functions)
    target_model = clade_models.get(predicted_clade)
    if not target_model: 
        target_model = list(clade_models.values())[0] # Fallback to first available
        
    pwm_logo = target_model['pwm_logo']
    codon_vocab = target_model['codon_vocab']
    hmm_emissions_bg = target_model['hmm_emissions_bg']
    hmm_emissions_cds = target_model['hmm_emissions_cds']
    codon_log_odds_map = target_model['codon_log_odds']
    
    # 3. Execute HMM with loaded parameters
    return decoder_hmm_tokenized(seq_str, tokens)

# ─────────────────────────────────────────────
# DECODERS (v1, v2, v3)
# ─────────────────────────────────────────────

def decoder_stride1_naive(seq_str, tokens):
    """
    Naive Stride-1 Baseline: Scans the raw string char-by-char.
    Detects EVERY ATG regardless of frame and picks the best PWM score.
    """
    best_idx, best_score = -1, -np.inf
    
    # Scan char-by-char (Stride 1) - simulates no tokenization
    for i in range(len(seq_str) - 2):
        if seq_str[i:i+3] == "ATG":
            upstream = seq_str[i-50:i]
            if len(upstream) == 50:
                p_score = score_promoter(upstream)
                if p_score > best_score:
                    best_score = p_score
                    best_idx = i
                    
    path = [0] * len(tokens)
    if best_idx != -1:
        # Map char index back to token index for compatibility with eval pipeline
        tok_idx = best_idx // 3
        if tok_idx < len(path):
            path[tok_idx] = 1
            for i in range(tok_idx + 1, len(tokens)): path[i] = 2
    return path

def decoder_v1(seq_str, tokens):
    best_start, best_score = -1, -np.inf
    for t, tok in enumerate(tokens):
        if tok == codon_vocab["ATG"]:
            nuc_idx = t * 3
            upstream = seq_str[nuc_idx-50:nuc_idx]
            if len(upstream) == 50:
                p = score_promoter(upstream)
                if p > best_score: best_score, best_start = p, t
    path = [0] * len(tokens)
    if best_start != -1:
        path[best_start] = 1
        for i in range(best_start+1, len(tokens)): path[i] = 2
    return path

def decoder_v2(seq_str, tokens, pwm_threshold=-5.0, min_cds_len_bp=90):
    candidates = []
    for t, tok in enumerate(tokens):
        if tok == codon_vocab["ATG"]:
            nuc_idx = t * 3
            upstream = seq_str[nuc_idx-50:nuc_idx]
            if len(upstream) == 50:
                p_score = score_promoter(upstream)
                if p_score >= pwm_threshold:
                    candidates.append((p_score, t))
    if not candidates: return [0] * len(tokens)
    _, best_start = max(candidates)
    if (len(tokens) - best_start - 1) * 3 < min_cds_len_bp: return [0] * len(tokens)
    path = [0] * len(tokens)
    path[best_start] = 1
    for i in range(best_start+1, len(tokens)): path[i] = 2
    return path

def decoder_v3(seq_str, tokens, pwm_threshold=-5.0, cub_weight=2.0):
    """v3 Contextual Gatekeeper: Uses Log-Odds Ratio for precise discrimination."""
    candidates = []
    global codon_log_odds_map
    
    for t, tok in enumerate(tokens):
        if tok == codon_vocab["ATG"]:
            nuc_idx = t * 3
            upstream = seq_str[nuc_idx-50:nuc_idx]
            downstream = seq_str[nuc_idx+3:nuc_idx+93] 
            
            if len(upstream) == 50 and len(downstream) >= 90:
                p_score = score_promoter(upstream)
                if p_score >= pwm_threshold:
                    cub_score = score_coding_potential(downstream, codon_log_odds_map)
                    total_score = p_score + (cub_score * cub_weight)
                    candidates.append((total_score, t))

    if not candidates: return [0] * len(tokens)
    _, best_start = max(candidates)
    path = [0] * len(tokens)
    path[best_start] = 1
    for i in range(best_start+1, len(tokens)): path[i] = 2
    return path

def decoder_baseline(seq_str, tokens):
    """Baseline: Plain codon tokenization - predicts first ATG as TIS."""
    path = [0] * len(tokens)
    for t, tok in enumerate(tokens):
        if tok == codon_vocab["ATG"]:
            path[t] = 1
            for i in range(t+1, len(tokens)): path[i] = 2
            return path
    return path

# ─────────────────────────────────────────────
# PIPELINE & EVALUATION
# ─────────────────────────────────────────────
def prepare_test_data(rec_test):
    test_cds_wins = []
    seen_test_tis = set()
    
    for f in rec_test.features:
        if f.type == "CDS":
            strand = f.location.strand
            tis_rel = int(f.location.start) if strand == 1 else int(f.location.end) - 1
            ws, we = (tis_rel - 150, tis_rel + 150) if strand == 1 else (tis_rel - 149, tis_rel + 151)
                
            if tis_rel in seen_test_tis: continue
            seen_test_tis.add(tis_rel)

            if ws >= 0 and we <= len(rec_test.seq):
                chunk = rec_test.seq[ws:we]
                if strand == -1: chunk = chunk.reverse_complement()
                seq_str = str(chunk).upper()
                if len(seq_str) == 300 and seq_str[150:153] == "ATG" and all(c in "ACGT" for c in seq_str):
                    test_cds_wins.append((seq_str, [0]*50 + [1] + [2]*48 + [3]))

    intergenics = []
    last_end = 0
    for f in sorted([f for f in rec_test.features if f.type == "CDS"], key=lambda x: int(x.location.start)):
        if int(f.location.start) > last_end + 300:
            intergenics.append((int(last_end), int(f.location.start)))
        last_end = max(last_end, int(f.location.end))

    test_int_wins = []
    for start, end in intergenics:
        for i in range(start, end - 300, 300):
            seq_str = str(rec_test.seq[i:i+300]).upper()
            if all(c in "ACGT" for c in seq_str):
                test_int_wins.append((seq_str, [0]*100))

    np.random.seed(42)
    np.random.shuffle(test_cds_wins)
    np.random.shuffle(test_int_wins)
    return test_cds_wins[:1000] + test_int_wins[:1000]

def evaluate(decoder_fn, label, test_data):
    all_t, all_p, all_scores = [], [], []
    true_starts, pred_starts = [], []
    exact_hits, total_cds = 0, 0

    for seq, lbls in test_data:
        toks = encode(seq)
        preds = decoder_fn(seq, toks)
        
        # FIX: Initialize a fallback baseline score for sequences with no predicted TIS
        final_score = -999.0 
        
        if 1 in preds:
            p_s = preds.index(1)
            if label == "v3":
                p_score = score_promoter(seq[p_s*3-50 : p_s*3])
                cub = score_coding_potential(seq[p_s*3+3 : p_s*3+93], codon_log_odds_map)
                final_score = p_score + (cub * 2.0)
            elif label == "baseline":
                final_score = 1.0 
            elif label.startswith("hmm"):
                # For the ROC curve, we can approximate the HMM's local confidence 
                # using the promoter score at its predicted TIS location
                final_score = score_promoter(seq[p_s*3-50 : p_s*3])
            else:
                final_score = score_promoter(seq[p_s*3-50 : p_s*3])
        
        bin_t = [1 if x > 0 else 0 for x in lbls]
        bin_p = [1 if x > 0 else 0 for x in preds]
        
        all_t.append(1 if sum(bin_t) > 0 else 0)
        all_p.append(1 if sum(bin_p) > 0 else 0)
        all_scores.append(final_score)

        if 1 in lbls:
            total_cds += 1
            t_s = lbls.index(1)
            
            # Move the true_starts append inside this condition
            if 1 in preds:
                p_s = preds.index(1)
                true_starts.append(t_s * 3)  # Tracks ONLY when a prediction exists
                pred_starts.append(p_s * 3)  # Tracks ONLY when a prediction exists
                if p_s == t_s: 
                    exact_hits += 1

    acc  = accuracy_score(all_t, all_p)
    prec = precision_score(all_t, all_p, zero_division=0)
    rec  = recall_score(all_t, all_p, zero_division=0)
    f1   = f1_score(all_t, all_p, zero_division=0)
    mcc  = matthews_corrcoef(all_t, all_p)
    mae  = mean_absolute_error(true_starts, pred_starts) if true_starts else 0
    exact_rate = exact_hits / total_cds if total_cds else 0

    # Explicitly enforce the label mapping to guarantee quadrant alignment
    cm_raw = confusion_matrix(all_t, all_p, labels=[0, 1])
    tn, fp, fn, tp = cm_raw.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal_acc = balanced_accuracy_score(all_t, all_p)

    return dict(label=label, accuracy=acc, precision=prec, recall=rec,
                f1=f1, mcc=mcc, mae=mae, exact_rate=exact_rate,
                specificity=specificity, balanced_accuracy=bal_acc,
                y_true=all_t, y_pred=all_p, y_scores=all_scores)
    
# ─────────────────────────────────────────────
# DIAGNOSTICS & PLOTTING
# ─────────────────────────────────────────────
def generate_aggregate_diagnostics_v3(tprs_dict, aucs_dict, mean_fpr, cms_dict):
    plt.style.use('seaborn-v0_8-whitegrid')
    # Expanded grid matrix to 8 rows to evaluate the new meta_router model
    fig, axes = plt.subplots(8, 2, figsize=(15, 40)) 
    fig.suptitle("Cross-Eukaryotic TIS Prediction Benchmark", 
                 fontsize=22, fontweight='bold', y=0.98)

    versions = ["naive", "baseline", "v1", "v2", "v3", "hmm_tokenized", "hmm_stride1", "meta_router"]
    colors = ['#000000', '#808080', '#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#1F77B4']
    cm_cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Reds', 'Oranges', 'YlGnBu', 'GnBu']
    
    for i, ver in enumerate(versions):
        ax_roc = axes[i, 0]
        ax_cm = axes[i, 1]
        
        current_tprs = tprs_dict.get(ver, [])
        
        # Check if we actually have data for this version
        if not current_tprs or len(current_tprs) == 0:
            ax_roc.set_title(f"{ver.upper()} (No Data)")
            ax_cm.set_title(f"{ver.upper()} (No Data)")
            continue

        # Ensure we are taking the mean across the correct axis
        mean_tpr = np.mean(current_tprs, axis=0)
        
        # Fix for the TypeError: Verify mean_tpr is an array before indexing
        if isinstance(mean_tpr, np.ndarray) and mean_tpr.ndim > 0:
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_tpr = np.std(current_tprs, axis=0)

            ax_roc.plot(mean_fpr, mean_tpr, color=colors[i], lw=3, label=f'Mean (AUC={mean_auc:.3f})')
            ax_roc.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), 
                                np.minimum(mean_tpr + std_tpr, 1), color=colors[i], alpha=0.2)
        
        ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax_roc.set_title(f"{ver.upper()} ROC Curve", fontsize=14, fontweight='bold')
        ax_roc.legend(loc="lower right")

        # Extract individual counts from the running database
        tn, fp, fn, tp = cms_dict[ver].ravel()
        
        # Re-arrange matrix array so that CDS (Positive) sits at the top/left index
        # Top-Left: TP, Top-Right: FN, Bottom-Left: FP, Bottom-Right: TN
        inverted_cm = np.array([[tp, fn], 
                                [fp, tn]])

        sns.heatmap(inverted_cm, annot=True, fmt='d', cmap=cm_cmaps[i], ax=ax_cm, cbar=False,
                    xticklabels=['CDS', 'Intergenic'], yticklabels=['CDS', 'Intergenic'],
                    annot_kws={"size": 18})
        ax_cm.set_title(f"{ver.upper()} Confusion Matrix (TP at Top-Left)", fontsize=14, fontweight='bold')
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("v3_comprehensive_comparison.png", dpi=300)
    plt.close()
    print("[✔] Saved -> v3_comprehensive_comparison.png")

def plot_resolution_sharpness():
    """Generates the IEEE Proof graph using a synthetic ideal sequence."""
    print("\nGenerating IEEE Sharpness Plot...")
    
    # Synthetic sequence: 100bp junk + 50bp Kozak + ATG + 147bp high-CUB gene
    # Using 'GAG' and 'CGC' to simulate strong coding potential
    upstream_junk = "TGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"
    kozak = "GCCGCCACC"
    atg = "ATG"
    downstream_gene = "GAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGC"
    
    test_seq = upstream_junk + kozak + atg + downstream_gene
    true_tis = len(upstream_junk) + len(kozak)
    
    scores = []
    positions = list(range(50, len(test_seq) - 93))
    
    for i in positions:
        upstream = test_seq[i-50:i]
        downstream = test_seq[i+3:i+93]
        
        if test_seq[i:i+3] != "ATG":
            scores.append(-20) # Frame failure baseline
            continue
            
        p_score = score_promoter(upstream)
        cub_score = score_coding_potential(downstream, codon_log_odds_map)
        scores.append(p_score + (cub_score * 2.0))

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 5))
    
    plt.plot(positions, scores, color='#C44E52', lw=2)
    plt.axvline(x=true_tis, color='black', linestyle='--', label=f'True TIS (bp {true_tis})')
    
    plt.annotate('Complete score failure\nat +1/-1 frame shift', 
                 xy=(true_tis+1, -15), xytext=(true_tis+15, -5),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

    plt.title("Explainable 1-bp Resolution via Codon Tokenization", fontsize=14, fontweight='bold')
    plt.xlabel("Genomic Position (bp)", fontsize=12)
    plt.ylabel("Contextual Prediction Score (PWM + CUB Log-Odds)", fontsize=12)
    plt.xlim([true_tis - 20, true_tis + 30])
    plt.ylim([-25, max(scores) + 5])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("ieee_resolution_sharpness.png", dpi=300)
    plt.close()
    print("[✔] Saved -> ieee_resolution_sharpness.png")

def testingFromjson(filepath="testing_sources.json"):
    testing_sources = load_training_sources(filepath)
    if not testing_sources: return

    # Check cache before evaluating
    records = get_cached_records(filepath, "testing_cache.pkl")

    # Initialize dictionaries for all 7 versions
    versions = ["naive", "baseline", "v1", "v2", "v3", "hmm_tokenized", "hmm_stride1", "meta_router"]
    all_metrics = {v: [] for v in versions}
    tprs = {v: [] for v in versions}
    aucs = {v: [] for v in versions}
    cms = {v: np.zeros((2, 2), dtype=int) for v in versions}
    
    mean_fpr = np.linspace(0, 1, 100)

    for source in testing_sources:
        species_name = source['name']
        if species_name not in records:
            continue
            
        print(f"\n>>> Evaluating Species: {species_name}")
        rec_test = records[species_name]

        test_data = prepare_test_data(rec_test)
        if not test_data: continue

        # Generate results for all decoders
        results = {
            "naive": evaluate(decoder_stride1_naive, "naive", test_data),
            "baseline": evaluate(decoder_baseline, "baseline", test_data),
            "v1": evaluate(decoder_v1, "v1", test_data),
            "v2": evaluate(decoder_v2, "v2", test_data),
            "v3": evaluate(decoder_v3, "v3", test_data),
            "hmm_tokenized": evaluate(decoder_hmm_tokenized, "hmm_tokenized", test_data),
            "hmm_stride1": evaluate(decoder_hmm_stride1, "hmm_stride1", test_data),
            "meta_router": evaluate(decoder_meta_router, "meta_router", test_data) # NEW
        }

        # Store results for each version
        for ver in versions:
            res = results[ver]
            fpr, tpr, _ = roc_curve(res["y_true"], res["y_scores"])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            
            tprs[ver].append(interp_tpr)
            aucs[ver].append(auc(fpr, tpr))
            cms[ver] += confusion_matrix(res["y_true"], res["y_pred"], labels=[0, 1])
            
            # Save metrics to list for CSV export
            clean_res = {k: v for k, v in res.items() if k not in ['y_true', 'y_pred', 'y_scores']}
            clean_res['species'] = species_name
            all_metrics[ver].append(clean_res)

    # Export all 5 CSVs
    for ver in versions:
        if all_metrics[ver]:
            pd.DataFrame(all_metrics[ver]).to_csv(f"all_species_results_{ver}.csv", index=False)
    
    print("\n[✔] All metrics saved to CSV.")
    generate_aggregate_diagnostics_v3(tprs, aucs, mean_fpr, cms)
    
# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ensemble Biological Gatekeeper Benchmark")
    parser.add_argument('--model_file', type=str, default='ensemble_model.pkl', help='Path to ensemble model')
    parser.add_argument('--test_json', type=str, default='testing_sources.json', help='Testing targets')
    parser.add_argument('--gen_sharpness', action='store_true', help='Generate IEEE Sharpness Plot')
    args = parser.parse_args()

    global meta_tree, clade_models
    
    # Seed the active globals just to prevent initialization errors
    global pwm_logo, codon_vocab, codon_log_odds_map, hmm_emissions_bg, hmm_emissions_cds

    if os.path.exists(args.model_file):
        print(f"Loading Ensemble Model from {args.model_file}...")
        with open(args.model_file, 'rb') as f:
            model_data = pickle.load(f)
            
        if 'meta_tree' not in model_data:
            print("[!] Old model structure detected. Re-training ensemble required.")
            model_data = train_model()
            with open(args.model_file, 'wb') as f: pickle.dump(model_data, f)
            
        meta_tree = model_data['meta_tree']
        clade_models = model_data['clade_models']
    else:
        print(f"Model file not found. Training Ensemble...")
        model_data = train_model()
        meta_tree = model_data['meta_tree']
        clade_models = model_data['clade_models']
        with open(args.model_file, 'wb') as f: pickle.dump(model_data, f)
        print(f"Ensemble Model saved to {args.model_file}.")

    # Pre-load fallback globals for standard decoders (v1, v2, v3)
    if clade_models:
        default_params = list(clade_models.values())[0]
        pwm_logo = default_params['pwm_logo']
        codon_vocab = default_params['codon_vocab']
        codon_log_odds_map = default_params['codon_log_odds']
        hmm_emissions_bg = default_params['hmm_emissions_bg']
        hmm_emissions_cds = default_params['hmm_emissions_cds']

    if args.gen_sharpness:
        plot_resolution_sharpness()
        return

    testingFromjson(args.test_json)

if __name__ == '__main__':
    main()