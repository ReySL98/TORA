"""
This script implements the structural affinity evaluation and ranking protocol 
for the Task-Oriented Rank Adaptation (TORA) framework. It evaluates the 
topological compatibility between sequential domains before parameter initialization,
following the procedure described in Section 3.3 of the TORA methodology.
"""

import os
import torch
import pandas as pd
import numpy as np
from safetensors.torch import load_file
from sklearn.metrics.pairwise import cosine_similarity

# Import mapping directly from the formalized TORA tools
from tools import build_reference_space, compute_task_fingerprint

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
# Memory pool containing the N-1 stored expert adapters (trained to convergence, 20 epochs)
EXPERT_ROOT = r"C:\Users\reisy\OneDrive\Documentos\TORA\exp_17\lora_weights"

# Incoming task adapters (partial training, 1 epoch) to compute the geometric fingerprint
PROBE_ROOT = r"C:\Users\reisy\OneDrive\Documentos\TORA\exp_19\lora_weights"

TASKS = [
    "agnews", "amazon", "boolq", "cb", "copa", "dbpedia",
    "imdb", "mnli", "multirc", "qqp", "rte", "sst2",
    "wic", "yahoo", "yelp"
]

# =============================================================================
# ADAPTER LOADING
# =============================================================================
def load_adapter(path, task_key):
    """
    Load the LoRA adapter safetensors into a state dictionary.
    Excludes classifier weights to ensure the SVD space is built exclusively
    on the structural projections.
    """
    file_path = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(file_path):
        print(f"   [Warning] File not found: {file_path}")
        return None
    state_dict = load_file(file_path, device="cpu")
    return {k: v for k, v in state_dict.items() if "classifier" not in k}

# =============================================================================
# MAIN ROUTING PROTOCOL SIMULATION
# =============================================================================
print("Initiating TORA Structural Affinity Evaluation (Partial Adapter vs. Expert Memory)\n")

affinity_ranking_matrix = []

for target_task in TASKS:
    print(f"--------------------------------------------------")
    print(f"Evaluating Incoming Task (T_new): {target_task.upper()}")

    # ------------------------------------------------------------------
    # STEP 1: Load the N-1 stored experts from the memory pool
    # Represents the established expert representations {e_1, e_2, ..., e_{N-1}}
    # ------------------------------------------------------------------
    expert_tasks = [t for t in TASKS if t != target_task]

    expert_pool = {}
    for expert in expert_tasks:
        expert_path = os.path.join(EXPERT_ROOT, f"{expert}_lora_ep20")
        sd = load_adapter(expert_path, expert)
        if sd:
            for k, v in sd.items():
                if k not in expert_pool:
                    expert_pool[k] = {}
                expert_pool[k][expert] = v

    if not expert_pool:
        print(f"   [Error] No experts loaded for {target_task}. Skipping...")
        continue

    # ------------------------------------------------------------------
    # STEP 2: SVD Space Construction
    # Build the shared mathematical space from historical data [Eq. 1-2]
    # ------------------------------------------------------------------
    print(f"   [Router] Constructing shared SVD reference space from {len(expert_tasks)} experts...")
    reference_basis = build_reference_space(expert_pool, vec_operator=True)

    # Determine k components satisfying the 60% variance criterion [Eq. 3]
    S2 = reference_basis[list(reference_basis.keys())[0]]["S2"]
    total_variance = torch.sum(S2)
    variance_ratio = S2 / total_variance
    cumulative_variance = torch.cumsum(variance_ratio, dim=0)
    k_components = (cumulative_variance >= 0.60).nonzero(as_tuple=True)[0][0].item() + 1
    print(f"   [Router] Retained principal components (k): {k_components}")

    # ------------------------------------------------------------------
    # STEP 3: Expert Fingerprint Computation
    # Project all N-1 experts onto the reference space yielding \phi_i [Eq. 4-5]
    # ------------------------------------------------------------------
    expert_fingerprints = {}
    for expert in expert_tasks:
        W_expert = {k: v[expert] for k, v in expert_pool.items() if expert in v}
        phi_expert = compute_task_fingerprint(reference_basis, W_expert, num_components=k_components)
        expert_fingerprints[expert] = torch.cat([v.flatten() for k, v in phi_expert.items() if "loadings" in k])

    # ------------------------------------------------------------------
    # STEP 4: Initial Training Simulation
    # Load the partial adapter weights for the incoming task (1 epoch)
    # ------------------------------------------------------------------
    probe_path = os.path.join(PROBE_ROOT, f"{target_task}_lora_ep1")
    W_new = load_adapter(probe_path, target_task)
    if not W_new:
        print(f"   [Error] Partial adapter not found for {target_task}. Skipping...")
        continue

    # ------------------------------------------------------------------
    # STEP 5: Incoming Task Fingerprint Generation
    # Project the incoming task onto the shared SVD space yielding \phi_{new}
    # ------------------------------------------------------------------
    phi_new = compute_task_fingerprint(reference_basis, W_new, num_components=k_components)
    phi_new_vec = torch.cat([v.flatten() for k, v in phi_new.items() if "loadings" in k])

    # ------------------------------------------------------------------
    # STEP 6: Similarity Ranking
    # Compute the cosine similarity and angular distance \theta_i [Eq. 6]
    # ------------------------------------------------------------------
    phi_new_np = phi_new_vec.detach().cpu().numpy().reshape(1, -1)

    local_affinities = []
    for expert, phi_expert in expert_fingerprints.items():
        phi_expert_np = phi_expert.detach().cpu().numpy().reshape(1, -1)
        sim = cosine_similarity(phi_new_np, phi_expert_np)[0][0]
        local_affinities.append((expert.upper(), round(float(sim), 4)))

    # Sort in ascending order of \theta_i (descending order of cosine similarity) [Eq. 7]
    local_affinities.sort(key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # STEP 7: Record the Ranking Matrix
    # ------------------------------------------------------------------
    row_data = {"Incoming_Task": target_task.upper()}
    for i, (exp, sim) in enumerate(local_affinities, start=1):
        # Calculate angular distance \theta_i
        ang = round(float(np.degrees(np.arccos(np.clip(sim, -1.0, 1.0)))), 4)
        row_data[f"Rank_{i}_Expert"] = exp
        row_data[f"Rank_{i}_CosineSim"] = sim
        row_data[f"Rank_{i}_Theta"] = ang

    affinity_ranking_matrix.append(row_data)
    print(f"   [Router] Top-3 Affinities: {local_affinities[:3]}")

# =============================================================================
# EXPORT RESULTS
# =============================================================================
df_matrix = pd.DataFrame(affinity_ranking_matrix)
output_path = "structural_affinity_matrix.csv"
df_matrix.to_csv(output_path, index=False)

print("\n" + "="*70)
print(f"Matrix successfully exported to: {output_path}")
print("="*70)
print("\nTop-1 Expert routing analysis per task:")
for _, row in df_matrix.iterrows():
    print(f"  {row['Incoming_Task']:10s} → Top-1: {row['Rank_1_Expert']:10s} (sim={row['Rank_1_CosineSim']:.4f}, \u03b8={row['Rank_1_Theta']:.2f}\u00b0)")