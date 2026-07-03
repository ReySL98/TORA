import re
from typing import Dict

import torch
import torch.nn as nn


def rename_state_dict_key(text: str, substring: str, replacement: str) -> str:
    """
    Map standard LoRA weight keys into the TORA geometric fingerprint namespace.

    This aligns the adapter weight names with the projection coefficients \lambda^{(l,p)} 
    derived in Eq. (4).

    Args:
        text: The original string.
        substring: The substring to find and replace from.
        replacement: The string to replace with.

    Returns:
        The modified string with the replacement applied.
    """
    pattern = re.compile(re.escape(substring) + r".*", re.DOTALL)
    return re.sub(pattern, replacement, text)


def compute_task_fingerprint(
    reference_basis: Dict[str, Dict[str, torch.Tensor]],
    W_new: Dict[str, torch.Tensor],
    num_components: int,
    loadings: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Generate the geometric task fingerprint (\phi) by projecting the incoming LoRA weights 
    onto the shared SVD subspace.

    As described in Section 3.2, this function characterizes a task in the reference space 
    to evaluate its structural affinity before routing. It implements Eq. (4) and (5) 
    by mapping the flattened adaptation matrices onto the k most relevant singular vectors, 
    compressing the high-dimensional parameter space into an exact 144-dimensional coordinate.
    This compact signature is subsequently used to decide between Boosting or Shielding.

    Args:
        reference_basis: Dictionary containing the unified reference space for each
            layer, with keys 'U' (left singular vectors U_k^(l,p)) and 'S2'
            (squared singular values), extracted from historical task representations.
        W_new: State dictionary of the source LoRA (new incoming task or stored expert)
            used to compute the geometric fingerprint.
        num_components: Number of top singular vectors (k) to retain, satisfying
            the 60% explained-variance criterion (Eq. 3).
        loadings: Whether to compute the projection coefficients \lambda (default: True).

    Returns:
        State dictionary containing the task fingerprint (\lambda^{(l,p)}) per layer, encoding 
        the absolute position of the task in the shared SVD space.
    """
    phi_sd = {}
    for k in W_new.keys():
        if k in reference_basis:
            U_k = nn.Parameter(reference_basis[k]["U"][:, :num_components]).contiguous().cpu()
            if loadings:
                W_vec = W_new[k].reshape(-1, 1).to(torch.float32).cpu()
                lambda_lp = nn.Parameter(torch.mm(U_k.t(), W_vec).squeeze(dim=1))

                if "lora_A" in k:
                    new_key = rename_state_dict_key(k, "lora_A", "fingerprint_A.loadings")
                else:
                    new_key = rename_state_dict_key(k, "lora_B", "fingerprint_B.loadings")

                phi_sd.update({new_key: lambda_lp})
    return phi_sd


def build_reference_space(
    expert_pool: Dict[str, Dict[str, torch.Tensor]],
    vec_operator: bool,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Construct the shared mathematical reference space from the historical task memory pool.

    Following Section 3.1 of the TORA framework, this function extracts the LORA weights of 
    all N-1 stored experts, flattens them, and stacks them into the consolidated variability 
    matrix M^(l,p). This matrix captures the complete spatial variance of the model's 
    history, establishing the foundation to extract shared principal components for dynamic routing.

    Args:
        expert_pool: Dictionary of combined LORA weights organized by layer, representing
            the N-1 established expert representations in the memory pool.
        vec_operator: If True, applies the vec(.) operator to flatten weight matrices 
            into column vectors before stacking (Eq. 1).

    Returns:
        Dictionary mapping layer keys to the extracted reference-space containing:
        - 'U': Left singular vectors U^(l,p) defining the geometric rotation.
        - 'S2': Singular values quantifying the magnitude of variance.
    """
    phi_space = {}
    total_layers = len(expert_pool.keys())
    print(f"\n Initializing SVD for {total_layers} layers/keys...")

    for i, layer_key in enumerate(expert_pool.keys()):
        print(f"   [{(i+1)}/{total_layers}] Analyzing: {layer_key}...", end=" ", flush=True)

        tensor_list = []
        for expert_name in expert_pool[layer_key].keys():
            tensor = expert_pool[layer_key][expert_name]

            if vec_operator:
                tensor = tensor.reshape((tensor.shape[0] * tensor.shape[1], 1))

            if tensor.shape[0] < tensor.shape[1]:
                tensor = tensor.t()
            tensor_list.append(tensor)

        M_lp = torch.cat(tensor_list, dim=1).to(torch.float32)

        try:
            svd_basis = svd_reference_space(M_lp)
            phi_space.update({layer_key: svd_basis})
            print(" OK")
        except Exception as e:
            print(f" ERROR: {e}")

    print(f"\n SVD completed for all layers.\n")
    return phi_space


def svd_reference_space(M_lp: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Apply Singular Value Decomposition (SVD) to the row-centered variability matrix 
    to extract the unified reference space.

    This implements Eq. (2) by obtaining the left singular vectors U^(l,p) that form 
    the orthonormal basis of the reference space. It computes the squared 
    singular values to quantify the magnitude of variance along each principal direction, 
    which is required to select the top-k components satisfying the 60% total variance 
    threshold (Eq. 3).

    Args:
        M_lp: The consolidated memory bank matrix M^(l,p) aggregating the parametric 
            signatures of all previously learned domains.

    Returns:
        Dictionary containing the core semantic directions:
        - 'U': Left singular vectors U^(l,p) as columns.
        - 'S2': Squared singular values.
    """
    M_lp = M_lp.to(torch.float32)
    mean = M_lp.mean(axis=1, keepdim=True)
    M_lp = M_lp - mean

    device = "cuda" if torch.cuda.is_available() else "cpu"
    M_lp = M_lp.to(device)
    M_lp = torch.nan_to_num(M_lp, nan=0.0, posinf=0.0, neginf=0.0)

    U, S, Vh = torch.linalg.svd(M_lp, full_matrices=False)

    U = U.cpu().real
    S2 = (S ** 2).cpu().real

    return {"U": U, "S2": S2}