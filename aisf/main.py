#!/usr/bin/env python3

"""
Script: run_refusal_steering.py

This script demonstrates how to:
1. Load a Qwen model and corresponding Sparse Autoencoder (SAE) from Weights & Biases.
2. Extract "refusal directions" (harmful vs. harmless instructions) using top-down mean-difference
   and various SAE-based methods.
3. Apply these steering vectors (via PyTorch hooks) to generate text completions
   that are steered (or un-steered) toward a refusal style.

Author: [Your Name / Organization]
"""

import os
import logging
import json
import functools
import gc
from typing import List, Callable

import numpy as np
import torch
import wandb
import einops

# --- [Commented out: Typically, you'd install before running the script, rather than in-script] ---
# os.system("pip install transformer_lens")
# os.system("pip install wandb")
# os.system("pip install plotly")
# os.system("pip install sae_lens")
# os.system("pip install colorama")
# os.system("pip install vllm")
# os.system("pip install litellm")

# Transformer Lens & SAE Lens
from transformer_lens import HookedTransformer, utils
from sae_lens.sae import SAE

# For visualization/debugging if needed
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------------------------------------------------------------------
# Logger Setup
# --------------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# Global Device
# --------------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------------
# 1. Load Qwen Model and SAE
# --------------------------------------------------------------------------------
def load_qwen_and_sae(
    model_name: str,
    artifact_path: str,
    sae_download_path: str,
    estimated_norm_scaling_factor: float,
) -> (HookedTransformer, SAE):
    """
    Load a Qwen model via TransformerLens HookedTransformer.from_pretrained()
    and a sparse auto-encoder (SAE) from a W&B artifact.

    Args:
        model_name: The model alias used by TransformerLens (e.g. "qwen1.5-0.5b-chat").
        artifact_path: The Weights & Biases artifact path, e.g. "entity/project/artifact:v11".
        sae_download_path: Local path to download the artifact.
        estimated_norm_scaling_factor: A known scaling factor for the b_dec in the SAE
            (if needed to correct a known bug or mismatch).

    Returns:
        (model, sae) where model is a HookedTransformer and sae is the loaded SAE.
    """
    logger.info(f"Loading Qwen model: {model_name}")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=DEVICE,
        default_padding_side='left',
    )
    model.tokenizer.padding_side = 'left'
    model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Download SAE from W&B
    logger.info(f"Downloading SAE artifact from W&B: {artifact_path}")
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact.download(root=sae_download_path)
    sae = SAE.load_from_pretrained(sae_download_path, device=DEVICE)

    # Adjust for known bug or mismatch:
    sae.b_dec.data /= estimated_norm_scaling_factor

    # "Fold" decoder norms
    decoder_norms = sae.W_dec.data.norm(dim=-1, keepdim=True)
    sae.W_enc.data *= decoder_norms.T
    sae.b_enc.data *= decoder_norms.squeeze()
    sae.W_dec.data /= decoder_norms

    logger.info(f"SAE loaded with shapes: W_enc={sae.W_enc.shape}, W_dec={sae.W_dec.shape}")
    return model, sae


def extract_refusal_direction_bottom_up(
    model: HookedTransformer,
    layer: int,
    pos: int,
    harmful_batch: torch.Tensor,
    harmless_batch: torch.Tensor,
    sae: SAE,
) -> torch.Tensor:
    # 5. Optional: Compute SAE-based directions
    #    We just do it for the mean harmful/harmless single vectors here:
    harmful_acts, harmless_acts = extract_harmful_harmless_acts(model=model, layer=layer, pos=pos,
                                                                harmful_batch=harmful_batch, 
                                                                harmless_batch=harmless_batch)
    sae_dirs = compute_sae_directions(sae, harmful_acts, harmless_acts)
    logger.info(f"Computed {len(sae_dirs)} SAE-based directions: {list(sae_dirs.keys())}") 


def extract_harmful_harmless_acts(
    model: HookedTransformer,
    layer: int,
    pos: int,
    harmful_batch: torch.Tensor,
    harmless_batch: torch.Tensor,
) -> torch.Tensor:
    logger.info("Running model on harmful instructions to get residual stream...")
    _, harmful_cache = model.run_with_cache(
        harmful_batch,
        names_filter=utils.get_act_name('resid_pre', layer),
        stop_at_layer=layer + 1,
    )

    logger.info("Running model on harmless instructions to get residual stream...")
    _, harmless_cache = model.run_with_cache(
        harmless_batch,
        names_filter=utils.get_act_name('resid_pre', layer),
        stop_at_layer=layer + 1,
    )

    harmful_acts = harmful_cache['resid_pre', layer][:, pos, :]
    harmless_acts = harmless_cache['resid_pre', layer][:, pos, :]
    return harmful_acts, harmless_acts


def extract_refusal_direction_top_down(
    model: HookedTransformer,
    layer: int,
    pos: int,
    harmful_batch: torch.Tensor,
    harmless_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Computes a "refusal direction" as the difference of mean activations
    (harmful_mean - harmless_mean) at a certain layer + token pos in the residual stream.

    Args:
        model: HookedTransformer with a run_with_cache method.
        layer: The index of the layer from which to extract 'resid_pre' or 'resid_post'.
        pos: The token index (e.g. -1 means the last token).
        harmful_batch: A batch of token IDs for "harmful" instructions.
        harmless_batch: A batch of token IDs for "harmless" instructions.

    Returns:
        The refusal direction as a 1D torch.Tensor (dimension = d_model).
    """
    harmful_acts, harmless_acts = extract_harmful_harmless_acts(model=model, layer=layer, pos=pos,
                                                                harmful_batch=harmful_batch, 
                                                                harmless_batch=harmless_batch)
    harmful_mean_act = harmful_acts.mean(dim=0)
    harmless_mean_act = harmless_acts.mean(dim=0)

    chat_refusal_dir = harmful_mean_act - harmless_mean_act
    logger.info(f"Extracted refusal direction from layer={layer}, pos={pos}")
    return chat_refusal_dir


# --------------------------------------------------------------------------------
# 3. SAE-based "refusal direction" alternatives
# --------------------------------------------------------------------------------
def compute_sae_directions(
    sae: SAE,
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
) -> dict:
    """
    Demonstrates various ways to get "SAE-based" refusal directions from harmful/harmless
    mean residual activations.

    Returns a dict of named directions, e.g.:
    {
      "sae_m1": direction_m1,
      "sae_m2": direction_m2,
      ...
    }
    """
    directions = {}
    harmful_mean_act = harmful_acts.mean(dim=0)
    harmless_mean_act = harmless_acts.mean(dim=0)
    # "chat_refusal_dir_sae_m1": decode(harmful_mean) - decode(harmless_mean)
    enc_harmful_mean = sae.encode(harmful_mean_act)
    enc_harmless_mean = sae.encode(harmless_mean_act)

    # Option 1: decode each mean, subtract
    directions["sae_m1"] = sae.decode(enc_harmful_mean) - sae.decode(enc_harmless_mean)

    # Option 2: encode difference, decode
    harmful_minus_harmless = harmful_mean_act - harmless_mean_act
    directions["sae_m2"] = sae.decode(sae.encode(harmful_minus_harmless))

    # Option 3 (another variation)
    directions["sae_m3"] = sae.decode(sae.encode(harmful_acts).mean(dim=0)) - sae.decode(sae.encode(harmless_acts).mean(dim=0))
    directions["sae_m4"] = sae.decode(sae.encode(harmful_acts).mean(dim=0) - sae.encode(harmless_acts).mean(dim=0))
    return directions


# --------------------------------------------------------------------------------
# 4. Hook Functions for Steering
# --------------------------------------------------------------------------------
def steering_fn(
    activations: torch.Tensor,
    hook,
    direction: torch.Tensor,
    steering_vector: torch.Tensor,
    mode: str,
    steering_strength: float=1.,
    max_act: float=1.,
):
    """
    Example hook that modifies `activations` at every forward pass.

    mode = "subtract_projection":
       We project out the component of `activations` in the steering_vector direction.
       In other words, we push the model AWAY from that direction.
    mode = "add_projection":
       We push the model TOWARD that direction.
    """
    if mode == "subtract_projection":
        # Project `activations` onto `steering_vector` and subtract
        proj = einops.einsum(activations, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
        return activations - proj

    elif mode == "add_act_subtract_projection":
        # Project `activations` onto `steering_vector2` and subtract
        proj = einops.einsum(activations, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
        return activations - proj + max_act * steering_strength * steering_vector
    
    elif mode == "add":
        return activations + max_act * steering_strength * steering_vector
    
    else:
        return activations


def generate_with_hooks(
    model: HookedTransformer,
    toks: torch.Tensor,
    max_tokens_generated: int,
    fwd_hooks: list,
    temperature: float = 1.0,
) -> List[str]:
    """
    Simple generation function that iteratively calls the model with hooking,
    using greedy decode if temperature=1.0.

    Returns list of generated strings for each batch row.
    """
    all_toks = torch.zeros(
        (toks.shape[0], toks.shape[1] + max_tokens_generated),
        dtype=torch.long, device=toks.device
    )
    all_toks[:, :toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :toks.shape[1] + i])
            if temperature == 1.0:
                next_tokens = logits[:, -1, :].argmax(dim=-1)
            else:
                # Example: sample with temperature
                scaled_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        all_toks[:, toks.shape[1] + i] = next_tokens

    return model.tokenizer.batch_decode(
        all_toks[:, toks.shape[1]:], skip_special_tokens=True
    )


# --------------------------------------------------------------------------------
# 5. Main or Demo Function
# --------------------------------------------------------------------------------
def main():
    # 1. Configuration (tweak these as you wish)
    qwen_model_name = "qwen1.5-0.5b-chat"
    artifact_path = "ckkissane/qwen-500M-chat-lmsys-1m-anthropic/sae_qwen1.5-0.5b-chat_blocks.13.hook_resid_pre_32768:v11"
    sae_local_path = "./artifacts/qwen-500M-chat-lmsys-1m-anthropic"
    estimated_norm_scaling_factor = 1.5326804001319736
    layer_of_interest = 13
    pos_of_interest = -1

    logger.info("=== Loading model & SAE ===")
    model, sae = load_qwen_and_sae(
        model_name=qwen_model_name,
        artifact_path=artifact_path,
        sae_download_path=sae_local_path,
        estimated_norm_scaling_factor=estimated_norm_scaling_factor
    )

    # 2. Load example instructions
    harmful_inst_train, harmful_inst_test = get_harmful_instructions()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions()

    # 3. Tokenize some training instructions for both classes
    n_inst_train = 4  # just a small number for demonstration
    template = QWEN_CHAT_TEMPLATE
    tokenize_fn = functools.partial(tokenize_instructions, tokenizer=model.tokenizer, template=template)

    harmful_toks = tokenize_fn(harmful_inst_train[:n_inst_train])
    harmless_toks = tokenize_fn(harmless_inst_train[:n_inst_train])

    # 4. Compute a top-down "refusal direction"
    chat_refusal_dir = extract_refusal_direction_top_down(
        model=model,
        layer=layer_of_interest,
        pos=pos_of_interest,
        harmful_batch=harmful_toks,
        harmless_batch=harmless_toks
    )
    logger.info(f"Refusal direction norm: {chat_refusal_dir.norm().item():.4f}")

    # 5. Optional: Compute SAE-based directions
    sae_dirs = extract_refusal_direction_bottom_up(
        model=model,
        layer=layer_of_interest,
        pos=pos_of_interest,
        harmful_batch=harmful_toks,
        harmless_batch=harmless_toks,
        sae=sae
    )
    logger.info(f"Computed {len(sae_dirs)} SAE-based directions: {list(sae_dirs.keys())}")

    # 6. Demonstrate hooking with "chat_refusal_dir" in subtract mode
    #    i.e. push away from "harmful direction".
    instructions_to_test = harmful_inst_test[:2]
    test_toks = tokenize_fn(instructions_to_test)

    # Run steering with different modes
    steering_feature = 18316
    steering_vector = sae.W_dec[steering_feature].to(model.cfg.device)
    sae_steer = partial(
        steering_fn,
        direction=None,
        steering_vector=steering_vector,
        steering_strength=3.,
        max_act=1.407,
        mode="add"
    )
    top_down_steer = partial(
        steering_fn,
        direction=chat_refusal_dir.to(model.cfg.device)/chat_refusal_dir.norm(),
        steering_vector=None,
        mode="subtract_projection"
    )
    hybrid_steer = partial(
        steering_fn,
        direction=chat_refusal_dir.to(model.cfg.device)/chat_refusal_dir.norm(),
        steering_vector=steering_vector,
        steering_strength=2.,
        max_act=1.407,
        mode="add_act_subtract_projection"
    )

    sae_td_steers = {}
    for sae_dir_name, sae_dir in sae_dirs.items():
        sae_td_steers[sae_dir_name] = partial(
                steering_fn,
                direction=sae_dir.to(model.cfg.device)/sae_dir.norm(),
                steering_vector=None,
                mode="subtract_projection"
            )

    fwd_hooks = {'top_down': [(utils.get_act_name('resid_pre', layer_of_interest), top_down_steer)],
                 'hybrid': [(utils.get_act_name('resid_pre', layer_of_interest), hybrid_steer)],
                 'sae': [(utils.get_act_name('resid_pre', layer_of_interest), sae_steer)]}
    for sae_dir_name in sae_td_steers:
        fwd_hooks[sae_dir_name] = [(utils.get_act_name('resid_pre', layer_of_interest), sae_td_steers[sae_dir_name])]
    # Generate with no steering
    logger.info("=== Generating baseline completions (no steering) ===")
    results = {}
    results["baseline"] = generate_with_hooks(
        model=model,
        toks=test_toks,
        max_tokens_generated=64,
        fwd_hooks=[],  # No hooks
        temperature=0.0
    )

    # Generate with steering
    logger.info("=== Generating completions with different steering methods ===")
    for steering_name in fwd_hooks:
        logger.info(f"Steering method: {steering_name}")
        results[steering_name] = generate_with_hooks(
            model=model,
            toks=test_toks,
            max_tokens_generated=64,
            fwd_hooks=fwd_hooks[steering_name],
            temperature=0.0
        )

    # Show results
    print(f'Results: {results}')

    # Cleanup
    del harmful_inst_train, harmful_inst_test, harmless_inst_train, harmless_inst_test
    gc.collect()

    logger.info("Done!")


# --------------------------------------------------------------------------------
# 6. Entry Point
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
