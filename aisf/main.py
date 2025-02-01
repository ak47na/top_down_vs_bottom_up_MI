"""
This script:
1. Loads a Qwen model and corresponding Sparse Autoencoder (SAE) from Weights & Biases.
2. Extracts "refusal directions" (harmful vs. harmless instructions) using top-down mean-difference
   and bottom-up steering: various SAE-based methods.
3. Applies these steering vectors (via PyTorch hooks) to generate text completions
   that are steered (or un-steered) toward a refusal style.
"""

import pickle
import argparse
import os
import logging
import json
from functools import partial
import gc
from typing import List, Callable
from utils import get_harmful_instructions, get_harmless_instructions, QWEN_CHAT_TEMPLATE, tokenize_instructions

import numpy as np
import torch
import wandb
import einops

from transformer_lens import HookedTransformer, utils
from sae_lens.sae import SAE

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
# Load Qwen Model and SAE
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
) -> dict:
    harmful_acts, harmless_acts = extract_harmful_harmless_acts(model=model, layer=layer, pos=pos,
                                                                harmful_batch=harmful_batch, 
                                                                harmless_batch=harmless_batch)
    sae_dirs = compute_sae_directions(sae, harmful_acts, harmless_acts)
    return sae_dirs


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
    del harmful_cache, harmless_cache
    torch.cuda.empty_cache()
    gc.collect()

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
    del harmful_mean_act, harmless_mean_act
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"Extracted refusal direction from layer={layer}, pos={pos}")
    return chat_refusal_dir


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
    # directions["sae_m1"] = sae.decode(enc_harmful_mean) - sae.decode(enc_harmless_mean)

    # Option 2: encode difference, decode
    # harmful_minus_harmless = harmful_mean_act - harmless_mean_act
    #directions["sae_m2"] = sae.decode(sae.encode(harmful_minus_harmless))

    # Option 3 (another variation)
    directions["sae_m3"] = sae.decode(sae.encode(harmful_acts).mean(dim=0)) - sae.decode(sae.encode(harmless_acts).mean(dim=0))
    #directions["sae_m4"] = sae.decode(sae.encode(harmful_acts).mean(dim=0) - sae.encode(harmless_acts).mean(dim=0))

    # Free memory after computing SAE directions
    del harmful_mean_act, harmless_mean_act, enc_harmful_mean, enc_harmless_mean
    torch.cuda.empty_cache()
    gc.collect()
    return directions


# --------------------------------------------------------------------------------
# Hook Functions for Steering
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

    with torch.no_grad():  # Disable gradient tracking
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
# Main or Demo Function
# --------------------------------------------------------------------------------
def main(wandb_api_key: str = None):
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.login(key=wandb_api_key)
    # 1. Configuration: tweak to reproduce for other models or associated SAEs
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

    # 2. Load training harmful/harmless instructions
    harmful_inst_train, harmful_inst_test = get_harmful_instructions()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions()

    # 3. Tokenize a batch of training instructions for both classes
    n_inst_train = 128
    template = QWEN_CHAT_TEMPLATE
    tokenize_fn = partial(tokenize_instructions, tokenizer=model.tokenizer, template=template)

    harmful_toks = tokenize_fn(instructions=harmful_inst_train[:n_inst_train])
    harmless_toks = tokenize_fn(instructions=harmless_inst_train[:n_inst_train])

    del harmful_inst_train, harmless_inst_train
    gc.collect()

    # 4. Compute "refusal direction" top-down
    chat_refusal_dir = extract_refusal_direction_top_down(
        model=model,
        layer=layer_of_interest,
        pos=pos_of_interest,
        harmful_batch=harmful_toks,
        harmless_batch=harmless_toks
    )
    logger.info(f"Refusal direction norm: {chat_refusal_dir.norm().item():.4f}")

    # 5. Compute SAE-based directions
    sae_dirs = extract_refusal_direction_bottom_up(
        model=model,
        layer=layer_of_interest,
        pos=pos_of_interest,
        harmful_batch=harmful_toks,
        harmless_batch=harmless_toks,
        sae=sae
    )
    logger.info(f"Computed {len(sae_dirs)} SAE-based directions: {list(sae_dirs.keys())}")

    del harmful_toks, harmless_toks
    torch.cuda.empty_cache()
    gc.collect()

    N_INST_TEST = min(len(harmful_inst_test), len(harmless_inst_test))
    print(f'Num inst: {len(harmful_inst_test), len(harmless_inst_test)}')
    test_toks = {}
    test_toks["harmful"] = tokenize_fn(instructions=harmful_inst_test[:N_INST_TEST])
    test_toks["harmless"] = tokenize_fn(instructions=harmless_inst_test[:N_INST_TEST])

    # 6. Run steering with different modes
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
    # TODO: find negative feature
    # sae_neg_steer = partial(
    #     steering_fn,
    #     direction=None,
    #     steering_vector=steering_vector,
    #     steering_strength=5.,
    #     max_act=1.407,
    #     mode="add"
    # )
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
    intervention_layers = list(range(model.cfg.n_layers)) # all layers
    fwd_hooks = {'top_down': [(utils.get_act_name(act_name, l), top_down_steer) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']],
                 'hybrid': [(utils.get_act_name(act_name, layer_of_interest), hybrid_steer) for act_name in ['resid_pre', 'resid_mid', 'resid_post']],
                 'hybrid2': [(utils.get_act_name('resid_pre', layer_of_interest), hybrid_steer)],
                 'sae': [(utils.get_act_name(act_name, layer_of_interest), sae_steer) for act_name in ['resid_pre', 'resid_mid', 'resid_post']],
                #  'sae_neg': [(utils.get_act_name(act_name, layer_of_interest), sae_neg_steer) for act_name in ['resid_pre', 'resid_mid', 'resid_post']]
                }
    for l in intervention_layers:
        if intervention_layers != layer_of_interest:
            fwd_hooks["hybrid2"].extend([(utils.get_act_name(act_name, l), top_down_steer) for act_name in ['resid_pre', 'resid_mid', 'resid_post']])
    
    for sae_dir_name in sae_td_steers:
        fwd_hooks[sae_dir_name] = [(utils.get_act_name(act_name, l), sae_td_steers[sae_dir_name]) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]
    # 7. Generate with no steering
    logger.info("=== Generating baseline completions (no steering) ===")
    results = {'harmful': {}, 'harmless': {}}
    for test_toks_name in test_toks:
        results[test_toks_name]["baseline"] = generate_with_hooks(
            model=model,
            toks=test_toks[test_toks_name],
            max_tokens_generated=64,
            fwd_hooks=[],  # No hooks
            temperature=1.0
        )

    # 8. Generate with steering
    logger.info("=== Generating completions with different steering methods ===")
    for steering_name in fwd_hooks:
        logger.info(f"Steering method: {steering_name}")
        for test_toks_name in test_toks:
            t = 1.0
            if steering_name == "sae_m1" or steering_name == "sae_m2" or steering_name == "sae_m4":
                t = 0.7
            results[test_toks_name][steering_name] = generate_with_hooks(
                model=model,
                toks=test_toks[test_toks_name],
                max_tokens_generated=64,
                fwd_hooks=fwd_hooks[steering_name],
                temperature=t
            )
    # 9. Save results
    with open("./results.pkl", "wb") as f:
        pickle.dump(results, f)
    # 10. Save completions
    completions = {steering_name: [] for steering_name in fwd_hooks}
    del fwd_hooks
    for i in range(N_INST_TEST):
        for prompt_category in ["harmless", "harmful"]:
            for steering_name in results[prompt_category]:
                completions[steering_name].append({"response": results[prompt_category][steering_name][i],
                                                    "prompt": harmful_inst_test[i] if prompt_category == "harmful" else harmless_inst_test[i],
                                                    "category": prompt_category})
    with open("./completions.pkl", "wb") as f:
        pickle.dump(completions, f)
    # 11. Show results
    print(f'Results: {results}')

    # 12. Cleanup
    del results, harmful_inst_test, harmless_inst_test
    gc.collect()

    logger.info("Done!")
    return completions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run refusal steering on Qwen model.")
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        help="Optional: Weights & Biases API key (set as env variable if provided).",
        default=None,
    )
    
    args = parser.parse_args()
    main(wandb_api_key=args.wandb_api_key)