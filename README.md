# Steering LLM Behavior: Top-Down vs. Bottom-Up Approaches

## **Overview**
This project is developed as part of the [AI Safety Fundamentals](https://aisafetyfundamentals.com/) Alignment course. It explores **behavioral steering** in Language Models (LMs), comparing **top-down** and **bottom-up** approaches for modifying model outputs. Specifically, the focus is on inducing/bypassing refusal using either:
- **Top-Down Steering**: Identifying a **refusal direction** in the activation space using contrastive activations.
- **Bottom-Up Steering**: Using **Sparse Autoencoders (SAEs)** to either reconstruct the previous refusal direction, or find and steer safety related dictionary features.

For a detailed report, please see this [google doc](https://docs.google.com/document/d/1tlKygNUuEK2-P6h6VK6Uh8_qYwvQ_rDjADKueiu_z30/edit?tab=t.0), which precedes a LessWrong post.

## **Project Goals**
- Implement different **top-down** and **bottom-up** steering methods.
- Evaluate strengths and weaknesses.
- Propose a **hybrid approach** that optimally combines both methods.

## **Methods**

### **1. Top-Down Steering**
The **refusal vector** is extracted by computing the mean activation difference between harmful and harmless instructions as proposed in [Arditi et al](https://arxiv.org/abs/2406.11717).


### **2. Bottom-Up Steering: Sparse Autoencoder (SAE) Based Approach**
SAEs are trained to encode model activations into a **latent representation**, which allows **identifying functional features** responsible for certain behaviors.

**Steps:**
1. Select **middle layers** of the model for intervention.
2. Identify **SAE latents** that are active in harmful vs. harmless cases.
3. Modify **SAE activations** (e.g., amplifying, flipping signs) to influence model behavior.
4. Reconstruct activations and evaluate refusal performance.

**Alternative**: Reconstruct the refusal direction by encoding harmful and harmless prompts, decoding the mean activations and using their difference for steering in activations' space.

---

## **Setup & Installation**

`pip install -r requirements.txt`

## **Reproduce results**

`python main.py --wandb_api_key <your-wandb-api-key>`

Completions for harmful and harmless prompt for all steering approaches are stored in `./completions.pkl`
