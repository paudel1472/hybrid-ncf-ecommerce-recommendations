# Hybrid E-Commerce Recommendation & Explainable AI (XAI) Framework

This repository delivers the core engineering infrastructure, model development, and benchmark evaluations for my MSc Dissertation. The project implements a high-performance **Hybrid Neural Collaborative Filtering (NCF)** framework in PyTorch designed to dramatically mitigate the classic data sparsity and user/item cold-start constraints dominant in modern e-commerce systems.

## 🌟 Framework Core Features
* **Statistically Representative Simulation Engine:** Replicates the data sparsity and cold-start mathematical bounds of massive public benchmarks.
* **Iterative Deep Learning Pipelines:** Scaled from linear Matrix Factorisation baselines to a fully structured contextual-attention neural architecture[cite: 1].
* **Explainable AI (XAI) Architecture:** Designed to seamlessly map data pipelines directly into post-hoc model explainability frameworks (SHAP and LIME)[cite: 1].
* **Evaluation Matrix:** Native pipelines tracking RMSE, NDCG@10, and Precision@10 across varying structural configurations[cite: 1].

---

## 🏗️ Architectural Evolution

The framework processes user-item relationships through four architectural baselines to isolate and prove iterative validation metrics:

1. **Matrix Factorisation (Baseline):** Classic linear collaborative modeling utilizing learnable user/item embedding layer dot products[cite: 1].
2. **Baseline NCF:** Introduces a non-linear Multi-Layer Perceptron (MLP) mapping layer instead of a basic inner product to capture complex user-item interaction dynamics[cite: 1].
3. **NCF + Contextual Attention Mechanism:** Integrates a localized attention layer that computes scalar weights for varying contextual features before feeding dense vectors into the MLP[cite: 1].
4. **Full Hybrid NCF Framework:** The final architecture proposed in the dissertation. It incorporates collaborative embeddings, contextual attention layers, and multimodal item feature fusions (simulated textual and visual item encodings)[cite: 1].

---

## 📊 Dataset & Reproducibility
To keep this repository self-contained, light, and immediately executable by reviewers without downloading hundreds of gigabytes of public data streams, **the system utilizes a procedural dataset simulation engine**[cite: 1]. 

The engine synthetically builds interaction matrices that replicate the concrete statistical realities of three standard industry benchmarks[cite: 1]:
* **Amazon Reviews** (Density: 1.8%, Cold-Start: 12%)[cite: 1]
* **MovieLens 20M** (Density: 2.5%, Cold-Start: 9%)[cite: 1]
* **Alibaba Tianchi** (Density: 1.2%, Cold-Start: 15%)[cite: 1]

**No external assets or downloads are required.** Running the notebook instantly populates these representative arrays locally[cite: 1].

---

## ⚙️ Environment Setup & Requirements
This framework is optimized for both CPU and CUDA-accelerated GPU pipelines[cite: 1].

### Installation
Clone this specific project module and run the requirements script:
```bash
git clone [https://github.com/paudel1472/hybrid-ncf-ecommerce-recommendations.git](https://github.com/paudel1472/hybrid-ncf-ecommerce-recommendations.git)
cd hybrid-ncf-ecommerce-recommendations
pip install -r requirements.txt