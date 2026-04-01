# From Single-Model Explanations to Rashomon-Based Explanation Uncertainty in Time Series

**ENFIELD Project — XAI & AutoML Integration**

---

## Page 1: XAI for Time Series — Current Landscape

XAI methods from images/tabular data **do not transfer well** to time series due to temporal dependence, lagged relationships, baseline/masking ambiguity, and cross-variable correlations. The literature is growing but fragmented — no single standard exists.

### Four Main XAI Families

| Family | Question Answered | Key Methods |
|--------|-------------------|-------------|
| **Attribution / Saliency** | Which time point or region influenced the prediction most? | SHAP, TimeSHAP, LIME, Saliency, Integrated Gradients, Grad-CAM, DeepLIFT |
| **Counterfactual / Example-based** | How should the series differ for the prediction to change? | Native Guide, ForecastCF, prototype selection |
| **Intrinsic / Architecture-level** | Can the model itself be interpretable? | XCM, MTEX-CNN, TSViz, TSXplain (natural language) |
| **Subsequence-based** | Which short pattern is discriminative? | Shapelets, Implet, Coh-Implet, timeXplain |

**Toolkits:** TSInterpret (unified Python API for TS interpretability), Captum (PyTorch)
**Evaluation:** AMEE (multi-referee explainer selection), WAE (attribution quality metric for forecasting)

---

## Page 2: Attribution Visualizations & Implet

### Visualization Challenge (Schlegel & Keim, 2021)
Heatmaps on time series are **hard to interpret** even for experts. User studies show explanation-by-example outperforms attribution heatmaps. Alternatives like pipes (timeXplain) and additive bar charts (MTSeer) improve readability but remain limited.

### Implet (Meng et al., 2025) — Bridging Attribution and Subsequence
Extracts **consecutive high-attribution segments** as post-hoc, model-aware explanations:

```
Time Series -> Model -> Attribution -> Implet Extraction -> Coh-Implet Clustering
                                        (segment-level)      (group-level via DTW)
```

- Best with **Saliency, Input x Gradient, DeepLIFT**
- Coh-Implet: first cohort explanation for time series (clusters similar Implets across samples)
- Faithful: removing Implets causes significantly larger accuracy drops than random removal

---

## Page 3: The Gap — Why Single-Model Explanations Fall Short

All existing methods share one **fundamental limitation**: they explain a **single trained model**.

```
Model A (acc: 95.2%)  -->  "Peak at t=50 matters"
Model B (acc: 95.1%)  -->  "Slope at t=30-40 matters"      --> Which one to trust?
Model C (acc: 95.3%)  -->  "Valley at t=70 matters"
```

This is the **Rashomon Effect**: multiple near-optimal models produce different yet equally valid explanations. For time series, temporal dependence and model underspecification make this especially problematic.

| Single-Model XAI | Rashomon-Based XAI |
|------------------|--------------------|
| One explanation, no uncertainty | Multiple explanations, agreement = confidence |
| May reflect spurious patterns | Captures the space of valid explanations |
| "This matters" | "This matters **consistently** (or not)" |

---

## Page 4: Our Framework — AutoML + Rashomon Explanation Sets

```
┌──────────────────────────────────────────────────┐
│  STAGE 1: AutoML (H2O / AutoGluon)               │
│  Train diverse models -> Ranked leaderboard       │
├──────────────────────────────────────────────────┤
│  STAGE 2: Rashomon Set                            │
│  Select top-k models within epsilon of best       │
├──────────────────────────────────────────────────┤
│  STAGE 3: Multi-Model Explanations                │
│  SHAP / Saliency / Implet for EACH model          │
├──────────────────────────────────────────────────┤
│  STAGE 4: Aggregation & Uncertainty               │
│  Measure agreement across explanation set          │
│  HIGH agreement = trustworthy explanation          │
│  LOW agreement  = flag for human review            │
├──────────────────────────────────────────────────┤
│  STAGE 5: Uncertainty-Aware Output                │
│  Consensus explanation + uncertainty map           │
└──────────────────────────────────────────────────┘
```

**Why AutoML?** It naturally explores diverse architectures (GBM, XGBoost, DL, ensembles) and produces a leaderboard — the top-k models become the Rashomon set at **zero additional cost**.

---

## Page 5: Summary

### What Exists vs. What We Add

| Existing Literature | Our Contribution |
|--------------------|-----------------|
| Attribution, counterfactual, intrinsic, subsequence methods | **Rashomon-based explanation sets** across multiple near-optimal models |
| Single-model explanations | **Explanation uncertainty** as epistemic signal |
| Manual model selection | **AutoML** (H2O + AutoGluon) as natural Rashomon set source |
| "What is important" | "How **stable** is this explanation across equally valid models" |

> **Key message:** Existing XAI tells you what one model learned. Our framework reveals how stable those explanations are — turning model disagreement into a practical uncertainty signal via AutoML.

### References
- Schlegel & Keim (2021). *Time Series Model Attribution Visualizations as Explanations.*
- Meng, Kan et al. (2025). *Implet: A Post-hoc Subsequence Explainer for Time Series Models.* IEEE ICDMW
- TimeSHAP, XCM, TSInterpret, Native Guide, ForecastCF, AMEE, WAE
