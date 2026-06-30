# Simple Deep-Learning Meeting Demonstration

## Run order

1. `10_PZT_Physics_Feature_MLP_Damage_Classifier.ipynb`
2. `11_LDV_PZT_Contrastive_Cross_Modal_Alignment.ipynb`
3. `12_AI_Assisted_RAPID_and_Meeting_Figures.ipynb`

## Setup

Edit `DATASET_ROOT` in `dl_settings.py`.

```bash
conda create -n paper5dl python=3.11 -y
conda activate paper5dl
pip install -r requirements.txt
jupyter notebook
```

## Scope

These notebooks are for a next-week proof-of-concept meeting demonstration.

- The PZT MLP classifies baseline versus magnet path features.
- The cross-modal model aligns healthy-state LDV virtual receiver features and physical PZT features.
- The AI-assisted RAPID notebook converts path probabilities into a magnet localization map.

Do not present these results as generalization to unseen damage positions or specimens.
