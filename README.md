# LGBA

LGBA (Language-Graph Bidirectional Attention) is a multimodal molecular learning framework for polyphenol anti-inflammatory activity prediction. It combines SMILES sequence representation, molecular graph representation, and bidirectional cross-modal attention for downstream classification, regression, prediction, and interpretation tasks.

## Repository Layout

```text
src/lgba/
  pretraining/
    smiles_awd_lstm/      # AWD-LSTM SMILES language-model pretraining
    dmpnn_graph/          # D-MPNN graph contrastive pretraining
  model/                  # LGBA fusion model, training, and prediction
  explain/                # interpretation utilities
scripts/
  data_curation/          # data curation workflow scripts
configs/                  # example configuration files
models/                   # pretrained encoder files and vocabularies
```

## Installation

Create a Python environment, then install the package and dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

RDKit is recommended through conda:

```bash
conda install -c conda-forge rdkit
```

## Main Components

- SMILES AWD-LSTM pretraining
- D-MPNN graph contrastive pretraining
- Multimodal classification and regression training
- Molecular property prediction
- Atom-level and group-level interpretation utilities
- Data curation scripts

## Pretrained Files

The `models/` directory contains pretrained LGBA encoder files and vocabularies used by the training and prediction scripts.

## Citation

If you use this repository, please cite the associated LGBA manuscript.
