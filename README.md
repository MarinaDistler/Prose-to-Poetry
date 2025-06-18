# Prose-to-Poetry

This repository provides tools and datasets for training models that convert Russian prose into structured poetry with meter and rhyme control.

## Installation

```bash
pip install -r requirements.txt
apt-get install espeak -y
export WB_TOKEN="your_token"
````

## Setup External Files

Download additional required files from [this Google Drive folder](https://drive.google.com/drive/folders/1oIEM5_UuK-5phD5LtJqCPnSQ5CVQiOoM?usp=sharing) and place them in the following locations:

```bash
cp verslibre-files/word2lemma.pkl external_code/verslibre/models/word2lemma.pkl
cp verslibre-files/accents.pkl external_code/verslibre/tmp/accents.pkl

mkdir -p external_code/verslibre/tmp/stress_model
cp verslibre-files/stress_model/config.json external_code/verslibre/tmp/stress_model/config.json
cp verslibre-files/stress_model/pytorch_model.pth external_code/verslibre/tmp/stress_model/pytorch_model.pth
```

## Dataset Preparation and Baseline

Folder `dataset-creation` includes notebooks for creation of datasets:

* `create-prose-dataset.ipynb`: creates a prose test dataset.
* `create-poetry-dataset.ipynb`: extracts quatrains with annotations (meter, rhyme scheme, stress markup, rhyme markup, combined markup). Also splits to test, train and pretrain parts.
* `create-prose-poetry-dataset.ipynb`: prepares prose–poetry pairs for training/testing and evaluates Gemini and GigaChat on test prose to create a baseline.

### `dataset/` contents:

| File/Folders          | Description                              |
| --------------------- | ---------------------------------------- |
| `all_poems.csv`       | Poems annotated with meter               |
| `all_stanzas.csv`     | Quatrains annotated with meter and rhyme |
| `prosa_test_text.csv` | Prose examples for evaluation            |
| `testset.csv`         | Prose–poetry test set                    |
| `trainset.csv`        | Prose–poetry training set                |
| `trainset_pretrain/`  | Poetry-only data for pretraining         |

## External Code

This repo includes a modified version of [`verslibre`](https://github.com/Koziev/verslibre) under `external_code/verslibre`, used for stress markup and annotation.

## Model Code (`prose-to-poetry/`)

### Training

```bash
# Pretraining on poetry generation
python3 prose-to-poetry/train.py --pretrain --model=qwen --save_steps=5000 \
  --train_dataset=dataset/trainset_pretrain --epochs=2 --log_steps=200 \
  --markup=rhyme_markup --warmup_steps=320 --lr=2e-5

# Fine-tuning on prose-to-poetry
python3 prose-to-poetry/train.py --model=qwen --save_steps=2000 \
  --from_pretrain=output/qwen-05-22-17-18-pretrain/checkpoint-10738 \
  --epochs=2 --log_steps=200 --markup=rhyme_markup --warmup_steps=30 --lr=5e-6
```

> **Note**: For **stress** and **rhyme\_stress** markups, I used **only 1 epoch** for pretraining.

**Arguments for `train.py`:**

```text
--name_run        str     Name of the run (for logging purposes). If empty uses args.model instead. 
--train_dataset   str     Path to training dataset (default: dataset/trainset.csv)
--test_dataset    str     Path to test dataset (default: dataset/testset.csv)
--output_dir      str     Directory to save model checkpoints (default: output/)
--checkpoint      str     Path to existing model checkpoint to resume training
--model           str     Model type: 't-lite' or 'qwen' (default: 't-lite')
--epochs          int     Number of training epochs (default: 10)
--lr              float   Learning rate (default: 2e-5)
--batch_size      int     Batch size (default: 32)
--save_steps      int     Save model every N // args.batch_size steps (default: 2000)
--warmup_steps    int     Number of warm-up steps (default: 100)
--log_steps       int     Log training metrics every N // args.batch_size steps (default: 100)
--pretrain                 If set, enables poetry-only pretraining mode
--from_pretrain   str     Path to pretrained model checkpoint
--markup          str     One of ['rhyme_markup', 'stress_markup', 'stanzas', 'rhyme_stress_markup']
```

---

### 🧪 Evaluation

```bash
# Evaluation on prose input
python3 prose-to-poetry/eval.py --name=qwen --model=qwen \
  --checkpoint=output/qwen-05-23-22-32/checkpoint-624 --markup=rhyme_markup

# Prompted generation from scratch (not from prose)
python3 prose-to-poetry/eval.py --name=qwen_generate --model=qwen \
  --checkpoint=output/qwen-05-22-17-18-pretrain/checkpoint-10738 \
  --markup=rhyme_markup --generate
```

**Difference**:

* With `--generate`, the model receives a general poetic prompt (e.g. "write a quatrain with iambic meter and ABAB rhyme").
* Without `--generate`, the model gets prose as input and must convert it to verse with structure.



### Score Computation

```bash
python3 prose-to-poetry/compute_scores.py
```



# Prose-to-Poetry

This project focuses on transforming Russian **prose** into **structured poetry**, specifically **quatrains** with target meter (e.g., iamb) and rhyme scheme (e.g., ABAB).  
It leverages large language models (Qwen) with fine-tuning and markup to control rhythm and rhyme.

## Model Comparison

| **Model**    | **BERTScore** ↑ | **Rhyme Score** ↑ | **Meter Penalty** ↓ | **Volume Score** ↑ |
| ------------ | --------------- | ----------------- | ------------------- | ------------------ |
| `Gemini`     | 0.708           | 0.449             | 0.282               | 1.000              |
| `GigaChat`   | 0.747           | 0.068             | 0.401               | 0.972              |
| `Qwen`       | 0.769           | 0.041             | 0.380               | 0.948              |
| `Qwen_G`     | 0.607           | 0.087             | 0.333               | 0.994              |
| `Qwen_R`     | 0.728           | 0.256             | 0.356               | 0.934              |
| `Qwen_R_G`   | 0.603           | 0.503         | 0.341               | 0.987              |
| `Qwen_R_S`   | 0.670           | 0.168             | 0.338               | 0.841              |
| `Qwen_R_S_G` | 0.590           | 0.343             | 0.313               | 0.990              |
| `Qwen_S`     | 0.704           | 0.046             | 0.329               | 0.936              |
| `Qwen_S_G`   | 0.597           | 0.089             | **0.272**           | 0.983              |

> *Legend:*
> - `Gemini` = Gemini-2.0-Flash
> - `GigaChat` = GigaChat-2-Lite
> - `_R` = rhyme_markup
> - `_S` = stress_markup
> - `_G` = after pretrain only (poetry generation)


## Summary

1. Among general-purpose models, **GigaChat** had strong semantic alignment but failed in poetic structure, while **Gemini** balanced meaning and form well, showing low meter penalty and high rhyme quality.
2. Fine-tuned models like **Qwen\_R** improved rhyme via markup without major loss in semantic similarity but still struggled with metrical conformity and logical coherence.
3. The best rhyme and overall poetic trade-off was achieved by **Qwen\_R\_G**, which was trained solely on poetry generation without prose input.
4. Models using both rhyme and stress annotations, such as **Qwen\_R\_S**, had improved meter but suffered in grammar and meaning, indicating possible conflict between constraints.
5. A trade-off emerged: prose-to-poetry models better preserved content but weakened poetic form, while purely poetry-trained models had stronger formal adherence.
6. BERTScore alone was insufficient to detect semantic drift or logical inconsistencies, and while length control (line count) was generally successful, rhyme and meter remained more challenging.

## Pretrained Models

| Model      | Markup                   | Description                                     | Download Link                                                                                    |
| ---------- | ------------------------ | ----------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `Qwen_R_G` | rhyme\_markup | After pretrain only (poetry generation) | [Download](https://drive.google.com/drive/folders/1MFOMyG1f8MnD1-G00nw6PKI7Gdntte90?usp=sharing) |
| `Qwen_R`   | rhyme\_markup            | Finetuned on prose-to-poetry transformation               | [Download](https://drive.google.com/drive/folders/1MFOMyG1f8MnD1-G00nw6PKI7Gdntte90?usp=sharing) |
---


## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
sudo apt-get install espeak -y
````

### 2. Download Resources

Download stress model & lemma mappings from [verslibre-files (Google Drive)](https://drive.google.com/drive/folders/1oIEM5_UuK-5phD5LtJqCPnSQ5CVQiOoM?usp=sharing)

```bash
cp verslibre-files/word2lemma.pkl external_code/verslibre/models/
cp verslibre-files/accents.pkl external_code/verslibre/tmp/
mkdir external_code/verslibre/tmp/stress_model
cp verslibre-files/stress_model/* external_code/verslibre/tmp/stress_model/
```

## Project Structure

```
├── dataset/                       # Final datasets (CSV)
│   ├── all_poems.csv             # Poems with meter annotation
│   ├── all_stanzas.csv           # Quatrains with meter & rhyme
│   ├── prosa_test_text.csv       # Prose test inputs
│   ├── testset.csv               # Paired prose-poetry test
│   ├── trainset.csv              # Paired prose-poetry train
│   └── trainset_pretrain/        # Quatrains only (no prose)
├── dataset-creation/             # Notebooks for dataset generation and baseline evaluation of Gemini and GigaChat
├── external_code/verslibre/      # Modified version of https://github.com/Koziev/verslibre
└── prose-to-poetry/              # Model code
```

## Training & Evaluation

### Pretrain (on poetic quatrains only)

```bash
python3 prose-to-poetry/train.py \
  --pretrain \
  --model='qwen' \
  --save_steps=5000 \
  --train_dataset=dataset/trainset_pretrain \
  --epochs=2 \
  --log_steps=200 \
  --markup=rhyme_markup \
  --warmup_steps=320 \
  --lr=2e-5
```

> **Note**: For **stress** and **rhyme\_stress** markups, I used **only 1 epoch** for pretraining.

### 🔧 Finetune (on prose-to-verse pairs)

```bash
python3 prose-to-poetry/train.py \
  --model='qwen' \
  --from_pretrain=output/qwen-05-22-17-18-pretrain/checkpoint-10738 \
  --save_steps=2000 \
  --train_dataset=dataset/trainset.csv \
  --epochs=2 \
  --log_steps=200 \
  --markup=rhyme_markup \
  --warmup_steps=30 \
  --lr=5e-6
```

### 📤 Evaluate

#### Prose-to-verse generation (default)

```bash
python3 prose-to-poetry/eval.py \
  --name=qwen \
  --model=qwen \
  --checkpoint=output/qwen/checkpoint-624 \
  --markup=rhyme_markup
```

#### Poetry generation

```bash
python3 prose-to-poetry/eval.py \
  --name=qwen_generate \
  --model=qwen \
  --checkpoint=utput/qwen-05-22-17-18-pretrain/checkpoint-10738 \
  --markup=rhyme_markup \
  --generate
```

### Compute Metrics

```bash
python3 prose-to-poetry/compute_scores.py
```

## License

This repository uses the `MIT License`, except for the `external_code/verslibre` module, which uses the `Unlicense`.



