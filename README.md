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



