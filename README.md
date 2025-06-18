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
* `create-prose-poetry-dataset.ipynb`: prepares prose–poetry pairs for training/testing and evaluates Gemini and Gigachat on test prose to create a baseline.

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

**Arguments for `eval.py`:**

```text
--name            str     Name of the experiment (default: 't-lite')
--test_dataset    str     Path to test dataset (default: dataset/prosa_test_text.csv)
--checkpoint      str     Path to model checkpoint
--output_dir      str     Output directory for results (default: output/)
--model           str     Model type: 't-lite' or 'qwen' (default: 't-lite')
--generate                If set, runs prompt-based poetry generation instead of prose-to-poetry conversion
--not_clean               If set, disables postprocessing (markup kept in output)
--markup          str     One of ['rhyme_markup', 'stress_markup', 'stanzas', 'rhyme_stress_markup']
```

---

### 📊 Score Computation

```bash
python3 prose-to-poetry/compute_scores.py
```

**Arguments for `compute_scores.py`:**

```text
--test_dataset    str     Path to the test prose dataset (default: dataset/prosa_test_text.csv)
--input_dir       str     Directory containing model outputs (default: output/models_output/)
--output_dir      str     Where to save computed metrics (default: output/)
```

---

## 📄 License

This repository uses the `MIT License`, except for the `external_code/verslibre` module, which uses the `Unlicense`.



