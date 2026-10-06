# Document Classification - SemClinBr

This repository contains the resources developed in the record type and medical specialty predictors based on the document-level annotations provided by the `SemClinBr v2.0` dataset.

## ToC

1. [Experiments](#experiments)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Replication Scripts](#replication-scripts)
5. [Cite](#cite)

## Experiments

The experiments aim to make document-level predictions over the `SemClinBr v2.0` corpus for record type and medical specialty over five classification strategies: a majority class baseline ([src/majority.py](src/majority.py)), bag of words and TF-IDF representations fed to a LinearSVC ([src/bow.py](src/bow.py), [src/tfidf.py](src/tfidf.py)), and the fine-tuning of two Portuguese BERT models, BERTimbau and BioBERTpt ([src/bert.py](src/bert.py)).

Each trained model goes over a nested cross validation procedure. Each fold's model is tuned with Optuna, maximizing the macro F1, and the model from the best trial is evaluated on the held-out data.

Both global and per class metrics per fold are saved and the final metrics are given through the mean and standard deviation of every predictions metrics. The global block holds the accuracy along with the macro and micro averages of precision, recall and F1, and the per class block holds precision, recall, F1 and support for every class.

Classes with a relative frequency lower than 6% were dropped for the record type classification and converted into `outra` for the medical specialty prediction. The affected labels are listed in [src/constants/trainingConstants.py](src/constants/trainingConstants.py).

## Dataset

The annotated corpus is available for download at [SemClinBr v2.0](#).

## Installation

Developed and tested with **Python 3.11.2**.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download pt_core_news_sm
```

The BERT scripts expect a CUDA-capable GPU. Training falls back to the CPU when no GPU is available. A complete tuning run is impractical in that setting.

## Replication Scripts

All the scripts to replicate the experiments are available at [scripts/](scripts/):

| Script                                             | Description                                                             |
| -------------------------------------------------- | ----------------------------------------------------------------------- |
| [majorityClass.sh](scripts/majorityClass.sh)       | Majority class baseline, over the same folds used by the trained models |
| [train_bows.sh](scripts/train_bows.sh)             | Bag of words + LinearSVC                                                |
| [train_tfidfs.sh](scripts/train_tfidfs.sh)         | TF-IDF + LinearSVC (generates word clouds)                              |
| [train_berts.sh](scripts/train_berts.sh)           | Fine-tuning of BERTimbau (`neuralmind/bert-base-portuguese-cased`)      |
| [train_biobertpts.sh](scripts/train_biobertpts.sh) | Fine-tuning of BioBERTpt (`pucpr/biobertpt-all`)                        |

Each script runs both classification objectives, `specialty` and `type`, in sequence. Every script resolves the repository root from its own location, so it can be launched from any working directory:

```bash
bash scripts/train_tfidfs.sh
```

Each run writes a timestamped results JSON — cross validation statistics, per class statistics and the hyperparameters used in each fold — to the `--output_dir` given in the script (`outputs/<approach>_<objective>/`), and a timestamped log to `logs/`. Additionally, the BERT runs save each fold's best model and loss curve into the `--output_dir`.

## Cite

- Record Type and Medical Specialty Classification of Clinical Notes in Brazilian Portuguese:

```bibtex
```

- SemClinBr v2.0 paper:

```bibtex
```
