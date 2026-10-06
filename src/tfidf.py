# local modules
from utils.datasetCreation import loadDataset
from utils.parser import tfidfArgs
from utils.hiperparameterTuning import objectiveTFIDF
from utils.log import setupLogger
from utils.trainingFuncs import computeDetailedMetrics, collectFoldMetrics
from constants.trainingConstants import LABELS, LABELS_TO_MODIFY, MERGE_LABEL
from utils.wordClouds import wordCloudsPerClass
# non-local modules
from sklearn.model_selection import StratifiedKFold, train_test_split
import optuna
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import spacy
# native modules
from functools import partial
from collections import defaultdict
import json
from datetime import datetime


def main(arguments):
    
    # lemmatizer with portuguese stop words and removal of oneletter words
    nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])
    def lemmatizer(text: str) -> list[str]:
        doc = nlp(text)
        return [
            token.lemma_.lower()
            for token in doc
            if not token.is_space and not token.is_punct and not token.is_stop and len(token.lemma_) > 1
        ]
        
    date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    OUTPUT_DIR = arguments.output_dir
    LOG_DIR = arguments.log_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    CLASSIFICATION_OBJECTIVE = arguments.to_train
    LABEL_LIST = LABELS[CLASSIFICATION_OBJECTIVE]

    logger = setupLogger(outputDir=LOG_DIR, logFileName='training_tfidf')
    logger.info('TFIDF pipeline initiated')

    SEED = arguments.seed
    TUNING_SEED = arguments.tuning_seed
    NUM_FOLDS = arguments.num_folds
    logger.info(f'TFIDF + LinearSVC | Seed: {SEED} | Folds: {NUM_FOLDS}')

    id2label = {}
    label2id = {}
    for idx, c in enumerate(LABEL_LIST):
        id2label[idx] = c
        label2id[c] = idx

    DROP_OR_CONVERT = ('drop', LABELS_TO_MODIFY['type'], None) if CLASSIFICATION_OBJECTIVE == 'type' \
        else ('convert', LABELS_TO_MODIFY['specialty'], MERGE_LABEL)
    try:
        dataset = loadDataset(
            pathJsonl=arguments.path_jsonl,
            evalProportion=0,
            testProportion=0,
            label2id=label2id,
            toModify=DROP_OR_CONVERT,
            seed=SEED
        )
    except FileNotFoundError:
        logger.exception(f'JSONL file not found: {arguments.path_jsonl}')
        raise
    except Exception as e:
        logger.exception(f'Dataset loading failed: {e}')
        raise

    all_texts = np.array(dataset['train']['text'])
    all_labels = np.array(dataset['train']['label'])

    logger.info(f"Dataset loaded | Total instances: {len(all_texts)} | {NUM_FOLDS}-fold CV")
    
    if arguments.word_clouds:
        WORD_CLOUDS_DIR = OUTPUT_DIR / 'word_clouds'
        wordCloudsPerClass(
            texts=all_texts.tolist(),
            labels=all_labels.tolist(),
            id2label=id2label,
            outputDir=WORD_CLOUDS_DIR,
            nlp=nlp,
            logger=logger
        )
        logger.info(f'Word clouds per class saved at \'{WORD_CLOUDS_DIR}\'')

    skfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(skfold.split(np.zeros(len(all_labels)), all_labels))

    TUNE = arguments.tune

    model_results = {
        'model': 'TF-IDF + SVM',
        'date_time': date_time,
        'classification_objective': CLASSIFICATION_OBJECTIVE,
        'num_folds': NUM_FOLDS,
        'folds': {}
    }

    fold_metrics = defaultdict(list)
    per_class_fold_metrics = defaultdict(lambda: defaultdict(list))

    for fold_idx, (train_eval_idx, test_idx) in enumerate(fold_splits, start=1):

        logger.info(f'[{fold_idx}/{NUM_FOLDS}] Fold {fold_idx} started | Test split size: {len(test_idx)}')

        train_eval_labels = all_labels[train_eval_idx]

        adjusted_eval_proportion = (arguments.eval_proportion * len(all_labels)) / len(train_eval_idx)
        train_idx, eval_idx = train_test_split(
            train_eval_idx,
            test_size=adjusted_eval_proportion,
            stratify=train_eval_labels,
            random_state=SEED
        )

        train_texts, train_labels = all_texts[train_idx].tolist(), all_labels[train_idx].tolist()
        eval_texts, eval_labels = all_texts[eval_idx].tolist(), all_labels[eval_idx].tolist()
        test_texts, test_labels = all_texts[test_idx].tolist(), all_labels[test_idx].tolist()

        if TUNE:
            num_trials = arguments.trials
            logger.info(f'[{fold_idx}/{NUM_FOLDS}] Hyperparameter tuning — {num_trials} trials')

            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=TUNING_SEED + fold_idx)
            )

            obj = partial(
                objectiveTFIDF,
                trainTexts=train_texts,
                trainLabels=train_labels,
                evalTexts=eval_texts,
                evalLabels=eval_labels,
                logger=logger,
                numTrials=num_trials,
                lemmatizationTokenizer=lemmatizer
            )

            study.optimize(obj, n_trials=num_trials)
            optimized_parameters = study.best_params
            
            max_features = optimized_parameters['max_features']
            ngram_max = optimized_parameters['ngram_max']
            min_df = optimized_parameters['min_df']
            c = optimized_parameters['C']

        else:
            max_features = arguments.max_features
            ngram_max = arguments.ngram_max
            min_df = arguments.min_df
            c = arguments.c

        logger.info(
            f'[{fold_idx}/{NUM_FOLDS}] Hyperparameters:\n'
            f'max_features: {max_features}\n'
            f'ngram_max: {ngram_max}\n'
            f'min_df: {min_df}\n'
            f'C: {c}\n'
        )

        pipeline = Pipeline([
                ('tfidf_vectorizer', TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, ngram_max),
                    min_df=min_df,
                    tokenizer=lemmatizer,
                    token_pattern=None
                )),
                ('svm_classifier', LinearSVC(
                    C=c,
                    max_iter=1000,
                    class_weight='balanced'
                ))
            ])
        
        try:
            pipeline.fit(train_texts, train_labels)
        except Exception as e:
            logger.exception(f'Fit failed: {e}')
            raise

        test_predictions = pipeline.predict(test_texts)
        
        detailed_metrics = computeDetailedMetrics(
            predictions=np.array(test_predictions),
            labels=np.array(test_labels),
            labelList=LABEL_LIST
        )
        
        model_results['folds'][str(fold_idx)] = {
            'test_fold_size': len(test_idx),
            'hyperparameters': {
                'max_features': max_features,
                'ngram_max': ngram_max,
                'min_df': min_df,
                'C': c,
                'tuned': TUNE
            },
            'metrics': detailed_metrics
        }

        collectFoldMetrics(
            foldMetrics=fold_metrics,
            perClassFoldMetrics=per_class_fold_metrics,
            detailedMetrics=detailed_metrics
        )

        logger.info(
            f"[{fold_idx}/{NUM_FOLDS}] Fold {fold_idx} finished | "
            f"accuracy={detailed_metrics['global']['accuracy']:.4f} | "
            f"macro_f1={detailed_metrics['global']['macro']['f1']:.4f}"
        )

    model_results['stats'] = {
        metric: {
            'mean': round(float(np.mean(values)), 4),
            'std': round(float(np.std(values)), 4)
        }
        for metric, values in fold_metrics.items()
    }
    
    model_results['stats_per_class'] = {
        class_name: {
            **{
                metric: {
                    'mean': round(float(np.mean(values)), 4),
                    'std': round(float(np.std(values)), 4)
                }
                for metric, values in class_metrics.items()
                if metric != 'support'
            },
            'support_total': int(sum(class_metrics['support']))
        }
        for class_name, class_metrics in per_class_fold_metrics.items()
    }

    for metric, values in model_results['stats'].items():
        logger.info(f"[FINAL STATS] {metric} - mean: {values['mean']:.4f} | std: {values['std']:.4f}")
        
    for class_name, class_stats in model_results['stats_per_class'].items():
        logger.info(
            f"[FINAL STATS - {class_name}] "
            f"f1: {class_stats['f1']['mean']:.4f}±{class_stats['f1']['std']:.4f} | "
            f"precision: {class_stats['precision']['mean']:.4f}±{class_stats['precision']['std']:.4f} | "
            f"recall: {class_stats['recall']['mean']:.4f}±{class_stats['recall']['std']:.4f}"
        )

    results_path = OUTPUT_DIR / f'results_tfidf_{date_time}.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(model_results, f, ensure_ascii=False, indent=4)

    logger.info(f'Results saved to {results_path}')
    logger.info('Pipeline finished')


if __name__ == '__main__':
    main(tfidfArgs())