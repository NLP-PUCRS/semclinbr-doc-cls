# local modules
from utils.datasetCreation import loadDataset
from utils.parser import majorityClassArgs
from utils.log import setupLogger
from utils.trainingFuncs import computeDetailedMetrics, collectFoldMetrics
from constants.trainingConstants import LABELS, LABELS_TO_MODIFY, MERGE_LABEL
# non-local modules
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
import numpy as np
# native modules
from collections import defaultdict
import json
from datetime import datetime


def main(arguments):

    date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    OUTPUT_DIR = arguments.output_dir
    LOG_DIR = arguments.log_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    CLASSIFICATION_OBJECTIVE = arguments.to_train
    LABEL_LIST = LABELS[CLASSIFICATION_OBJECTIVE]

    logger = setupLogger(outputDir=LOG_DIR, logFileName='baseline_majority')
    logger.info('Majority class baseline pipeline initiated')

    SEED = arguments.seed
    NUM_FOLDS = arguments.num_folds
    logger.info(f'Majority class | Seed: {SEED} | Folds: {NUM_FOLDS}')

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

    # documents ignored, only labels are needed
    all_labels = np.array(dataset['train']['label'])

    logger.info(f'Dataset loaded | Total instances: {len(all_labels)} | {NUM_FOLDS}-fold CV')

    label_counts = np.bincount(all_labels, minlength=len(LABEL_LIST))
    logger.info(
        'Dataset class distribution | ' + ' | '.join(
            f'{id2label[idx]}: {count} ({count / len(all_labels):.4f})'
            for idx, count in enumerate(label_counts)
        )
    )

    # same split techinique and seeds
    skfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(skfold.split(np.zeros(len(all_labels)), all_labels))

    model_results = {
        'model': 'Majority class',
        'date_time': date_time,
        'classification_objective': CLASSIFICATION_OBJECTIVE,
        'num_folds': NUM_FOLDS,
        'folds': {}
    }

    fold_metrics = defaultdict(list)
    per_class_fold_metrics = defaultdict(lambda: defaultdict(list))

    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits, start=1):

        logger.info(f'[{fold_idx}/{NUM_FOLDS}] Fold {fold_idx} started | Test split size: {len(test_idx)}')

        # no eval split due to nothing to tune
        train_labels = all_labels[train_idx]
        test_labels = all_labels[test_idx]

        # dummy classifier to guess the most frequent class
        baseline = DummyClassifier(strategy='most_frequent')
        baseline.fit(np.zeros((len(train_idx), 1)), train_labels)

        majority_class_id = int(baseline.classes_[np.argmax(baseline.class_prior_)])
        majority_class_prior = float(np.max(baseline.class_prior_))

        logger.info(
            f'[{fold_idx}/{NUM_FOLDS}] Majority class: \'{id2label[majority_class_id]}\' | '
            f'Train frequency: {majority_class_prior:.4f}'
        )

        test_predictions = baseline.predict(np.zeros((len(test_idx), 1)))

        detailed_metrics = computeDetailedMetrics(
            predictions=test_predictions,
            labels=test_labels,
            labelList=LABEL_LIST
        )

        model_results['folds'][str(fold_idx)] = {
            'train_fold_size': len(train_idx),
            'test_fold_size': len(test_idx),
            'majority_class': {
                'label': id2label[majority_class_id],
                'label_id': majority_class_id,
                'train_frequency': round(majority_class_prior, 4)
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

    results_path = OUTPUT_DIR / f'results_majority_baseline_{date_time}.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(model_results, f, ensure_ascii=False, indent=4)

    logger.info(f'Results saved to {results_path}')
    logger.info('Pipeline finished')


if __name__ == '__main__':
    main(majorityClassArgs())
