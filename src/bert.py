# local modules
from utils.datasetCreation import loadDataset
from utils.trainingFuncs import tokenize, plotLoss, computeSlidingWindowPrediction, computeMetrics, computeDetailedMetrics, collectFoldMetrics, WeightedTrainer, tokenLenDistribution
from utils.parser import args
from utils.hiperparameterTuning import objective
from utils.log import setupLogger
from utils.tfidfPooling import fitTokenIdf, addTfidfWeights, tfidfCoverage, TfidfDataCollator, loadTfidfPooledClassifier
from constants.trainingConstants import LABELS, LABELS_TO_MODIFY, EFFECTIVE_BATCH_SIZE, MERGE_LABEL
# non-local modules
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
    )
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import optuna
import numpy as np
# native modules
from functools import partial
from collections import defaultdict
import gc
import json
from datetime import datetime


def garbage_collector():
    gc.collect()
    torch.cuda.empty_cache()


def main(arguments):
    
    date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    MODEL = arguments.model
    TOKENIZER = arguments.tokenizer
    MAX_LENGTH = arguments.max_length
    
    OUTPUT_DIR = arguments.output_dir
    LOG_DIR = arguments.log_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    CLASSIFICATION_OBJECTIVE = arguments.to_train
    LABEL_LIST = LABELS[CLASSIFICATION_OBJECTIVE]
    
    USE_TFIDF = arguments.tfidf_pooling
    
    MODEL_NAME_TO_FILES = MODEL.split('/')[-1] + ('_tfidfPooling' if USE_TFIDF else '')
    logger = setupLogger(outputDir=LOG_DIR, logFileName=f"training_{MODEL_NAME_TO_FILES}")
    logger.info('Pipeline initiated')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = arguments.seed
    TUNING_SEED = arguments.tuning_seed
    NUM_FOLDS = arguments.num_folds
    POOLING = 'TF-IDF weighted mean' if USE_TFIDF else '[CLS]'
    logger.info(f'Model: {MODEL} | Seed: {SEED} | Folds: {NUM_FOLDS} | Pooling: {POOLING}')

    model_results = {
        'model': MODEL,
        'date_time': date_time,
        'classification_objective': CLASSIFICATION_OBJECTIVE,
        'num_folds': NUM_FOLDS,
        'pooling_strategy': POOLING,
        'folds': {}
    }
        
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
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    # tfidf collator is the base collator with weights padding added
    data_collator = TfidfDataCollator(tokenizer=tokenizer) if USE_TFIDF else DataCollatorWithPadding(tokenizer=tokenizer)

    def buildModel():
        """
        Model built based on the pooling strategy
        tfidf pooling results in buildTfidfPooledClassifier
        and [CLS] pooling results in regular hugging face AutoModel
        """
        if USE_TFIDF:
            return loadTfidfPooledClassifier(
                modelName=MODEL,
                numLabels=len(LABEL_LIST),
                id2label=id2label,
                label2id=label2id
            )
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL,
            num_labels=len(LABEL_LIST),
            id2label=id2label,
            label2id=label2id
        )

    all_texts = dataset['train']
    all_labels = np.array(dataset['train']['label'])
    
    logger.info(
        f"Dataset loaded | Total instances: {len(all_texts)} | {NUM_FOLDS}-fold CV"
    )
    
    token_len_stats = tokenLenDistribution(
        texts=all_texts['text'],
        tokenizer=tokenizer,
        maxLength=MAX_LENGTH,
    )
    logger.info(token_len_stats)
    model_results['token_length_stats'] = token_len_stats
    
    skfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(skfold.split(np.zeros(len(all_labels)), all_labels))
    
    TUNE = arguments.tune
    OUTPUT_DIR_MODEL = OUTPUT_DIR / 'model_output' / MODEL_NAME_TO_FILES
    
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

        train_dataset_raw = all_texts.select(train_idx)
        eval_dataset_raw = all_texts.select(eval_idx)
        test_dataset_raw = all_texts.select(test_idx)
        
        # tokenizing the dataset per fold avoids data leakage from stratified k-fold
        train_tokenized_dataset = train_dataset_raw.map(
            tokenize, batched=True, with_indices=True, remove_columns=train_dataset_raw.column_names,
            fn_kwargs={'tokenizer': tokenizer, 'maxLength': MAX_LENGTH}
        )
        
        eval_tokenized_dataset = eval_dataset_raw.map(
            tokenize, batched=True, with_indices=True, remove_columns=eval_dataset_raw.column_names,
            fn_kwargs={'tokenizer': tokenizer, 'maxLength': MAX_LENGTH}
        )
        
        test_tokenized_dataset = test_dataset_raw.map(
            tokenize, batched=True, with_indices=True, remove_columns=test_dataset_raw.column_names,
            fn_kwargs={'tokenizer': tokenizer, 'maxLength': MAX_LENGTH}
        )
        
        if USE_TFIDF:
            # the IDF is fitted only on the training documents of this fold, for the same reason
            # the tokenization is redone per fold: no test information may reach the weights
            # token weights based on train dataset
            idf_table = fitTokenIdf(texts=train_dataset_raw['text'], tokenizer=tokenizer)
            special_token_ids = set(tokenizer.all_special_ids)
            logger.info(f'[{fold_idx}/{NUM_FOLDS}] IDF fitted on {len(train_idx)} training documents')

            # adding the weights into the tokenized splits
            train_tokenized_dataset = train_tokenized_dataset.map(
                addTfidfWeights, batched=True, fn_kwargs={'idf': idf_table, 'specialTokenIds': special_token_ids}
            )
            eval_tokenized_dataset = eval_tokenized_dataset.map(
                            addTfidfWeights, batched=True, fn_kwargs={'idf': idf_table, 'specialTokenIds': special_token_ids}
            )
            test_tokenized_dataset = test_tokenized_dataset.map(
                            addTfidfWeights, batched=True, fn_kwargs={'idf': idf_table, 'specialTokenIds': special_token_ids}
            )
            
            # logging the split coverage based on the IDF table
            tfidfCoverage(tokenizedDataset=train_tokenized_dataset, idf=idf_table, logger=logger, splitName='train')
            tfidfCoverage(tokenizedDataset=eval_tokenized_dataset, idf=idf_table, logger=logger, splitName='eval')
            tfidfCoverage(tokenizedDataset=test_tokenized_dataset, idf=idf_table, logger=logger, splitName='test')

        # weights calculated based on the sliding window label count to not overweight classes that are overlapped
        train_labels_fold = np.array(train_tokenized_dataset['label'])
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(len(LABEL_LIST)),
            y=train_labels_fold
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        
        tokenized_dataset_fold = {
            'train': train_tokenized_dataset,
            'eval': eval_tokenized_dataset,
            'test': test_tokenized_dataset
        }
        
        compute_metrics = computeMetrics(
            evalDocIds=eval_tokenized_dataset['doc_id'],
            numEvalDocs=len(eval_idx),
            evalDocLabels=all_labels[eval_idx]
        )
        
        FOLD_OUTPUT_DIR = OUTPUT_DIR / MODEL_NAME_TO_FILES / f'fold_{fold_idx}'
        FOLD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if TUNE:
        
            num_trials = arguments.trials
            logger.info(f'[{fold_idx}/{NUM_FOLDS}] Fold {fold_idx} | Hyperparameters tunning intiated - Running for {num_trials} trials')
            
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=TUNING_SEED + fold_idx),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
            )
            
            obj = partial(
                objective,
                modelInit=buildModel,
                tokenizer=tokenizer,
                dataCollator=data_collator,
                outputDir=FOLD_OUTPUT_DIR,
                device=device,
                tokenizedDataset=tokenized_dataset_fold,
                metricForBestModel='f1',
                seed=SEED,
                effectiveBatchSize=EFFECTIVE_BATCH_SIZE,
                logger=logger,
                numTrials=num_trials,
                classWeights=class_weights_tensor,
                computeMetrics=compute_metrics
            )
            
            study.optimize(obj, n_trials=num_trials)
            optimized_parameters = study.best_params
            
            lr = optimized_parameters['learning_rate']
            batch_size = optimized_parameters['per_device_train_batch_size']
            epochs = optimized_parameters['num_train_epochs']
            weight_decay = optimized_parameters['weight_decay']
            lr_scheduler_type = optimized_parameters['lr_scheduler_type']
            warmup_ratio = optimized_parameters['warmup_ratio']
            patience = optimized_parameters['patience']
            adam_epsilon = optimized_parameters['adam_epsilon']
            max_grad_norm = optimized_parameters['max_grad_norm']
            
        else:
            
            lr = arguments.lr
            batch_size = arguments.batch_size
            epochs = arguments.epochs
            weight_decay = arguments.weight_decay
            lr_scheduler_type = arguments.lr_scheduler_type
            warmup_ratio = arguments.warmup_ratio
            patience = arguments.patience
            adam_epsilon = arguments.adam_epsilon
            max_grad_norm = arguments.max_grad_norm
        
        hyperparameters = {
            'learning_rate': lr,
            'batch_size': batch_size,
            'num_train_epochs': epochs,
            'weight_decay': weight_decay,
            'lr_scheduler_type': lr_scheduler_type,
            'warmup_ratio': warmup_ratio,
            'patience': patience,
            'adam_epsilon': adam_epsilon,
            'max_grad_norm': max_grad_norm,
            'tuned': TUNE
        }
    
        logger.info(
            f'Hyperparameters being used for training:\n'
            f'Learning Rate: {lr}\n'
            f'Batch size: {batch_size}\n'
            f'Maximum number of epochs: {epochs}\n'
            f'Patience: {patience}\n'
            f'Weight decay: {weight_decay}\n'
            f'Learning rate scheduler type: {lr_scheduler_type}\n'
            f'Warmup ratio: {warmup_ratio}\n'
            f'Adam epsilon: {adam_epsilon}\n'
            f'Max grad norm: {max_grad_norm}\n'
        )

        GRADIENT_ACCUMULATION_STEPS = max(1, EFFECTIVE_BATCH_SIZE // batch_size)

        set_seed(SEED + fold_idx)
        model = buildModel().to(device)

        FOLD_MODEL_OUTPUT_DIR = OUTPUT_DIR_MODEL / f'fold_{fold_idx}'
        training_args = TrainingArguments(
                    output_dir=FOLD_MODEL_OUTPUT_DIR,
                    seed=SEED,
                    data_seed=SEED,
                    eval_strategy='epoch',
                    save_strategy='epoch',
                    learning_rate=lr,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    num_train_epochs=epochs,
                    weight_decay=weight_decay,
                    lr_scheduler_type=lr_scheduler_type,
                    adam_epsilon=adam_epsilon,
                    max_grad_norm=max_grad_norm,
                    warmup_ratio=warmup_ratio,
                    load_best_model_at_end=True,
                    save_total_limit=1,
                    metric_for_best_model='f1',
                    greater_is_better=True,
                    bf16=True if device == 'cuda' else False,
                    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
                )
            
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset_fold['train'],
            eval_dataset=tokenized_dataset_fold['eval'],
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
            classWeights=class_weights_tensor
        )
        
        try:
            trainer.train()
        except Exception as e:
            logger.exception(f'[{fold_idx}/{NUM_FOLDS}] Training failed for fold {fold_idx}: {e}')
            del trainer, model
            garbage_collector()
            raise

        plotLoss(
            logHistory=trainer.state.log_history,
            outputDir=FOLD_OUTPUT_DIR
        )
        
        test_predictions = trainer.predict(tokenized_dataset_fold['test'])
        
        # document-level prediction is given by the mean logits of every sliding window 
        # generated when passing the dataset through the tokenization process
        doc_ids = test_tokenized_dataset['doc_id']
        num_test_docs = len(test_idx)
        predicted_labels = computeSlidingWindowPrediction(
            logits=test_predictions.predictions,
            docIds=doc_ids,
            numDocs=num_test_docs
        )
        
        true_labels = all_labels[test_idx]
    
        detailed_metrics = computeDetailedMetrics(
            predictions=predicted_labels,
            labels=true_labels,
            labelList=LABEL_LIST
        )

        model_results['folds'][str(fold_idx)] = {
            'test_fold_size': len(test_idx),
            'hyperparameters': hyperparameters,
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
        
        del trainer, model
        garbage_collector()
    
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
    
    results_path = OUTPUT_DIR / f'results_{MODEL_NAME_TO_FILES}_{date_time}.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(model_results, f, ensure_ascii=False, indent=4)

    logger.info(f'Results saved to {results_path}')
    logger.info('Pipeline finished')


if __name__ == '__main__':
    main(args())