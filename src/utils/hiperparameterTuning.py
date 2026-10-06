# local modules
from utils.trainingFuncs import OptunaPruningCallback, WeightedTrainer
# non-local modules
import optuna
from transformers import (
    AutoTokenizer, DataCollatorWithPadding,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
# native modules
from pathlib import Path
import gc
import shutil
from logging import Logger


def garbage_collector():
    gc.collect()
    torch.cuda.empty_cache()


def objective(
    trial:optuna.Trial,
    modelInit,
    tokenizer:AutoTokenizer,
    dataCollator:DataCollatorWithPadding,
    outputDir:Path,
    device:str,
    tokenizedDataset:dict,
    metricForBestModel:str,
    seed:int,
    effectiveBatchSize:int,
    logger:Logger,
    numTrials:int,
    classWeights:torch.tensor,
    computeMetrics
):
    
    lr = trial.suggest_float(name='learning_rate', low=5e-6, high=1e-4, log=True)
    batch_size = trial.suggest_categorical('per_device_train_batch_size', [4, 8, 16])
    epochs = trial.suggest_int(name='num_train_epochs', low=5, high=50)
    weight_decay = trial.suggest_float(name='weight_decay', low=0.001, high=0.25)
    lr_scheduler_type = trial.suggest_categorical(
        name='lr_scheduler_type', 
        choices=['linear', 'cosine', 'constant_with_warmup', 'cosine_with_restarts', 'polynomial']
    )
    warmup_ratio = trial.suggest_float(name='warmup_ratio', low=0.0, high=0.2)
    patience = trial.suggest_int(name='patience', low=2, high=10)
    adam_epsilon = trial.suggest_categorical('adam_epsilon', [1e-8, 1e-6])
    max_grad_norm = trial.suggest_float('max_grad_norm', low=0.5, high=2.0)
    
    trial_number = trial.number + 1
    logger.info(f'[{trial_number}/{numTrials}] Trial {trial_number} started | ' 
                f'lr={lr} | batch_size={batch_size} | epochs={epochs} | weight_decay={weight_decay} | '
                f'lr_scheduler_type={lr_scheduler_type} | warmup_ratio={warmup_ratio} | patience={patience} | '
                f'adam_epsilon={adam_epsilon} | max_grad_norm={max_grad_norm}'
    )
    
    set_seed(seed=seed)
    
    model = modelInit().to(device)    
    
    trial_output_dir = outputDir / f'optuna_model_output_trial_{trial_number}'
    GRADIENT_ACCUMULATION_STEPS = max(1, effectiveBatchSize // batch_size)
    training_args = TrainingArguments(
        output_dir=trial_output_dir,
        seed=seed,
        data_seed=seed,
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
        train_dataset=tokenizedDataset['train'],
        eval_dataset=tokenizedDataset['eval'],
        processing_class=tokenizer,
        compute_metrics=computeMetrics,
        data_collator=dataCollator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=patience),
            OptunaPruningCallback(trial, metricForBestModel)
        ],
        classWeights=classWeights
    )
    
    # training
    # exceptions for pruning and for generic error (such as OOM)
    try:
        trainer.train()
        
        if trial.should_prune():
            del trainer
            del model
            garbage_collector()
            
            if trial_output_dir.exists():
                shutil.rmtree(trial_output_dir)
                
            logger.info(f'[{trial_number}/{numTrials}] Trial pruned')
            raise optuna.exceptions.TrialPruned()
    # pruning exception
    except optuna.exceptions.TrialPruned:
        raise
    # general error exception
    except Exception as e:
        del trainer
        del model
        garbage_collector()
        
        if trial_output_dir.exists():
            shutil.rmtree(trial_output_dir)
        logger.exception(f'[{trial_number}/{numTrials}] Unexpected error during training: {e}')
        raise
    
    results = trainer.evaluate()
    
    # getting the metric to compare
    try:
        metric_value = results[f'eval_{metricForBestModel}']
    except KeyError:
        logger.exception(f'[{trial_number}/{numTrials}] Metric "eval_{metricForBestModel}" not found in results. Available keys: {list(results.keys())}')
        raise

    # best value information
    try:
        best_info = f"{trial.study.best_value:.4f} (trial {trial.study.best_trial.number + 1})"
    except ValueError:
        best_info = "None"
    
    logger.info(
        f"[{trial_number}/{numTrials}] Trial {trial_number} finished | {metricForBestModel}={metric_value:.4f} | "
        f"Best so far: {best_info}"
    )
    
    trial.set_user_attr(metricForBestModel, metric_value)
    
    del trainer
    del model
    garbage_collector()
    
    if trial_output_dir.exists():
        shutil.rmtree(trial_output_dir)
        
    return metric_value


def objectiveTFIDF(
    trial: optuna.Trial,
    trainTexts: list[str],
    trainLabels: list[int],
    evalTexts: list[str],
    evalLabels: list[int],
    logger: Logger,
    numTrials: int,
    lemmatizationTokenizer
) -> float:
    
    max_features = trial.suggest_categorical('max_features', [5000, 10000, 20000, 50000])
    ngram_max = trial.suggest_categorical('ngram_max', [1, 2, 3])
    min_df = trial.suggest_int('min_df', 1, 5)
    c = trial.suggest_float('C', 1e-3, 1e2, log=True)

    logger.info(
        f'[{trial.number + 1}/{numTrials}] Trial {trial.number + 1} started | '
        f'max_features={max_features} | ngram_max={ngram_max} | min_df={min_df} | C={c:.4f}'
    )

    pipeline = Pipeline([
        ('tfidf_vectorizer', TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, ngram_max),
            min_df=min_df,
            tokenizer=lemmatizationTokenizer,
            token_pattern=None
        )),
        ('svm_classifier', LinearSVC(
            C=c,
            max_iter=1000,
            class_weight='balanced'
        ))
    ])

    try:
        pipeline.fit(trainTexts, trainLabels)
    except Exception as e:
        logger.exception(f'[{trial.number + 1}/{numTrials}] Fit failed: {e}')
        raise

    eval_predictions = pipeline.predict(evalTexts)
    metric_value = f1_score(evalLabels, eval_predictions, average='macro', zero_division=0)

    logger.info(f'[{trial.number + 1}/{numTrials}] Trial {trial.number + 1} finished | macro_f1={metric_value:.4f}')

    trial.set_user_attr('macro_f1', metric_value)

    return metric_value


def objectiveBOW(
    trial: optuna.Trial,
    trainTexts: list[str],
    trainLabels: list[int],
    evalTexts: list[str],
    evalLabels: list[int],
    logger: Logger,
    numTrials: int,
    lemmatizationTokenizer
) -> float:
    """
    Same search space as objectiveTFIDF so that the only difference between the two
    pipelines is the feature weighting: raw term counts here, TF-IDF there
    """

    max_features = trial.suggest_categorical('max_features', [5000, 10000, 20000, 50000])
    ngram_max = trial.suggest_categorical('ngram_max', [1, 2, 3])
    min_df = trial.suggest_int('min_df', 1, 5)
    c = trial.suggest_float('C', 1e-3, 1e2, log=True)

    logger.info(
        f'[{trial.number + 1}/{numTrials}] Trial {trial.number + 1} started | '
        f'max_features={max_features} | ngram_max={ngram_max} | min_df={min_df} | C={c:.4f}'
    )

    pipeline = Pipeline([
        ('bow_vectorizer', CountVectorizer(
            max_features=max_features,
            ngram_range=(1, ngram_max),
            min_df=min_df,
            tokenizer=lemmatizationTokenizer,
            token_pattern=None
        )),
        ('svm_classifier', LinearSVC(
            C=c,
            max_iter=1000,
            class_weight='balanced'
        ))
    ])

    try:
        pipeline.fit(trainTexts, trainLabels)
    except Exception as e:
        logger.exception(f'[{trial.number + 1}/{numTrials}] Fit failed: {e}')
        raise

    eval_predictions = pipeline.predict(evalTexts)
    metric_value = f1_score(evalLabels, eval_predictions, average='macro', zero_division=0)

    logger.info(f'[{trial.number + 1}/{numTrials}] Trial {trial.number + 1} finished | macro_f1={metric_value:.4f}')

    trial.set_user_attr('macro_f1', metric_value)

    return metric_value
