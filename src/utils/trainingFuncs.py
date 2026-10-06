# non-local modules
from transformers import AutoTokenizer, TrainerCallback, Trainer
import evaluate
import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
# native modules
from pathlib import Path


def tokenize(
    batch:dict,
    indices:list[int],
    tokenizer:AutoTokenizer,
    maxLength:int
) -> dict:
    
    overlap = maxLength // 4 # 25% token overlap
    
    tokenized =  tokenizer(
        batch['text'],
        truncation=True,
        max_length=maxLength,
        stride=overlap,
        return_overflowing_tokens=True,
        return_length=False,
        padding=False
    )
    
    sample_map = tokenized.pop('overflow_to_sample_mapping')
    
    tokenized['label'] = [batch['label'][i] for i in sample_map]
    tokenized['doc_id'] = [indices[i] for i in sample_map]
    
    return tokenized


def computeSlidingWindowPrediction(logits, docIds, numDocs):
    """
    Computes the document final prediction as the mean logits of every given sliding window
    """
    doc_logit_sums = np.zeros((numDocs, logits.shape[-1]))
    doc_window_counts = np.zeros(numDocs)
    
    for window_idx, doc_id in enumerate(docIds):
        doc_logit_sums[doc_id] += logits[window_idx]
        doc_window_counts[doc_id] += 1
        
    doc_mean_logits = doc_logit_sums / doc_window_counts[:, None]

    return np.argmax(doc_mean_logits, axis=-1)


_metric = evaluate.load('f1')
def computeMetrics(evalDocIds, numEvalDocs, evalDocLabels):
    """
    compute_metrics function to convert window-level logits into document-level to use in F1
    """
    captured_doc_ids = list(evalDocIds)
    captured_doc_labels = np.asarray(evalDocLabels)

    def computeDocMetrics(preds):
        logits = preds.predictions if hasattr(preds, 'predictions') else preds[0]

        doc_predictions = computeSlidingWindowPrediction(
            logits=logits,
            docIds=captured_doc_ids,
            numDocs=numEvalDocs
        )

        return _metric.compute(
            predictions=doc_predictions,
            references=captured_doc_labels,
            average='macro'
        )

    return computeDocMetrics


def computeDetailedMetrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    labelList: list[str]
) -> dict:
    
    # every average is taken over the full label list so macro, micro and the per class
    # table always describe the same set of classes, even on a fold where some class is
    # never predicted
    label_ids = list(range(len(labelList)))

    accuracy = accuracy_score(labels, predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', labels=label_ids, zero_division=0
    )
    # single label multiclass: every document yields exactly one prediction and one true
    # label, so pooling the per class counts makes micro precision, recall and F1 all
    # collapse into accuracy - micro is therefore only meaningful globally, never per class
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, predictions, average='micro', labels=label_ids, zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        labels, predictions, average=None, labels=label_ids, zero_division=0
    )

    per_class = {
        labelList[i]: {
            'precision': round(float(precision_per_class[i]), 4),
            'recall': round(float(recall_per_class[i]), 4),
            'f1': round(float(f1_per_class[i]), 4),
            'support': int(support_per_class[i])
        }
        for i in range(len(labelList))
    }

    return {
        'global': {
            'accuracy': round(float(accuracy), 4),
            'macro': {
                'precision': round(float(precision_macro), 4),
                'recall': round(float(recall_macro), 4),
                'f1': round(float(f1_macro), 4)
            },
            'micro': {
                'precision': round(float(precision_micro), 4),
                'recall': round(float(recall_micro), 4),
                'f1': round(float(f1_micro), 4)
            }
        },
        'per_class': per_class
    }


def collectFoldMetrics(
    foldMetrics: dict,
    perClassFoldMetrics: dict,
    detailedMetrics: dict
):
    """
    Accumulates one fold of computeDetailedMetrics into the cross validation buckets.

    Both averages are read from the 'global' block: macro averages the per class scores
    unweighted, micro pools the per class counts before scoring. Micro is never derived
    from the per class table because, in single label multiclass, that pooling always
    collapses into accuracy.
    """

    foldMetrics['accuracy'].append(detailedMetrics['global']['accuracy'])

    for average in ('macro', 'micro'):
        for metric in ('precision', 'recall', 'f1'):
            foldMetrics[f'{average}_{metric}'].append(detailedMetrics['global'][average][metric])

    for class_name, class_metrics in detailedMetrics['per_class'].items():
        for metric in ('precision', 'recall', 'f1', 'support'):
            perClassFoldMetrics[class_name][metric].append(class_metrics[metric])


class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial: optuna.Trial, metric: str):
        self.trial = trial
        self.metric = metric

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        value = metrics.get(f'eval_{self.metric}')
        if value is None:
            return
        self.trial.report(value, step=state.epoch)
        if self.trial.should_prune():
            control.should_training_stop = True


def plotLoss(
    logHistory:list[dict], 
    outputDir:Path
):
    
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    
    for entry in logHistory:
        if 'loss' in entry:
            train_steps.append(entry['step'])
            train_losses.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_losses.append(entry['eval_loss'])
    
    stopped_at_step = eval_steps[-1] if eval_steps else None
    
    steps_per_epoch = stopped_at_step / len(eval_steps) if eval_steps else 1
    epoch_ticks = [int(i * steps_per_epoch) for i in range(1, len(eval_steps) + 1)]
    epoch_labels = [str(i) for i in range(1, len(eval_steps) + 1)]
    
    _, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(train_steps, train_losses, label='Train loss', alpha=0.6, linewidth=1)
    ax.plot(eval_steps, eval_losses, label='Eval loss', marker='o', markersize=3)
    
    if stopped_at_step is not None:
        ax.axvline(
            x=stopped_at_step,
            color='red',
            linestyle='--',
            linewidth=1.2,
            label=f'Training stopped'
        )
    
    ax.set_xticks(epoch_ticks)
    ax.set_xticklabels(epoch_labels)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(outputDir / 'training_history.png', dpi=150)
    plt.close()
    
    
class WeightedTrainer(Trainer):
    def __init__(self, *args, classWeights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = classWeights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def tokenLenDistribution(
    texts: list[str],
    tokenizer: AutoTokenizer,
    maxLength: int,
    percentiles: tuple = (50, 75, 90, 95, 99)
):

    lengths = np.array([
        len(tokenizer(text, truncation=False, add_special_tokens=True)['input_ids'])
        for text in texts
    ])

    stats = {
        'count': int(len(lengths)),
        'mean': float(np.mean(lengths)),
        'std': float(np.std(lengths)),
        'min': int(np.min(lengths)),
        'max': int(np.max(lengths)),
        **{f'p{p}': float(np.percentile(lengths, p)) for p in percentiles}
    }

    num_exceeding = int(np.sum(lengths > maxLength))
    
    percentile_str = ' | '.join(f'p{p}={stats[f"p{p}"]:.0f}' for p in percentiles)
    stats_string = f'Token length distribution: ' \
        f' count: {stats["count"]} | mean: {stats["mean"]:.1f} | std: {stats["std"]:.1f} |' \
        f' min: {stats["min"]} | max: {stats["max"]} |' \
        f' {percentile_str} |' \
        f' max_length={maxLength} | documents exceeding: {num_exceeding}' \

    return stats_string