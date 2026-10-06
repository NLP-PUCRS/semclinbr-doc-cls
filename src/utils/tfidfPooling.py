# non-local modules
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
# native modules
from collections import Counter
from math import log
from logging import Logger


def fitTokenIdf(
    texts:list[str],
    tokenizer:AutoTokenizer,
    batchSize:int = 32
) -> dict[int, float]:
    """
    token id to idf table based on number of docs that the token appears in
    """

    document_frequency = Counter()
    num_documents = 0

    for start in range(0, len(texts), batchSize):
        batch = texts[start : start + batchSize]
        encoded = tokenizer(batch, truncation=False, add_special_tokens=False)['input_ids']
        for token_ids in encoded:
            # set tokens that appear in a doc
            document_frequency.update(set(token_ids))
            num_documents += 1

    # smooth inverse document frequency pert token id
    return {
        token_id: log((1 + num_documents) / (1 + df)) + 1.0
        for token_id, df in document_frequency.items()
    }


def addTfidfWeights(
    batch:dict,
    idf:dict[int, float],
    specialTokenIds:set[int]
) -> dict:
    """
    one weight calculated by fitTokenIdf per position of input ids

    not seen tokens get weighted as 0, ignoring out of vocab words and special tokens
    """

    weights_batch = []

    for input_ids in batch['input_ids']:
        token_counts = Counter(input_ids)
        weights_batch.append([
            0.0 if token_id in specialTokenIds
            else (1.0 + log(token_counts[token_id])) * idf.get(token_id, 0.0)
            for token_id in input_ids
        ])

    return {'tfidf_weights': weights_batch}


def tfidfCoverage(
    tokenizedDataset,
    idf:dict[int, float],
    logger:Logger,
    splitName:str
):
    """
    Logging step to see how much of a split the idf table covers
    
    windows with no weights fall back to mean pooling low coverage is a bad sign for token weighting
    """

    num_windows = len(tokenizedDataset)
    num_uncovered_windows = 0
    num_tokens = 0
    num_weighted_tokens = 0

    for weights in tokenizedDataset['tfidf_weights']:
        positive = sum(1 for weight in weights if weight > 0)
        num_tokens += len(weights)
        num_weighted_tokens += positive
        if positive == 0:
            num_uncovered_windows += 1

    logger.info(
        f'TF-IDF weights [{splitName}] | vocabulary: {len(idf)} tokens | '
        f'weighted positions: {num_weighted_tokens}/{num_tokens} '
        f'({num_weighted_tokens / max(num_tokens, 1):.4f}) | '
        f'windows falling back to mean pooling: {num_uncovered_windows}/{num_windows}'
    )


class TfidfDataCollator:
    """
    DataCollatorWithPadding wrap 
    
    pad the tfidf weight per token in the same way data collator pads input ids
    """

    def __init__(self, tokenizer:AutoTokenizer):
        self.base_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self.padding_side = tokenizer.padding_side

    def __call__(self, features:list[dict]) -> dict:

        weights = [feature['tfidf_weights'] for feature in features]
        features = [
            {key: value for key, value in feature.items() if key != 'tfidf_weights'}
            for feature in features
        ]

        batch = self.base_collator(features)
        target_length = batch['input_ids'].shape[1]

        padded_weights = []
        for window_weights in weights:
            padding = [0.0] * (target_length - len(window_weights))
            padded_weights.append(
                padding + list(window_weights) if self.padding_side == 'left'
                else list(window_weights) + padding
            )

        batch['tfidf_weights'] = torch.tensor(padded_weights, dtype=torch.float)

        return batch


class TfidfPooledClassifier(PreTrainedModel):
    """
    Replaces [CLS] polling with a mean over token embeddings weighted by tfidf weights
    Sequence classifier that replaces the [CLS] pooling used by AutoModelForSequenceClassification
    with a mean over the token embeddings weighted by each token's TF-IDF score.
    """

    base_model_prefix = 'encoder'

    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True

    def __init__(self, config, encoder=None):

        super().__init__(config)

        self.num_labels = config.num_labels
        self.encoder = encoder if encoder is not None else AutoModel.from_config(config)

        dropout_prob = config.classifier_dropout \
            if getattr(config, 'classifier_dropout', None) is not None \
            else getattr(config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(dropout_prob)

        # new head initialized
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier.weight.data.normal_(mean=0.0, std=getattr(config, 'initializer_range', 0.02))
        self.classifier.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        tfidf_weights=None,
        labels=None,
        **kwargs
    ):

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        hidden_states = outputs.last_hidden_state.float()
        mask = attention_mask.float()

        # weights to every valid token
        weights = mask if tfidf_weights is None else tfidf_weights.float() * mask

        # window with unseen tokens in the train split fall back to mean pooling
        weight_sums = weights.sum(dim=1, keepdim=True)
        weights = torch.where(weight_sums > 0, weights, mask)
        weight_sums = weights.sum(dim=1, keepdim=True).clamp(min=1e-9)

        pooled = (hidden_states * weights.unsqueeze(-1)).sum(dim=1) / weight_sums
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


def loadTfidfPooledClassifier(
    modelName:str,
    numLabels:int,
    id2label:dict = None,
    label2id:dict = None
) -> TfidfPooledClassifier:

    config = AutoConfig.from_pretrained(
        modelName,
        num_labels=numLabels,
        id2label=id2label,
        label2id=label2id
    )
    encoder = AutoModel.from_pretrained(modelName)

    return TfidfPooledClassifier(config=config, encoder=encoder)
