# non-local modules
import jsonlines
from datasets import Dataset, DatasetDict
# native modules
from pathlib import Path
from random import Random
from typing import TypeAlias, Literal


PathLike: TypeAlias = str | Path
ToModify: TypeAlias = tuple[Literal['drop', 'convert'], list, str]


def _loadJSONL(path:PathLike, toModify:ToModify) -> list[dict]:
    
    CHANGE_OR_DROP, LIST_TO_MODIFY, MERGE_LABEL = toModify
    
    with jsonlines.open(path) as file:
        return [
            {
            'text': instance['text'],
            'label': instance['label'],
            }
            for instance in file 
            if instance['label'] not in LIST_TO_MODIFY 
        ] if CHANGE_OR_DROP == 'drop' else [
            {
                'text': instance['text'],
                'label': instance['label'] if instance['label'] not in LIST_TO_MODIFY else MERGE_LABEL
            }
            for instance in file
        ]


def _randomizeAndSplit(
    instances:list[dict],
    evalProportion:float,
    testProportion:float,
    seed:int
) -> list:
    
    rng = Random(seed)
    rng.shuffle(instances)
    
    dataset_length = len(instances)
    num_test = round(testProportion * dataset_length)
    num_eval = round(evalProportion * dataset_length)
    num_train = dataset_length - num_test - num_eval
    
    return (
        instances[: num_train],
        instances[num_train : num_train + num_eval],
        instances[num_train + num_eval :]
    )


def _buildDataset(
    trainInstances:list[dict],
    evalInstances:list[dict],
    testInstances:list[dict],
    label2id:dict
) -> DatasetDict:

    return DatasetDict({
        'train': Dataset.from_list([
            {'text': instance['text'], 'label': label2id[instance['label']]}
            for instance in trainInstances
        ]),
        'eval': Dataset.from_list([
            {'text': instance['text'], 'label': label2id[instance['label']]}
            for instance in evalInstances
        ]),
        'test': Dataset.from_list([
            {'text': instance['text'], 'label': label2id[instance['label']]}
            for instance in testInstances
        ])
    })


def loadDataset(
    pathJsonl:PathLike,
    evalProportion:float = 0.1,
    testProportion:float = 0.2,
    label2id:dict = None,
    toModify:ToModify = (),
    seed:int = 42
):
    
    instances = _loadJSONL(path=pathJsonl, toModify=toModify)
    
    train_instances, eval_instances, test_instances = _randomizeAndSplit(
        instances=instances, 
        evalProportion=evalProportion, 
        testProportion=testProportion, 
        seed=seed
    )
    
    return _buildDataset(
        trainInstances=train_instances,
        evalInstances=eval_instances,
        testInstances=test_instances,
        label2id=label2id
    )
    