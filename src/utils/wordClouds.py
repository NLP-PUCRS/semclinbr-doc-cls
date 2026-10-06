# non-local modules
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# native modules
from collections import defaultdict
from pathlib import Path
from logging import Logger
import re


def _sanitizeFilename(label: str) -> str:
    """
    formatting a class name to use as a file name
    """
    
    file_name = re.sub(r'[^\w]+', '_', label).strip('_').lower()
    return file_name


def _lemmatizeCorpus(nlp, texts: list[str], batchSize: int = 32) -> list[str]:

    lemmatized = []
    for doc in nlp.pipe(texts, batch_size=batchSize):
        lemmatized.append(' '.join(
            token.lemma_.lower()
            for token in doc
            if not token.is_space
            and not token.is_punct
            and not token.is_stop
            and len(token.lemma_) > 1
        ))
    return lemmatized


def wordCloudsPerClass(
    texts: list[str],
    labels: list[int],
    id2label: dict,
    outputDir: Path,
    nlp=None,
    logger:Logger=None,
    maxWords:int=100,
    width:int=1200,
    height:int=600
) -> dict:

    outputDir.mkdir(parents=True, exist_ok=True)

    texts_by_label = defaultdict(list)
    for text, label_id in zip(texts, labels):
        texts_by_label[int(label_id)].append(text)

    generated = {}

    for label_id, label_texts in sorted(texts_by_label.items()):
        label_name = id2label[label_id]

        processed = _lemmatizeCorpus(nlp, label_texts) if nlp is not None else [t.lower() for t in label_texts]

        joined = ' '.join(processed)

        if not joined.strip():
            if logger:
                logger.warning(f'No usable tokens for class "{label_name}" — skipping word cloud')
            continue

        cloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=maxWords,
            collocations=False
        ).generate(joined)

        _, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.imshow(cloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{label_name} ({len(label_texts)} instances)')
        plt.tight_layout()

        path = outputDir / f'wordcloud_{_sanitizeFilename(label_name)}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        generated[label_name] = path
        if logger:
            logger.info(f'Word cloud saved at \'{path}\' for class \'{label_name}\' using {len(label_texts)} documents')

    return generated