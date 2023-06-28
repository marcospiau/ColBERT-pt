"""Some collections have documents that are too long for our model. This script
chunks the documents into smaller segments, which are then used as
new documents. In addtion to the new collection, it also creates a mapping
between the new documents and the original documents.
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Iterator, List

import polars as pl
import spacy
from tqdm import tqdm

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_tsv',
                    type=str,
                    default=None,
                    help='Path to the original collection tsv')
parser.add_argument('--output_dir',
                    type=str,
                    default=None,
                    help='Path to the output directory.')
parser.add_argument(
    '--language',
    type=str,
    default=None,
    help='Language of the collection, will be used to'
    'initialize the spacy pipeline')
parser.add_argument('--stride',
                    type=int,
                    default=5,
                    help='Window stride')
parser.add_argument('--max_length',
                    type=int,
                    default=10,
                    help='Max window length')
parser.add_argument('--max_lines',
                    type=int,
                    default=None,
                    help='Max number of lines to process')
parser.add_argument('--max_doc_length',
                    type=int,
                    default=None,
                    required=False,
                    help='Max number of characters allowed in a document. '
                    'If None (default), will use sys.maxsize')


def count_lines(path: str) -> int:
    """Counts the number of lines in a file.

    Args:
        path(str): path to the file

    Returns:
        number of lines in the file

    """
    cmd_output = subprocess.check_output(f'wc -l {path}', shell=True)
    return int(cmd_output.decode('utf8').split()[0])


def get_sentences_from_doc(doc_text: str, nlp) -> List[str]:
    """Splits a document into sentences.

    Args:
        doc_text (str): text of the document

    Returns:
        List[str]: list of sentences

    """
    doc = nlp(doc_text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


def chunk_sentences(
        sentences: List[str],
        stride=5,
        max_length=10) -> Iterator[str]:
    """Chunk sentences in windows.

    Args:
        sentences (List[str]): list of sentences
        stride (int, optional): window stride. Defaults to 5.
        max_length (int, optional): max window length. Defaults to 10.

    Yields:
        Iterator[str]: iterator over the chunks
    """
    for i in range(0, len(sentences), stride):
        segment = ' '.join(sentences[i:i + max_length])
        yield segment
        if i + max_length >= len(sentences):
            break


def chunk_collection(
        nlp,
        tsv_in: str,
        output_dir: str,
        stride=5,
        max_length=10,
        max_lines=None,
        max_doc_length=None):
    """Chunks a collection into smaller documents.

    Args:
        nlp: spacy pipeline
        tsv_in (str): path to the input tsv file
        output_dir (str): path to the output directory
        stride (int, optional): window stride. Defaults to 5.
        max_length (int, optional): max window length. Defaults to 10.
        max_lines (int, optional): max number of lines to process.
            Defaults to None.
        max_doc_length (int, optional): max number of characters allowed in a
            document. If None (default), will not truncate documents.

    Returns:
        None

    """
    total_lines = count_lines(tsv_in)
    if max_lines is not None:
        total_lines = min(max_lines, total_lines)
    output_dir = Path(output_dir)
    id_chunk_map = []
    new_id = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(tsv_in, 'r', encoding='utf8') as fin:
        with open(output_dir / 'collection.tsv', 'w', encoding='utf8') as tsv_fout:
        # for line in tqdm(itertools.islice(f, None)):
            for line in tqdm(itertools.islice(fin, max_lines),
                             total=total_lines):
                doc_id, doc_text = line.strip().split('\t')
                if max_doc_length is not None:
                    doc_text = doc_text[:max_doc_length]
                # get setence chunks, used as new documents
                sentences = get_sentences_from_doc(doc_text, nlp)
                sentence_chunks = chunk_sentences(sentences, stride, max_length)
                for chunk_id, chunk in enumerate(sentence_chunks):
                    id_chunk_map.append((new_id, doc_id, chunk_id))
                    tsv_fout.write(f'{new_id}\t{chunk}\n')
                    new_id += 1

        id_chunk_map = pl.DataFrame(
            id_chunk_map, schema=['new_doc_id', 'original_docid', 'chunk_id'])
        id_chunk_map.write_parquet(output_dir / 'id_chunk_map.parquet')
        total_docs_new = len(id_chunk_map)
        new_to_original_docs_describe = id_chunk_map.groupby(
            'original_docid').count()['count'].describe()
        new_to_orginal_docs_ratio = new_to_original_docs_describe.filter(
            pl.col('statistic').eq('mean'))['value'].item()
        # print some statistics
        print(f'Number of documents in the original collection: {total_lines}')
        print(f'Number of documents in the new collection: {total_docs_new}')
        print(f'New to original documents ratio: {new_to_orginal_docs_ratio}')
        print('New to original documents ratio statistics:')
        print(new_to_original_docs_describe)


if __name__ == '__main__':
    args = parser.parse_args()
    print('Using args:')
    print(args)
    nlp = spacy.blank(args.language)
    nlp.add_pipe('sentencizer')
    nlp.max_length = args.max_doc_length or sys.maxsize
    chunk_collection(nlp=nlp,
                     tsv_in=args.input_tsv,
                     output_dir=args.output_dir,
                     stride=args.stride,
                     max_length=args.max_length,
                     max_lines=args.max_lines,
                     max_doc_length=args.max_doc_length)
