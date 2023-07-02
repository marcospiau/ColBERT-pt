"""This script counts lines on a text file. To keep things simple, each line
will be considered a document. All other preprocessing should be done outside
of this script.
A common usage envolves removing the first column of the file (usually and id)
on queries and collections TSV files. This can be done with the following
command: `cut -f 2- queries.tsv` and passing the output of this pipe to this
script.
We also simply use tokenizer.__call__ to tokenize the documents instead of
customizing the tokenizer to the collection. This is probably not ideal for
all datasets, but is enough to get an idea of document (or query) length.
"""
import argparse
import itertools
import os

import more_itertools
import numpy as np
import pandas as pd
import tqdm
from transformers import AutoTokenizer

pd.options.display.float_format = '{:.2f}'.format

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_file',
                    type=str,
                    default=None,
                    help='Path to the original text file.')
parser.add_argument('--tokenizer_path',
                    type=str,
                    default=None,
                    help='Path to the tokenizer, will be loaded using '
                    'AutoTokenizer.from_pretrained')
parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='How many lines to process at once')
parser.add_argument('--max_lines',
                    type=int,
                    default=None,
                    help='Max number of lines to read from the input file')
parser.add_argument('--nouse_fast_tokenizer',
                    action='store_true',
                    help='Use fast tokenizer')


def count_tokens(tsv_path, tokenizer, batch_size, max_lines=None):

    def process_batch(batch):
        lengths = tokenizer(batch, return_length=True).length
        lengths = np.array(lengths)
        return lengths

    with open(tsv_path, 'r') as f:
        # decode lines
        lines = tqdm.tqdm(f, desc='Tokenizing file')
        lines = itertools.islice(lines, max_lines)
        lines = map(str.strip, lines)
        # process lines in batches
        df = []
        for batch in more_itertools.chunked(lines, batch_size):
            df.append(process_batch(batch))
        df = np.concatenate(df)
        df = pd.Series(df).to_frame('length')
        return df


if __name__ == '__main__':
    args = parser.parse_args()
    # enforce tokenizers parallelism to 1
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, use_fast=not args.nouse_fast_tokenizer)
    df_token_lengths = count_tokens(tsv_path=args.input_file,
                                    tokenizer=tokenizer,
                                    batch_size=args.batch_size,
                                    max_lines=args.max_lines)
    describe = df_token_lengths.describe(
        percentiles=[.25, .5, .75, .9, .95, .99])
    print('Token length statistics:')
    print(describe)
