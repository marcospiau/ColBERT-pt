"""Index the MS MARCO passage collection with ColBERT.
There will be no enough time to experiment with different configurations, so
we will use the default configuration and only expose basic configurations.
"""
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pprint

parser = ArgumentParser(
    description='Index the MS MARCO passage collection with ColBERT.',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--checkpoint',
                    required=True,
                    type=str,
                    help='The checkpoint to use.')
# bool resume flag
parser.add_argument('--resume',
                    action='store_true',
                    help='Resume from the checkpoint.')
parser.add_argument('--index_root',
                    default='indexes/mmarco-pt-full',
                    type=str,
                    help='The root directory of the index.')
parser.add_argument('--collection',
                    default='/path/to/MSMARCO/collection.tsv',
                    type=str,
                    help='The path to the MSMARCO collection.')
parser.add_argument('--index_name',
                    default='msmarco',
                    type=str,
                    help='The name of the index.')
parser.add_argument('--nbits',
                    default=2,
                    type=int,
                    help='The number of bits for the index.')

if __name__ == '__main__':
    args = parser.parse_args()
    with Run().context(RunConfig(nranks=1, experiment='msmarco')):
        config = ColBERTConfig.load_from_checkpoint(args.checkpoint)
        config.configure(
            nbits=args.nbits,
            root=args.index_root,
        )
        print('config is')
        pprint.pprint(config)
        indexer = Indexer(checkpoint=args.checkpoint, config=config)
        indexer.index(name=args.index_name,
                      collection=args.collection,
                      overwrite='resume' if args.resume else True)
