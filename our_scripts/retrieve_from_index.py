"""Index the MS MARCO passage collection with ColBERT.
There will be no enough time to experiment with different configurations, so
we will use the default configuration and only expose basic configurations.
"""
import pprint
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from colbert import Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig

parser = ArgumentParser(
    description='Index the MS MARCO passage collection with ColBERT.',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--checkpoint',
                    required=True,
                    type=str,
                    help='The checkpoint to use.')
# bool resume flag
parser.add_argument('--index_root',
                    default='indexes/mmarco-pt-full',
                    type=str,
                    help='The root directory of the index.')
parser.add_argument('--queries_path',
                    default='/path/to_queries.tsv',
                    type=str,
                    help='The path to the queries.')
parser.add_argument('--index_name',
                    default='msmarco',
                    type=str,
                    help='The name of the index.')
parser.add_argument('--k',
                    default=1_000,
                    type=int,
                    help='How many documents to retrieve per query.')
parser.add_argument('--output_path',
                    default='output_path.tsv',
                    type=str,
                    help='Where to save the ranking results.')

if __name__ == '__main__':
    args = parser.parse_args()
    # run configs are mostly kept fixed to the defaults
    with Run().context(RunConfig(nranks=1, experiment='msmarco')):
        config = ColBERTConfig.load_from_checkpoint(args.checkpoint)
        assert config is not None, f'Could not load checkpoint from {args.checkpoint}'
        print('config is')
        pprint.pprint(config)
        config.configure(root=args.index_root)
        searcher = Searcher(index=args.index_name, config=config)
        queries = Queries(args.queries_path)
        ranking = searcher.search_all(queries, k=args.k)
        ranking.save(args.save_path)
