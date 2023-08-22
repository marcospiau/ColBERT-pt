import itertools

import pytest
import torch
from colbert.infra import ColBERTConfig
from colbert.modeling.tokenization import QueryTokenizer
from transformers import AutoTokenizer

# As we are going to use t5 and bert models, we are going to focus our tests
# on these models
# PONTO DE ATENcao: o bert-base-portuguese-cased nao tem o token [unused0]
# ele acaba virando o unk
tokenizer_configs = {
    'bert-base-uncased':
    ColBERTConfig(checkpoint='bert-base-uncased',
                  query_maxlen=32,
                  query_token='[unused0]',
                  marker_token_position=1,
                  attend_to_mask_tokens=False,
                  query_expand_token='[MASK]'),
    # 'neuralmind/bert-base-portuguese-cased-v1':
    # ColBERTConfig(
    #     checkpoint='neuralmind/bert-base-portuguese-cased',
    #     query_maxlen=32,
    #     query_token='[unused0]', # usamos o '[unused0]', mapeia pro unk
    #     marker_token_position=1,
    #     attend_to_mask_tokens=False,
    #     query_expand_token='[MASK]'),
    'neuralmind/bert-base-portuguese-cased':
    ColBERTConfig(
        checkpoint='neuralmind/bert-base-portuguese-cased',
        query_maxlen=32,
        query_token='[unused1]',  # usamos o '[unused0]', mapeia pro unk
        marker_token_position=1,
        attend_to_mask_tokens=False,
        query_expand_token='[MASK]'),
    'unicamp-dl/ptt5-base-portuguese-vocab':
    ColBERTConfig(checkpoint='unicamp-dl/ptt5-base-portuguese-vocab',
                  query_maxlen=32,
                  query_token='<extra_id_0>',
                  marker_token_position=0,
                  attend_to_mask_tokens=False,
                  query_expand_token='<extra_id_2>')
}

example_queries = [
    'This is an example text.', 'This is another example text.',
    'This is a third example text.', 'This is a fourth example text.',
    'This is a fifth example text.'
]


@pytest.mark.parametrize("name_or_path,attend_to_mask_tokens",
                         itertools.product(tokenizer_configs.keys(),
                                           [True, False]))
def test_query_tokenizer_tensorize(name_or_path, attend_to_mask_tokens):
    raw_tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    config = tokenizer_configs[name_or_path]
    config.attend_to_mask_tokens = attend_to_mask_tokens
    query_tokenizer = QueryTokenizer(config)
    ids, mask = query_tokenizer.tensorize(example_queries)
    # check if the ids are correct
    assert ids.size() == (len(example_queries), config.query_maxlen), \
        'Output tensor has incorrect shape'

    # check token marker position and value
    query_token_id = raw_tokenizer.vocab[config.query_token]

    # check if the mask is correct
    assert ids[:, config.marker_token_position].eq(query_token_id).all(), \
        'The query token was not inserted in the correct position'
    if config.attend_to_mask_tokens:
        assert mask.eq(1).all(), 'Mask tensor has incorrect values'

    # check if outputs for batched and non-batched inputs are the same
    batch = query_tokenizer.tensorize(example_queries, bsize=2)
    ids_batch, mask_batch = zip(*batch)
    ids_batch = torch.cat(ids_batch)
    mask_batch = torch.cat(mask_batch)
    assert ids_batch.equal(ids), 'Batched ids are incorrect'
    assert mask_batch.equal(mask), 'Batched masks are incorrect'
