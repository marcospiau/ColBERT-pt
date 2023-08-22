import pytest
import torch
from colbert.infra import ColBERTConfig
from colbert.modeling.tokenization import DocTokenizer
from transformers import AutoTokenizer

# As we are going to use t5 and bert models, we are going to focus our tests
# on these models
# PONTO DE ATENcao: o bert-base-portuguese-cased nao tem o token [unused0]
# ele acaba virando o unk
tokenizer_configs = {
    'bert-base-uncased':
    ColBERTConfig(checkpoint='bert-base-uncased',
                  doc_maxlen=300,
                  doc_token='[unused1]',
                  marker_token_position=1,
                  query_expand_token='[MASK]'),
    'neuralmind/bert-base-portuguese-cased':
    ColBERTConfig(
        checkpoint='neuralmind/bert-base-portuguese-cased',
        doc_maxlen=300,
        doc_token=
        '[unused2]',  # no v1 usamos o 1, mas já deixei ele pra queries no v2
        marker_token_position=1),
    'unicamp-dl/ptt5-base-portuguese-vocab':
    ColBERTConfig(checkpoint='unicamp-dl/ptt5-base-portuguese-vocab',
                  query_maxlen=32,
                  doc_token='<extra_id_1>',
                  marker_token_position=0)
}

example_docs = [
    'This is an example text.', 'This is another example text.',
    'This is a third example text.', 'This is a fourth example text.',
    'This is a fifth example text.'
]


@pytest.mark.parametrize("name_or_path", tokenizer_configs.keys())
def test_doc_tokenizer_tensorize(name_or_path):
    raw_tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    lengths = torch.tensor(
        raw_tokenizer(example_docs, return_length=True).length)
    config = tokenizer_configs[name_or_path]
    # unbatched output
    doc_tokenizer = DocTokenizer(config)
    ids, mask = doc_tokenizer.tensorize(example_docs)
    # check if the ids are correct
    assert ids.size() == (len(example_docs), max(lengths) + 1), \
        'Output tensor has incorrect shape'

    # check token marker position and value
    doc_token_id = raw_tokenizer.vocab[config.doc_token]

    # check if the mask is correct
    assert ids[:, config.marker_token_position].eq(doc_token_id).all(), \
        'The query token was not inserted in the correct position'

    # check if outputs for batched and non-batched inputs are the same
    batches, reverse_indices = doc_tokenizer.tensorize(example_docs, bsize=2)
    ids_batch, mask_batch = zip(*batches)
    ids_batch = torch.cat(ids_batch)[reverse_indices]
    mask_batch = torch.cat(mask_batch)[reverse_indices]
    assert lengths.argsort().equal(reverse_indices)
    assert ids_batch.equal(ids), 'Batched ids are incorrect'
    assert mask_batch.equal(mask), 'Batched masks are incorrect'
