import pytest
import torch
from colbert.modeling.hf_colbert import (base_class_mapping,
                                         model_object_mapping)
from colbert.modeling.tokenization.colbert_tokenizer import \
    insert_constant_column
from transformers import AutoTokenizer


@pytest.mark.parametrize("arr,pos,fill_value,output",
                         [(torch.tensor([[1, 2, 3], [4, 5, 6]]), 1, 0,
                           torch.tensor([[1, 0, 2, 3], [4, 0, 5, 6]])),
                          (torch.tensor([[1, 2, 3], [4, 5, 6]]), 2, 0,
                           torch.tensor([[1, 2, 0, 3], [4, 5, 0, 6]]))])
def test_insert_constant_column(arr, pos, fill_value, output):
    assert (insert_constant_column(arr, pos, fill_value) == output).all()


@pytest.mark.parametrize("pretrained_model_name_or_path",
                         model_object_mapping.keys())
def test_can_load_tokenizer(pretrained_model_name_or_path):
    """Test if the tokenizer can be loaded"""
    _ = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)


@pytest.mark.parametrize("pretrained_model_name_or_path",
                         model_object_mapping.keys())
def test_marker_placeholder_new_behavior(pretrained_model_name_or_path):
    """Asserts that the new marker placeholder behavior (insert a tensor
    after tokenization) is equivalent to the old behavior (insert a token
    before tokenization)"""
    text = 'This is an example text.'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    # to make things simple, will just insert a -999 token at second position
    # old behavior
    old_ids = tokenizer('. ' + text, return_tensors='pt').input_ids
    marker_token_id = 123
    old_ids[:, 1] = marker_token_id
    new_ids = tokenizer(text, return_tensors='pt').input_ids
    new_ids = insert_constant_column(new_ids, 1, marker_token_id)
    assert (old_ids == new_ids).all()
