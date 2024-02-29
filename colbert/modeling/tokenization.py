import string

import torch
from transformers import AutoTokenizer

from colbert.clean_training_codes.modeling import ColbertConfig
from colbert.modeling.tokenization.utils import (_sort_by_length,
                                                 _split_into_batches)

from transformers import BatchEncoding
from tokenizers import AddedToken


def insert_constant_column(arr, pos, fill_value):
    """Insert a column of `fill_value` at the specified position in a 2D
        array.
    """
    begin, end = arr[:, :pos], arr[:, pos:]
    fill = torch.full_like(arr[:, 0], fill_value).unsqueeze(-1)
    return torch.cat([begin, fill, end], dim=-1)


def add_special_tokens(raw_tokenizer, marker_token, mask_expand_token):
    """Prepare the raw tokenizer, including marker and expansion as special
        if they are not already.
    """
    tokens_to_add = [marker_token, mask_expand_token]
    for token in tokens_to_add:
        raw_tokenizer.add_tokens(AddedToken(token,
                                            rstrip=False,
                                            lstrip=False,
                                            single_word=False,
                                            normalized=False,
                                            special=True),
                                 special_tokens=True)
    return raw_tokenizer


class ColbertTokenizer:

    def __init__(
        self,
        tokenizer_model_path: str,
        max_length: int,
        marker_token: str,
        marker_token_position: int,
        attend_to_mask_tokens: bool = False,
        mask_expand_token: str = None,
        tokenizer_alias: str = 'TokenizerAliasPlaceholder',
        tokenizer_kwargs: dict = None,
        validate_special_tokens: bool = True,
        mask_punctuation: bool = False,
    ):
        """Base class for document and query ColBERT tokenizers.

        Args:
            raw_tokenizer (PretrainedTokenizerFast): The raw tokenizer.
            tokenizer_model_path (str): The name or path of the pretrained
                model to use to initialize the tokenizer.
            max_length (int): The maximum length of the tokenized text.
            marker_token (str): The marker token.
            marker_token_position (int): The position of the marker token.
                For BERT models this is 1, for T5 models this is 0.
            attend_to_mask_tokens (bool, optional): Whether to attend to mask
                tokens. Defaults to False.
            mask_expand_token (str, optional): The mask expand token. Defaults
                to None. Usually only queries are expanded with mask tokens.
            tokenizer_alias (str, optional): The name of the tokenizer.
                Defaults to 'TokenizerAliasPlaceholder'.
            tokenizer_kwargs (dict, optional): Keyword arguments for the
                tokenizer. Defaults to {}.
            validate_special_tokens (bool, optional): Whether to validate the
                special tokens. Defaults to True. Ideally this should be True,
                but it can be turned off for compatibility with some models.
            mask_punctuation (bool, optional): Whether to mask punctuation.
                It is used on document tokenization to zero out scores
                for punctuation. Defaults to False.
        """
        self.tokenizer_model_path = tokenizer_model_path
        self.raw_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_path)
        self.raw_tokenizer = add_special_tokens(
            self.raw_tokenizer,
            marker_token=marker_token,
            mask_expand_token=mask_expand_token)
        self.max_length = max_length
        self.marker_token = marker_token
        self.marker_token_position = marker_token_position
        self.attend_to_mask_tokens = attend_to_mask_tokens
        self.mask_expand_token = mask_expand_token
        self.mask_punctuation = mask_punctuation
        self.mask_punctuation = mask_punctuation
        if self.mask_punctuation:
            # THIS IS WRONG, but I keep it here to remember where it came
            # from on original codebase
            # self.skiplist = {
            #     w: True
            #     for symbol in string.punctuation
            #     for w in [symbol, self.doc_tokenizer.tok.encode(
            #     symbol, add_special_tokens=False)[0]]}
            self.tokens_to_skip = torch.tensor(
                list(
                    filter(
                        None,
                        map(self.raw_tokenizer.vocab.get,
                            string.punctuation))))

        # tokens_to_ids = self.tok.convert_tokens_to_ids
        # this is used instead tokenizer.convert_tokens_to_ids to raise an
        # error for OOV tokens
        convert_tokens_to_ids = self.raw_tokenizer.vocab.__getitem__
        if self.mask_expand_token is not None:
            self.mask_expand_token_id = convert_tokens_to_ids(
                self.mask_expand_token)
        else:
            self.mask_expand_token_id = None
        self.tokenizer_alias = tokenizer_alias
        self.tokenizer_kwargs = tokenizer_kwargs or {}

        self.marker_token_id = convert_tokens_to_ids(self.marker_token)
        self.validate_special_tokens = validate_special_tokens
        if validate_special_tokens:
            self.check_special_tokens()
        self.is_used = False

    def check_special_tokens(self):
        if self.marker_token_id is None:
            raise ValueError("marker_token_id cannot be None.")
        if self.marker_token_id == self.raw_tokenizer.unk_token_id:
            raise ValueError(
                "marker_token_id cannot be equal to unk_token_id.")
        if self.marker_token_id == self.raw_tokenizer.pad_token_id:
            raise ValueError(
                "marker_token_id cannot be equal to pad_token_id.")
        if self.mask_expand_token_id is not None:
            if self.mask_expand_token_id == self.raw_tokenizer.unk_token_id:
                raise ValueError(
                    "mask_expand_token_id cannot be equal to unk_token_id.")
            if self.mask_expand_token_id == self.raw_tokenizer.pad_token_id:
                raise ValueError(
                    "mask_expand_token_id cannot be equal to pad_token_id.")

    def encode_texts(self, batch_text):
        """Encode a list of texts and return the input ids and attention
            mask.
        """
        assert type(batch_text) in [list, tuple], (type(batch_text))
        encoding = self.raw_tokenizer(batch_text, **self.tokenizer_kwargs)
        return encoding.input_ids, encoding.attention_mask

    def add_marker_token(self, input_ids, attention_mask):
        """Add the marker token to the input ids and attention mask."""
        input_ids = insert_constant_column(input_ids,
                                           self.marker_token_position,
                                           self.marker_token_id)
        attention_mask = insert_constant_column(attention_mask,
                                                self.marker_token_position, 1)
        return input_ids, attention_mask

    def process_mask_expansion(self, input_ids, attention_mask):
        """Expand pad tokens with mask expand token."""
        if self.mask_expand_token is not None:
            should_expand = input_ids == self.raw_tokenizer.pad_token_id
            input_ids = input_ids.masked_fill(should_expand,
                                              self.mask_expand_token_id)
            if self.attend_to_mask_tokens:
                attention_mask = attention_mask.masked_fill(should_expand, 1)
                assert attention_mask.sum().item() == attention_mask.size(
                    0) * attention_mask.size(1), attention_mask
        return input_ids, attention_mask

    def get_output_scores_mask(self, input_ids):
        # pad token scores are always zero
        mask = input_ids != self.encoder.config.pad_token_id
        if self.mask_punctuation is True:
            mask[torch.isin(input_ids, self.tokens_to_skip, invert=False)] = 0
        return mask

    def debug_once(self, batch_text, bsize, ids, mask):
        if self.is_used is False:
            self.is_used = True
            print()
            print(
                f"#>{self.__class__.__name__}.tensorize(batch_text, bsize) ==")
            print(f"#>{self.tokenizer_alias}.tensorize(batch_text, bsize) ==")
            print(f"#> Input: {batch_text}, \t\t {bsize}")
            print(f"#> Output IDs: {ids.size()}, {ids}")
            print(f"#> Output Mask: {mask.size()}, {mask}")
            print(f"#> Output Mask sum: {mask.sum(-1)}")
            print()

    def __call__(self, batch_text):
        """IMPORTANT: do not use setattr on this class to add new values,
            we should use the encoding.data dict instead. If we use setattr
            we will have different values for x.input_ids and x['input_ids']
            for example.
        """
        input_ids, attention_mask = self.encode_texts(batch_text)
        input_ids, attention_mask = self.add_marker_token(
            input_ids, attention_mask)
        input_ids, attention_mask = self.process_mask_expansion(
            input_ids, attention_mask)
        output_scores_mask = self.get_output_scores_mask(input_ids)
        self.debug_once(batch_text, None, input_ids, attention_mask)
        # using huggingface tokenizers to avoid having to implement
        batch_encoding = BatchEncoding(
            data=dict(input_ids=input_ids,
                      attention_mask=attention_mask,
                      output_scores_mask=output_scores_mask))
        return batch_encoding


class DocTokenizer(ColbertTokenizer):

    def __init__(self, colbert_config: MyColbertConfig):
        """Tokenizer for document text.

        Args:
            config (ColBERTConfig): The ColBERT configuration.
        """
        self.colbert_config = colbert_config
        super().__init__(
            tokenizer_model_path=colbert_config.pretrained_encoder,
            max_length=colbert_config.doc_maxlen,
            marker_token=colbert_config.doc_token,
            marker_token_position=colbert_config.marker_token_position,
            attend_to_mask_tokens=colbert_config.attend_to_mask_tokens,
            mask_expand_token=None,
            tokenizer_alias='DocTokenizer',
            mask_punctuation=colbert_config.mask_punctuation,
            # this config supposes that will be no mask expansion
            tokenizer_kwargs=dict(padding='longest',
                                  truncation='longest_first',
                                  return_tensors='pt',
                                  max_length=colbert_config.doc_maxlen - 1))

    def tensorize(self, batch_text, bsize=None):
        """This method is kept only for compatibility with original code and
            is not used for training.
        """
        encoding = self(batch_text)
        ids = encoding.input_ids
        mask = encoding.attention_mask

        if bsize is not None:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices
        return ids, mask

    def _sort_by_length(self, input_ids, attention_mask, output_scores_mask,
                        bsize):
        # I modified this function to also account for output_scores_mask
        if self.mask_expand_token is not None:
            raise NotImplementedError(
                'This function cannot be used when there is document '
                'expansion.')
        if input_ids.size(0) <= bsize:
            return input_ids, attention_mask, output_scores_mask, torch.arange(
                input_ids.size(0))

        combined_mask = attention_mask * output_scores_mask
        indices = combined_mask.sum(-1).sort().indices
        reverse_indices = indices.sort().indices

        return (input_ids[indices], attention_mask[indices],
                output_scores_mask[indices], reverse_indices)


class QueryTokenizer(ColbertTokenizer):

    def __init__(self, colbert_config: MyColbertConfig):
        """Tokenizer for query text.

        Args:
            config (ColBERTConfig): The ColBERT configuration.
        """
        self.colbert_config = colbert_config
        super().__init__(
            tokenizer_model_path=colbert_config.pretrained_encoder,
            max_length=colbert_config.query_maxlen,
            marker_token=colbert_config.query_token,
            marker_token_position=colbert_config.marker_token_position,
            attend_to_mask_tokens=colbert_config.attend_to_mask_tokens,
            mask_expand_token=colbert_config.query_expand_token,
            tokenizer_alias='QueryTokenizer',
            mask_punctuation=False,
            tokenizer_kwargs=dict(padding='max_length',
                                  truncation=True,
                                  return_tensors='pt',
                                  return_length=True,
                                  max_length=colbert_config.query_maxlen - 1),
            validate_special_tokens=colbert_config.validate_special_tokens)

    def tensorize(self, batch_text, bsize=None, context=None):
        """This method is kept only for compatibility with original code and
            is not used for training.
        """
        encoding = self(batch_text)
        ids = encoding.input_ids
        mask = encoding.attention_mask
        # context is used only for Baleen
        # will not use this, so no need to implement, but we kept here as a
        # reminder. If you want to use this, you need to implement this after
        # mask expansion, but before attend_to_mask_tokens
        if context is not None:
            raise NotImplementedError('Do not use this, it is only kept for '
                                      'compatibility with original code.')

        if bsize is not None:
            return _split_into_batches(ids, mask, bsize)

        return ids, mask
