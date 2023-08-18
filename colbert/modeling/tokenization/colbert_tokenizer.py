import torch
from colbert.modeling.tokenization.utils import (_sort_by_length,
                                                 _split_into_batches)
from colbert.utils.utils import insert_constant_column


def insert_constant_column(arr, pos, fill_value):
    """Insert a column of `fill_value` at the specified position in a 2D array"""
    begin, end = arr[:, :pos], arr[:, pos:]
    fill = torch.full_like(arr[:, 0], fill_value).unsqueeze(-1)
    return torch.cat([begin, fill, end], dim=-1)


class ColbertTokenizer:
    def __init__(
        self,
        raw_tokenizer,
        max_length: int,
        marker_token: str,
        marker_token_position: int,
        attend_to_mask_tokens: bool = False,
        mask_expand_token: str = None,
        tokenizer_alias: str = 'TokenizerAliasPlaceholder',
    ):
        """Base class for ColBERT tokenizers.

        Args:
            raw_tokenizer (PretrainedTokenizerFast): The raw tokenizer.
            max_length (int): The maximum length of the tokenized text.
            marker_token (str): The marker token.
            marker_token_position (int): The position of the marker token.
                For BERT models this is 1, for T5 models this is 0.
            attend_to_mask_tokens (bool, optional): Whether to attend to mask
                tokens. Defaults to False.
            mask_expand_token (str, optional): The mask expand token. Defaults
                to None. Usually only queries are expanded with mask tokens.
            tokenizer_alias (str, optional): The name of the tokenizer. Defaults
                to 'TokenizerAliasPlaceholder'.
        """
        self.tok = raw_tokenizer
        self.max_length = max_length
        self.marker_token = marker_token
        self.marker_token_position = marker_token_position
        # self.background_maxlen = 512 - self.query_maxlen + 1  # FIXME: Make this configurable
        self.attend_to_mask_tokens = attend_to_mask_tokens
        self.mask_expand_token = mask_expand_token
        self.mask_expand_token_id = self.tok.convert_tokens_to_ids(
            self.mask_expand_token)
        self.tokenizer_alias = tokenizer_alias

        self.marker_token_id = self.tok.convert_tokens_to_ids(
            self.marker_token)
        # self.marker_token_id cannot be None or equal to unk_token_id or pad_token_id
        if (self.marker_token_id is None
                or self.marker_token_id == self.tokenizer.unk_token_id
                or self.marker_token_id == self.tokenizer.pad_token_id):
            raise ValueError(
                f"marker_token_id cannot be None or equal to unk_token_id or pad_token_id. "
            )
        self.is_used = False

    # TODO: break this into smaller functions
    def __tensorize(self,
                    batch_text,
                    bsize=None,
                    context=None,
                    sort_by_length=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # we subtract 1 because we add the [Q]/[D] marker token
        obj = self.tok(batch_text,
                       padding='max_length',
                       truncation=True,
                       return_tensors='pt',
                       max_length=self.max_length - 1)

        # postprocess for the [Q]/[D] marker and the [MASK] augmentation
        ids, mask = obj['input_ids'], obj['attention_mask']
        ids = insert_constant_column(ids, self.marker_token_position,
                                     self.marker_token_id)
        mask = insert_constant_column(mask, self.marker_token_position, 1)
        if self.mask_expand_token is not None:
            ids[ids == self.pad_token_id] = self.mask_expand_token_id

        # context is used only for Baleen
        # will not use this, so no need to implement
        if context is not None:
            raise NotImplementedError

        if self.attend_to_mask_tokens and self.mask_token_id is not None:
            mask[ids == self.mask_token_id] = 1
            assert mask.sum().item() == mask.size(0) * mask.size(1), mask

        if bsize:
            if sort_by_length:
                ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
                batches = _split_into_batches(ids, mask, bsize)
                return batches, reverse_indices
            else:
                batches = _split_into_batches(ids, mask, bsize)
                return batches

        if self.used is False:
            self.used = True

            firstbg = (context is None) or context[0]

            print()
            print("#>self.__class__.__name__.tensorize(batch_text, bsize) ==")
            print(f"#>{self.tokenizer_alias}.tensorize(batch_text, bsize) ==")
            print(f"#> Input: {batch_text[0]}, \t\t {firstbg}, \t\t {bsize}")
            print(f"#> Output IDs: {ids[0].size()}, {ids[0]}")
            print(f"#> Output Mask: {mask[0].size()}, {mask[0]}")
            print()

        return ids, mask
