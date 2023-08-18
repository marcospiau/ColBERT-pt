from colbert.infra import ColBERTConfig
from colbert.modeling.hf_colbert import class_factory
from colbert.modeling.tokenization.colbert_tokenizer import ColbertTokenizer
from colbert.modeling.tokenization.utils import _split_into_batches


class QueryTokenizer(ColbertTokenizer):
    def __init__(self, config: ColBERTConfig):
        """Tokenizer for query text.

        Args:
            config (ColBERTConfig): The ColBERT configuration.
        """
        HF_ColBERT = class_factory(config.checkpoint)
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(config.checkpoint)
        self.config = config
        super().__init__(raw_tokenizer=self.tok,
                         max_length=config.query_maxlen,
                         marker_token=config.query_token,
                         marker_token_position=config.marker_token_position,
                         attend_to_mask_tokens=config.attend_to_mask_tokens,
                         mask_expand_token=config.query_expand_token,
                         tokenizer_alias='QueryTokenizer',
                         tokenizer_kwargs=dict(padding='max_length',
                                               truncation=True,
                                               return_tensors='pt',
                                               max_length=config.max_length -
                                               1))

    def tensorize(self, batch_text, bsize=None, context=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))
        ids, mask = self.encode_texts(batch_text)
        ids, mask = self.add_marker_token(ids, mask)
        ids, mask = self.process_mask_expansion(ids)
        # context is used only for Baleen
        # will not use this, so no need to implement, but we kept here as a
        # reminder. If you want to use this, you need to implement this after mask
        # expansion, but before attend_to_mask_tokens
        if context is not None:
            raise NotImplementedError

        if bsize is not None:
            return _split_into_batches(ids, mask, bsize)

        self.debug_once(batch_text, bsize, ids, mask)

        return ids, mask
