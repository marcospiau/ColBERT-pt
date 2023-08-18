from colbert.infra import ColBERTConfig
from colbert.modeling.hf_colbert import class_factory
from colbert.modeling.tokenization.colbert_tokenizer import ColbertTokenizer


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
                         tokenizer_alias='QueryTokenizer')

    def tensorize(self, batch_text, bsize=None, context=None):
        return super().__tensorize(batch_text, bsize, context)
