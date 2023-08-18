from colbert.infra import ColBERTConfig
from colbert.modeling.hf_colbert import class_factory
from colbert.modeling.tokenization.colbert_tokenizer import ColbertTokenizer


class DocTokenizer(ColbertTokenizer):
    def __init__(self, config: ColBERTConfig):
        """Tokenizer for document text.

        Args:
            config (ColBERTConfig): The ColBERT configuration.
        """
        HF_ColBERT = class_factory(config.checkpoint)
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(config.checkpoint)
        self.config = config
        super().__init__(raw_tokenizer=self.tok,
                         max_length=config.doc_maxlen,
                         marker_token=config.doc_token,
                         marker_token_position=config.marker_token_position,
                         attend_to_mask_tokens=config.attend_to_mask_tokens,
                         mask_expand_token=config.doc_expand_token,
                         tokenizer_alias='DocTokenizer')

    def tensorize(self, batch_text, bsize=None):
        return super().__tensorize(batch_text, bsize, sort_by_length=True)
