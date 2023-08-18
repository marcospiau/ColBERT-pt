from colbert.infra import ColBERTConfig
from colbert.modeling.hf_colbert import class_factory
from colbert.modeling.tokenization.colbert_tokenizer import ColbertTokenizer
from colbert.modeling.tokenization.utils import (_sort_by_length,
                                                 _split_into_batches)


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
                         tokenizer_alias='DocTokenizer',
                         tokenizer_kwargs=dict(padding='longest',
                                               truncation='longest_first',
                                               return_tensors='pt',
                                               max_length=self.max_length - 1))

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))
        ids, mask = self.encode_texts(batch_text)
        ids, mask = self.add_marker_token(ids, mask)
        ids, mask = self.process_mask_expansion(ids)

        if bsize is not None:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask
