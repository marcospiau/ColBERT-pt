import cytoolz as toolz
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import PretrainedConfig, PreTrainedModel


class ColbertConfig(PretrainedConfig):
    def __init__(self,
                 dim: int = 128,
                 doc_maxlen: int = 300,
                 mask_punctuation: bool = True,
                 query_maxlen: int = 32,
                 attend_to_mask_tokens: bool = False,
                 nway: int = 2,
                 encoder_class_str: str = 'T5EncoderModel',
                 pretrained_encoder: str = 'unicamp-dl/ptt5-small-portuguese-vocab',
                 query_token: str = '<extra_id_0>',
                 doc_token: str = '<extra_id_1>',
                 query_expand_token: str = '<extra_id_2>',
                 validate_special_tokens: bool = True,
                 **kwargs):
        self.dim = dim
        self.doc_maxlen = doc_maxlen
        self.pretrained_encoder = pretrained_encoder
        self.mask_punctuation = mask_punctuation
        self.query_maxlen = query_maxlen
        self.attend_to_mask_tokens = attend_to_mask_tokens
        self.query_token = query_token
        self.doc_token = doc_token
        self.query_expand_token = query_expand_token
        self.validate_special_tokens = validate_special_tokens
        self.nway = nway
        self.encoder_class_str = encoder_class_str
        super().__init__(**kwargs)


class ColbertModel(PreTrainedModel):
    config_class = ColbertConfig

    _no_split_modules = ['linear']

    def __init__(self, config: ColbertConfig, **kwargs):
        super().__init__(config)
        encoder_cls = getattr(transformers, config.encoder_class_str)
        self.encoder = encoder_cls.from_pretrained(config.pretrained_encoder,
                                                   **kwargs)
        self.linear = nn.Linear(self.encoder.config.hidden_size,
                                config.dim,
                                bias=False,
                                device=self.encoder.device,
                                dtype=self.encoder.dtype)
        # init weights for linear layer
        # nn.init.normal_(self.linear.weight, std=0.02)
        # TODO: check how to init weights for linear layer
        self.eval()

    # def mask(self, input_ids, punctuation_mask=None):
    #     # TODO: check the dtype of this mask (float vs half)
    #     mask = input_ids != self.encoder.config.pad_token_id
    #     if punctuation_mask is not None:
    #         mask = mask * punctuation_mask
    #     mask = mask.unsqueeze(2)
    #     return mask

    def forward_reps(self, input_ids, attention_mask, output_scores_mask):
        """Forward token representations"""
        x = self.encoder(input_ids,
                         attention_mask=attention_mask).last_hidden_state
        x = self.linear(x)
        x =  x * output_scores_mask.unsqueeze(2).float()
        x = F.normalize(x, p=2, dim=2, eps=1e-12)
        # default value of eps is 1e-12 can cause nan values for fp16
        # https://github.com/pytorch/pytorch/issues/32137
        # https://github.com/pytorch/pytorch/pull/33596
        # https://github.com/pytorch/pytorch/issues/41527
        # x = F.normalize(x.float(), p=2, dim=2, eps=1e-12).to(x.dtype)
        # ValueError: Attempting to unscale FP16 gradients.
        # https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372/2
        return x, output_scores_mask


    def query(self, input_ids, attention_mask, output_scores_mask):
        Q, _ = self(input_ids, attention_mask, output_scores_mask)
        return Q
        Q = self.encoder(input_ids,
                         attention_mask=attention_mask).last_hidden_state
        Q = self.linear(x)
        Q = F.normalize(Q, p=2, dim=2, eps=1e-12)
        return Q

    def doc(self, input_ids, attention_mask, output_scores_mask=1):
        """Forward pass for documents.

        Args:
            input_ids (torch.Tensor): input ids, shape (bsize, doc_len)
            attention_mask (torch.Tensor): attention mask, shape
                (bsize, doc_len)
            output_scores_mask (int, optional): mask to zero out score for 
                punctuation tokens. Defaults to 1 (no removal).
        Returns:
            tuple: document embeddings and mask. Mask will be used to compute
                ColBERT score.
        """
        D, mask = self(input_ids, attention_mask, output_scores_mask)
        # TODO: fazer o cast pra float16 aqui
        # D = torch.nn.functional.normalize(D, p=2, dim=2)
        # if self.use_gpu:
        #     D = D.half()
        return D, mask
    
    # https://huggingface.co/sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco
    def forward(self, query_encodings, doc_encodings):
        """Forward pass for queries, computes scores"""
        Q = self.query(**query_encodings)
        D, D_mask = self.doc(**doc_encodings)

    # def forward(self, input_ids, attention_mask, output_scores_mask=1):
    #     """Forward pass common to both query and doc"""
    #     x = self.encoder(input_ids,
    #                      attention_mask=attention_mask).last_hidden_state
    #     x = self.linear(x)
    #     x = x * mask
    #     mask = mask.bool()
    #     x = F.normalize(x, p=2, dim=2, eps=1e-12)
    #     return x, mask



# Q = torch.rand(1, 32, 384)
# Q = torch.nn.functional.normalize(Q, p=2, dim=2)
# Q_mask = torch.ones(1, 32).bool()
# D = torch.rand(8, 128, 384)
# D = torch.nn.functional.normalize(D, p=2, dim=2)
# D_mask = torch.ones(8, 128).bool()

def compute_token_level_scores(Q,  D):
    scores = D @ Q.permute(0, 2, 1)
    return scores

def reduce_token_scores(token_scores, Q_mask, D_mask):
    # mask out padding on the doc dimension (mask by -1000, because max should
    # not select those, setting it to 0 might select them)
    D_padding = ~D_mask.view(token_scores.size(0),
                             token_scores.size(1)).bool()
    token_scores[D_padding] = float('-inf')
    # max pooling over document dimension
    scores = token_scores.max(1).values
    # return scores

    # mask out paddding query values
    # scores[Q_mask] = 0 # PRECISA ARRUMAR
    # scores = scores * Q_mask.unsqueeze(1).float()  # torch.Size([1, 8, 32])
    # TALVEZ Não precise dessa parte aqui, porque output scores de query já ficam zerados se a mask zerar antes
    scores = scores * Q_mask.float() ## torch.Size([8])
    scores = scores.sum(-1)
    return scores

def colbert_scores_complete(Q, D, Q_mask, D_mask):
    scores = compute_token_level_scores(Q, D)
    scores = reduce_token_scores(scores, Q_mask, D_mask)
    return scores

scores1 = compute_token_level_scores(Q,  D); scores1.size()
# torch.Size([8, 128, 32])
scores2 = reduce_token_scores(scores1, Q_mask, D_mask); scores2.size()

scores3 = colbert_scores_complete(Q, D, Q_mask, D_mask); scores3.size()
assert scores2.allclose(scores3)






# TODO: check docstrings
def colbert_score_reduce(scores_padded, D_mask):
    """Reduce scores to compute ColBERT score.

    Args:
        scores_padded (torch.Tensor): scores, shape (bsize, doc_len, query_len)
        D_mask (torch.Tensor): document mask, shape (bsize, doc_len)

    Returns:
        torch.Tensor: scores, shape (bsize, )
    """
    D_padding = ~D_mask.view(scores_padded.size(0),
                             scores_padded.size(1)).bool()
    scores_padded[D_padding] = float('-inf')
    scores = scores_padded.max(1).values
    scores = scores.sum(-1)
    return scores


# TODO: check docstrings
def colbert_score(D, Q, D_mask):
    """Compute ColBERT score.

    Args:
        D (torch.Tensor): document embeddings, shape (bsize, doc_len, dim)
        Q (torch.Tensor): query embeddings, shape (bsize, query_len, dim)
        D_mask (torch.Tensor): document mask, shape (bsize, doc_len)

    Returns:
        torch.Tensor: scores, shape (bsize, )
    """
    scores = D @ Q.to(dtype=D.dtype).permute(0, 2, 1)
    return colbert_score_reduce(scores, D_mask)


def distilation_loss(scores, target_scores, nway, distillation_alpha=1.0):
    """Compute distillation loss.

    Args:
        scores (torch.Tensor): scores from the model, shape (bsize, nway)
        target_scores (torch.Tensor): scores from the teacher model, shape
            (bsize, nway). This shape is not sctricly enforced because data
            will be reshaped to (bsize, nway).
        nway (int): number of documents per query
        distillation_alpha (float): distillation alpha parameter

    Returns:
        torch.Tensor: distillation loss
    """
    target_scores = target_scores.view(-1, nway) * distillation_alpha
    target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)
    scores = scores.view(-1, nway)
    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
    loss = torch.nn.KLDivLoss(reduction='batchmean',
                              log_target=True)(log_scores, target_scores)
    return loss


def compute_distillation_loss(Q, D_padded, D_mask, target_scores, nway,
                              distillation_alpha):
    """Duplicate queries and compute distillation loss."""
    # distillation loss is computed on the duplicated queries
    Q_duplicated = Q.repeat_interleave(nway, dim=0).contiguous()
    scores_duplicated = colbert_score(D=D_padded,
                                      Q=Q_duplicated,
                                      D_mask=D_mask)
    loss = distilation_loss(scores_duplicated, target_scores, nway,
                            distillation_alpha)
    return loss


def compute_ib_loss(Q, D, D_mask, nway):
    """Compute in batch negatives loss."""
    # TODO: Organize the code below! Quite messy.
    scores = (D.unsqueeze(0) @ Q.permute(0, 2, 1).unsqueeze(1)).flatten(
        0, 1)  # query-major unsqueeze

    scores = colbert_score_reduce(scores, D_mask.repeat(Q.size(0), 1, 1))

    all_except_self_negatives = [
        list(range(qidx * D.size(0),
                   qidx * D.size(0) + nway * qidx + 1)) + list(
                       range(qidx * D.size(0) + nway * (qidx + 1),
                             qidx * D.size(0) + D.size(0)))
        for qidx in range(Q.size(0))
    ]
    all_except_self_negatives = list(toolz.concat(all_except_self_negatives))
    scores = scores[all_except_self_negatives]
    scores = scores.view(Q.size(0),
                         -1)  # D.size(0) - self.colbert_config.nway + 1)

    labels = torch.arange(0, Q.size(0), device=scores.device) * nway

    return torch.nn.CrossEntropyLoss()(scores, labels)


def compute_loss(batch, colbert, distillation_alpha=1.0):
    """Compute the loss for a batch of queries and documents.

    Args:
        batch (dict): batch of queries and documents. Shouold contain the
            following keys:
            - queries: queries input_ids and attention_mask
            - docs: documents input_ids, attention_mask and punctuation_mask
            - target_scores: scores from the teacher model
            This batch is generated by the TrainTriplesDataset.
        colbert (MyColbert): ColBERT model
        distillation_alpha (float): distillation alpha parameter

    Returns:
        tuple: distillation loss and in batch negatives loss
    """
    D_padded, D_mask = colbert.doc(**batch['docs'])
    Q = colbert.query(**batch['queries'])
    # distillation loss
    nway = colbert.config.nway
    distill_loss = compute_distillation_loss(
        Q=Q,
        D_padded=D_padded,
        D_mask=D_mask,
        target_scores=batch['target_scores'],
        nway=nway,
        distillation_alpha=distillation_alpha)
    # in batch negatives loss
    ib_loss = compute_ib_loss(Q=Q, D=D_padded, D_mask=D_mask, nway=nway)
    return distill_loss, ib_loss
