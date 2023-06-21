import time
import torch
import random
import torch.nn as nn
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints
import os
import wandb
from pathlib import Path


def save_checkpoint_v2(colbert, optimizer, batch_idx, checkpoints_path):
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
    path_save = os.path.join(checkpoints_path, f"checkpoint-{batch_idx}.pt")
    print(f"#> Saving a checkpoint to {path_save} ..")
    # save model state
    # colbert.save(path_save)
    # save training state
    checkpoint = {}
    checkpoint['batch'] = batch_idx
    checkpoint['model_state_dict'] = colbert.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, path_save)
    return path_save


def train(config: ColBERTConfig, triples, queries=None, collection=None):
    config.checkpoint = config.checkpoint or 'bert-base-uncased'
    wandb_run = wandb.init(**config.wandb.val, config=vars(config))

    if config.rank < 1:
        config.help()

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)


    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
        else:
            reader = LazyBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
    else:
        raise NotImplementedError()

    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    colbert.train()

    # colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
    #                                                     output_device=config.rank,
    #                                                     find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=config.maxsteps)

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0

    # WHY IS THIS COMMENTED?
    # if config.resume:
    #     assert config.checkpoint is not None
    #     start_batch_idx = checkpoint['batch']

    #     reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])
    seen_examples = 0
    seen_batches = 0

    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        seen_examples += config.bsize
        seen_batches += 1
        if (warmup_bert is not None) and warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            warmup_bert = None

        this_batch_loss = 0.0

        # limiting the number of print messages
        should_print_batch = (
            batch_idx <= 10 or 
            (batch_idx == 0 or batch_idx % config.stdout_log_every == 0) and
            config.rank < 1)

        for batch in BatchSteps:
            logs = {}
            with amp.context():
                try:
                    queries, passages, target_scores = batch
                    encoding = [queries, passages]
                except:
                    encoding, target_scores = batch
                    encoding = [encoding.to(DEVICE)]

                scores = colbert(*encoding)

                if config.use_ib_negatives:
                    scores, ib_loss = scores

                scores = scores.view(-1, config.nway)

                if len(target_scores) and not config.ignore_scores:
                    target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    target_scores = target_scores * config.distillation_alpha
                    target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
                else:
                    loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])

                logs['loss'] = loss.item()
                logs['total_loss'] = loss.item()
                if config.use_ib_negatives:
                    logs['ib_loss'] = ib_loss.item()
                    logs['total_loss'] += ib_loss.item()

                if config.use_ib_negatives and should_print_batch:
                    print('\t\t\t\t', loss.item(), ib_loss.item())

                    loss += ib_loss

                loss = loss / config.accumsteps
                logs['loss_div_accumsteps'] = logs['loss'] / config.accumsteps

            if should_print_batch:
                print_progress(scores, batch_idx=batch_idx)

            amp.backward(loss)

            this_batch_loss += loss.item()

        train_loss = this_batch_loss if train_loss is None else train_loss
        train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss
        logs['train_loss'] = train_loss
        logs['train_loss_mu'] = train_loss_mu
        logs['seen_examples'] = seen_examples
        logs['seen_batches'] = seen_batches

        amp.step(colbert, optimizer, scheduler)
        logs['learning_rate'] = scheduler.get_lr()[0] if scheduler is not None else config.lr
        logs['batch_idx'] = batch_idx

        # log everything to wandb
        if config.rank < 1:
            wandb.log(logs)

        # if config.rank < 1:
        if should_print_batch:
            print_message(batch_idx, train_loss)
        # CHECAR ESSE CARA
        if config.rank < 1 and (
            # first step
            batch_idx == 0
            # during training
            or batch_idx % config.save_every == 0
            # last step
            or batch_idx == config.maxsteps - 1
        ):
            print_message(f'#> Saving checkpoint... batch_idx = {batch_idx}')
            save_checkpoint_v2(
                colbert=colbert,
                optimizer=optimizer,
                batch_idx=batch_idx,
                checkpoints_path=config.checkpoints_path)
    # finish wandb logging
    wandb.finish()


def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
