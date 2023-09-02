"""
time python3 ../ColBERT-pt/our_scripts/create_index.py \
--checkpoint=checkpoints_path/step-10 \
--collection=debug_data/head_1k_collection.tsv \
--index_name=index_teste
--index_root=indexes
"""

import time
import torch
import random
import torch.nn as nn
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress
import wandb
from pathlib import Path
from colbert.modeling.colbert import ColBERT


# check if are there any checkpoints
def get_last_checkpoint(checkpoints_path):
    checkpoints_path = Path(checkpoints_path)
    # if not exists, nothiung
    if not checkpoints_path.is_dir():
        return
    # use glob to get all files starting with 'checkpoint-'
    steps_and_paths = [(int(f.name.split('-')[1]), f)
                       for f in checkpoints_path.glob('checkpoint-*')]
    if not steps_and_paths:
        return
    last_step_and_path = max(steps_and_paths, key=lambda x: x[0])
    return last_step_and_path

def save_checkpoint_v3(checkpoint_dir, colbert, optimizer=None):
    """Save everything need for resume training and inference"""
    # extract model from a distributed/data-parallel wrapper
    print(f"#> Saving a checkpoint to {checkpoint_dir} ...")
    colbert = colbert.module if hasattr(colbert, 'module') else colbert
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # this saves both model, tokenizer and colbert_config
    colbert.save(str(checkpoint_dir))
    if optimizer is not None:
        torch.save(optimizer.state_dict(),
                   checkpoint_dir / 'optimizer_state.pt')
    return checkpoint_dir


def train(config: ColBERTConfig, triples, queries=None, collection=None):
    # config.checkpoint = config.checkpoint or 'bert-base-uncased'
    wandb_run = wandb.init(**config.wandb, config=vars(config))

    # check if training if finished
    last_step_and_path = get_last_checkpoint(config.checkpoints_path)
    if last_step_and_path is not None and last_step_and_path[
            0] >= config.maxsteps:
        print_message(
            f"#> Training is already finished at step {last_step_and_path[0]}."
        )
        return last_step_and_path[1]

    if config.rank < 1:
        config.help()

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print("Using config.bsize =", config.bsize,
          "(per process) and config.accumsteps =", config.accumsteps)

    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    reader = LazyBatcher(config, triples, queries, collection,
                         (0 if config.rank == -1 else config.rank),
                         config.nranks)

    if last_step_and_path is not None:
        print_message(f"#> Found a checkpoint at {last_step_and_path}.")
        print_message(f"Loading model from {last_step_and_path[1]}")
        colbert = ColBERT(name=str(last_step_and_path[-1]),
                          colbert_config=config)
    colbert = colbert.to(DEVICE)
    colbert.train()

    colbert = torch.nn.parallel.DistributedDataParallel(
        colbert,
        device_ids=[config.rank],
        output_device=config.rank,
        find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()),
                      lr=config.lr,
                      eps=1e-8)
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        # TODO: talvez usar o last_epoch aqui ao invés de dar step manualmente
        print(
            f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps."
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup,
            num_training_steps=config.maxsteps,
            last_epoch=-1)

    # check if there are any checkpoints
    # load optimizer state if exists
    # advance the scheduler and data reader
    if last_step_and_path is not None:
        start_batch_idx = last_step_and_path[0]
        if last_step_and_path[1].joinpath('optimizer_state.pt').exists():
            print(f"#> Loading optimizer state from {last_step_and_path[1]}.")
            optimizer.load_state_dict(
                torch.load(last_step_and_path[1] / 'optimizer_state.pt'))
        if scheduler is not None:
            print(f"#> Advancing scheduler to step {last_step_and_path[0]}.")
            for _ in range(start_batch_idx):
                scheduler.step()
        reader.skip_to_batch(start_batch_idx, config.bsize)
        print_message(
            f"#> Will train for additional {config.maxsteps - start_batch_idx} steps."
        )
    else:
        start_batch_idx = 0

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps),
                                     reader):
        if (warmup_bert is not None) and warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            warmup_bert = None

        this_batch_loss = 0.0

        # limiting the number of print messages
        should_print_batch = (
            batch_idx + start_batch_idx <= 10
            or (batch_idx == 0 or batch_idx % config.stdout_log_every == 0)
            and config.rank < 1)

        for n, batch in enumerate(BatchSteps):
            print('n = ', n)
            print('len(batch) = ', len(batch))
            print('batch = ', batch)
            # sizes
            sizes = dict(enumerate(map(lambda x: len(x), batch)))
            print('sizes = ', sizes)
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
                    # scores here are of the repeated queries
                    scores, ib_loss = scores
                    print('scores.shape', scores.shape)

                scores = scores.view(-1, config.nway)

                if len(target_scores) and not config.ignore_scores:
                    # distillation loss
                    target_scores = torch.tensor(target_scores).view(
                        -1, config.nway).to(DEVICE)
                    target_scores = target_scores * config.distillation_alpha
                    target_scores = torch.nn.functional.log_softmax(
                        target_scores, dim=-1)

                    log_scores = torch.nn.functional.log_softmax(scores,
                                                                 dim=-1)
                    loss = torch.nn.KLDivLoss(reduction='batchmean',
                                              log_target=True)(log_scores,
                                                               target_scores)
                else:
                    loss = nn.CrossEntropyLoss()(scores,
                                                 labels[:scores.size(0)])

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
        train_loss = train_loss_mu * train_loss + (
            1 - train_loss_mu) * this_batch_loss
        logs['train_loss'] = train_loss
        logs['seen_examples'] = (batch_idx + 1) * config.bsize
        logs['seen_batches'] = batch_idx + 1

        amp.step(colbert, optimizer, scheduler)
        logs['learning_rate'] = scheduler.get_lr(
        )[0] if scheduler is not None else config.lr
        logs['batch_idx'] = batch_idx

        # log everything to wandb
        if config.rank < 1:
            wandb.log(logs, step=batch_idx)

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
                or batch_idx == config.maxsteps - 1):
            checkpoint_dir = Path(
                config.checkpoints_path) / f'checkpoint-{batch_idx + 1}'
            print_message(f'#> Saving checkpoint to {checkpoint_dir} ...')
            checkpoint_path = save_checkpoint_v3(checkpoint_dir=checkpoint_dir,
                                                 colbert=colbert,
                                                 optimizer=optimizer)
    # finish wandb logging
    wandb.finish()
    return checkpoint_path


def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
