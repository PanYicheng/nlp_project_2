import argparse
import sys
import time
import os
import math

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from adaptive_softmax.adaptive_softmax import AdaptiveLoss
import adaptive_softmax.model as model
from utils.corpus import Corpus

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    help='location of the data corpus')
parser.add_argument('--dic', type=str,
                    help='path to dictionary pickle')
parser.add_argument('--old', type=str, default=None,
                    help='old model to keep training')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, GRU)')
parser.add_argument('--emsize', type=int, default=1024,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1024,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--cutoffs', nargs='+', type=int,
                    help='cutoffs for buckets in adaptive softmax')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--ar', type=float, default=0.9,
                    help='learning rate annealing rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=1024, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
# Hardware
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gpu', type=int,  default=0,
                    help='gpu to use')
args = parser.parse_args()

# Create save directory if it not exists
if not os.path.exists(os.path.dirname(os.path.realpath(args.save))):
    os.makedirs(os.path.dirname(os.path.realpath(args.save)))

# default `log_dir` is "runs" - we'll be more specific here
tb_writer = SummaryWriter(os.path.join('runs/',
                                       os.path.basename(args.save)))

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda", args.gpu)
else:
    device = torch.device("cpu")

corpus = Corpus(args.data, args.dic)
ntokens = len(corpus.dictionary.idx2word)
cutoffs = args.cutoffs + [ntokens]
print('ntokens:', ntokens)
print('cutoffs:', cutoffs)

if args.old is None:
    model = model.RNNModel(args.model, ntokens, args.emsize,
                           args.nhid, args.nlayers, cutoffs, args.dropout, args.tied)
else:
    with open(args.old, 'rb') as model_file:
        model = torch.load(model_file)
if args.cuda:
    model = model.to(device)

    criterion = AdaptiveLoss(cutoffs)
    # optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr,
    #                              weight_decay=1e-3, t0=10)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

###############################################################################
# Training code
###############################################################################

# Loop over epochs.
global lr, best_val_loss, global_step
lr = args.lr
best_val_loss = None


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def repackage_hidden(h):
    """Detaches hidden states from their history"""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(split):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss, nbatches = 0, 0
    ntokens = len(corpus.dictionary.idx2word)
    with torch.no_grad():
        hidden = model.init_hidden(args.eval_batch_size)
        for source, target in corpus.iter(split, args.eval_batch_size, args.bptt,
                                          use_cuda=args.cuda, device=device):
            model.softmax.set_target(target.data.view(-1))
            output, hidden = model(source, hidden)
            total_loss += criterion(output, target.view(-1)).data.sum().item()
            hidden = repackage_hidden(hidden)
            nbatches += 1
    return total_loss / nbatches


def train():
    global lr, best_val_loss, global_step
    # Turn on training mode which enables dropout.
    model.train()
    start_time = time.time()
    total_loss, nbatches = 0, 0
    ntokens = len(corpus.dictionary.idx2word)
    hidden = model.init_hidden(args.batch_size)
    for b, batch in enumerate(corpus.iter('train', args.batch_size, args.bptt,
                                          use_cuda=args.cuda, device=device)):
        model.train()
        source, target = batch
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        model.softmax.set_target(target.view(-1))
        output, hidden = model(source, hidden)
        loss = criterion(output, target.view(-1))
        loss.backward()

        # `clip_grad_norm_` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # update the model parameters
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(-lr, p.grad.data)
        # optimizer.step()

        total_loss += loss.data.cpu()

        if b % args.log_interval == 0 and b > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            val_loss = evaluate('valid')
            # scheduler.step(val_loss)
            print('| epoch {:3d} | batch {:5d} | lr {:05.4e} | ms/batch {:3.2e} '
                  '| loss {:5.4e} | ppl {:5.4e} '
                  '| valid loss {:5.4e} | valid ppl {:5.4e}'
                  .format(
                      epoch, b, lr,
                      elapsed * 1000 / args.log_interval,
                      cur_loss, math.exp(cur_loss),
                      val_loss, math.exp(val_loss)))
            # add log to tensorboard scalar
            tb_writer.add_scalar('cur_loss', cur_loss, global_step)
            tb_writer.add_scalar('val_loss', val_loss, global_step)
            global_step += 1

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                pass
                lr *= args.ar
                # optimizer.lr = optimizer.lr * args.ar

            total_loss = 0
            start_time = time.time()


# At any point you can hit Ctrl + C to break out of training early.
try:
    # reset global setp for the summary writer
    global_step = 0
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate('valid')
        print('{:-^80}'.format(''))
        print('| end of epoch {:3d} | time: {:4.3e}s '
              '| valid loss {:5.4e} |valid ppl {:5.4e}'
              .format(epoch, (time.time() - epoch_start_time),
                      val_loss, math.exp(val_loss)))
        print('{:-^80}'.format(''))
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate('test')
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
