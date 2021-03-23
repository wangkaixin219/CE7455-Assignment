import argparse
import time
import math
import torch
import torch.nn as nn
import torch.onnx
import data
import model


def batchify(raw_data, batch_size):
    nbatch = raw_data.size(0) // batch_size
    raw_data = raw_data.narrow(0, 0, nbatch * batch_size)
    raw_data = raw_data.view(batch_size, -1).t().contiguous()
    return raw_data.to(device)


def get_batch(source, i):
    seq_len = min(args.window_size, len(source) - 1 - i)
    x = source[i: i + seq_len]
    y = source[i + 1: i + 1 + seq_len].view(-1)
    return x, y


def evaluate(data_source):
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.batch_size):
            inputs, targets = get_batch(data_source, i)
            outputs = model(inputs)
            outputs = outputs.view(-1, n_tokens)
            total_loss += len(inputs) * criterion(outputs, targets).item()

    return total_loss / (len(data_source) - 1)


def train():
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.batch_size)):
        inputs, targets = get_batch(train_data, i)
        outputs = model(inputs)
        outputs = outputs.view(-1, n_tokens)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'
                  .format(epoch, batch, len(train_data) // args.batch_size, optimizer.param_groups[0]['lr'],
                          elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Language Model - Wikitext-2 Token Embedding')
    parser.add_argument('--data', type=str, default='./data/wikitext-2', help='location of the data corpus')
    parser.add_argument('--embed', type=int, default=200, help='size of word embeddings')
    parser.add_argument('--hidden', type=int, default=200, help='number of hidden units per layer')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
    parser.add_argument('--window_size', type=int, default=35, help='sequence length')
    parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # =============== Load device ===============
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    # =============== Load data ===============
    corpus = data.Corpus(args.data)
    n_tokens = len(corpus.dictionary)
    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)
    # print(n_tokens, train_data.size(), val_data.size(), test_data.size())

    # =============== Build the model ===============
    model = model.FNNModel(n_tokens, args.embed, args.hidden, args.tied).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # print(model, criterion, optimizer)

    # =============== Train the model ===============
    try:
        best_val_loss = None
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'
                  .format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # =============== Test the model ===============
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

