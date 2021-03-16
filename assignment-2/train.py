import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import numpy as np
from matplotlib import pyplot as plt


def get_chunk_type(tok, idx_to_tag):
    """
    The function takes in a chunk ("B-PER") and then splits it into the tag (PER) and its class (B)
    as defined in BIOES
    """
    if type(tok) == torch.Tensor:
        tok = tok.item()

    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """
    Given a sequence of tags, group entities and their position
    """
    default = tags["O"]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None

    for i, tok in enumerate(seq):
        if tok == default and chunk_type is not None:
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def evaluating(model, dataset, tag2idx, best_f1, args, dataset_type='train'):
    """
    The function takes as input the model, data and calcuates F-1 Score. It performs conditional updates
     1) Flag to save the model
     2) Best F1 score, if the F1 score calculated improves on the previous F1 score
    """
    save = False
    correct_pred, total_correct, total_pred = 0., 0., 0.

    for data in dataset:
        ground_truth_id = data['tags']
        chars2 = data['chars']

        if args.char_encoder == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda x: len(x), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))
        else:
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))

        if args.cuda:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, chars2_length, d)
        predicted_id = out

        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks = set(get_chunks(ground_truth_id, tag2idx))
        lab_pred_chunks = set(get_chunks(predicted_id, tag2idx))

        # Updating the count variables
        correct_pred += len(lab_chunks & lab_pred_chunks)
        total_pred += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    # Calculating the F1-Score
    p = correct_pred / total_pred if correct_pred > 0 else 0
    r = correct_pred / total_correct if correct_pred > 0 else 0
    new_f1 = 2 * p * r / (p + r) if correct_pred > 0 else 0
    new_acc = p

    print("{}: new_F: {} best_F: {} new_acc:{} ".format(dataset_type, new_f1, best_f1, p))

    if new_f1 > best_f1:
        best_f1 = new_f1
        save = True

    return best_f1, new_f1, new_acc, save


def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if metrics is None:
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


def create_and_train_model(model, train_dataset, dev_dataset, test_dataset, tag2idx, args):

    print(f"\nChar mode: {args.char_encoder}, Encoder mode: {args.word_encoder}")

    learning_rate = args.lr
    number_of_epochs = args.epoch
    momentum = args.momentum
    decay_rate = args.decay_rate
    gradient_clip = args.clip
    plot_interval = args.plot_interval
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    losses = []
    loss = 0.0
    best_dev_f1 = -1.0
    best_test_f1 = -1.0
    best_train_f1 = -1.0
    all_f1 = [[0, 0, 0]]
    all_acc = [[0, 0, 0]]
    eval_interval = len(train_dataset)
    count = 0

    es = EarlyStopping(patience=10)

    start = time.time()
    model.train(True)
    for epoch in range(1, number_of_epochs):
        print(f'Epoch {epoch}:')
        for i, index in enumerate(np.random.permutation(len(train_dataset))):
            count += 1
            data = train_dataset[index]

            model.zero_grad()

            sentence_in = data['words']
            sentence_in = Variable(torch.LongTensor(sentence_in))
            tags = data['tags']
            chars2 = data['chars']

            if args.char_encoder == 'LSTM':
                chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
                d = {}
                for j, cj in enumerate(chars2):
                    for k, ck in enumerate(chars2_sorted):
                        if cj == ck and not k in d and not j in d.values():
                            d[k] = j
                            continue
                chars2_length = [len(c) for c in chars2_sorted]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
                for i, c in enumerate(chars2_sorted):
                    chars2_mask[i, :chars2_length[i]] = c
                chars2_mask = Variable(torch.LongTensor(chars2_mask))
            else:
                d = {}
                chars2_length = [len(c) for c in chars2]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
                for j, c in enumerate(chars2):
                    chars2_mask[j, :chars2_length[j]] = c
                chars2_mask = Variable(torch.LongTensor(chars2_mask))

            targets = torch.LongTensor(tags)

            if args.cuda:
                neg_log_likelihood = model.get_neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(), chars2_length, d)
            else:
                neg_log_likelihood = model.get_neg_log_likelihood(sentence_in, targets, chars2_mask, chars2_length, d)

            loss += neg_log_likelihood.item() / len(data['words'])
            neg_log_likelihood.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            if count % plot_interval == 0:
                loss /= plot_interval
                print(count, ': ', loss)
                if not losses:
                    losses.append(loss)
                losses.append(loss)
                loss = 0.0

            if count % eval_interval == 0 and count > (eval_interval * 20) or \
                    count % (eval_interval * 4) == 0 and count < (eval_interval * 20):
                print(f'Evaluating on Train, Test, Dev Sets at count={count}')
                model.train(False)
                best_train_f1, new_train_f1, new_train_acc, _ = evaluating(model, train_dataset, tag2idx, best_train_f1, args, dataset_type='train')
                best_dev_f1, new_dev_f1, new_dev_acc, save = evaluating(model, dev_dataset, tag2idx, best_dev_f1, args, dataset_type='dev')
                if save or count == eval_interval:
                    print("Saving Model to ", args.model_dir + args.save_path)
                    with open(args.model_dir + args.char_encoder + '-' + args.word_encoder + '-' + args.save_path, 'wb') as f:
                        torch.save(model, f)
                best_test_f1, new_test_f1, new_test_acc, _ = evaluating(model, test_dataset, tag2idx, best_test_f1, args, dataset_type='test')

                all_f1.append([new_train_f1, new_dev_f1, new_test_f1])
                all_acc.append([new_train_acc, new_dev_acc, new_test_acc])

                model.train(True)

            # Performing decay on the learning rate
            if count % len(train_dataset) == 0:
                adjust_learning_rate(optimizer, lr=learning_rate / (1 + decay_rate * count / len(train_dataset)))

        if es.step(all_acc[-1][1]):
            print(f'Early stopping: epoch={epoch}, count={count}, new_acc_F={all_acc[-1][1]}')
            break

    print(f'{(time.time() - start) / 60} minutes')

    plt.plot(losses)
    fig = plt.gcf()
    fig.savefig('figs/loss-' + args.char_encoder + '-' + args.word_encoder + '.eps', format='eps', dpi=1000)
    plt.show()

    return all_f1, all_acc
