import argparse
import torch
import data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Language Model - Wikitext-2 Text Generation')
    parser.add_argument('--data', type=str, default='./data/wikitext-2', help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='./model.pt', help='model checkpoint to use')
    parser.add_argument('--outfile', type=str, default='generated.txt', help='output file for generated text')
    parser.add_argument('--words', type=int, default='1000', help='number of words to generate')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature - higher will increase diversity')
    parser.add_argument('--log-interval', type=int, default=100, help='reporting interval')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3")

    # =============== Load device ===============
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    # =============== Load model ===============
    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f).to(device)
    model.eval()

    # =============== Load data ===============
    corpus = data.Corpus(args.data)
    n_tokens = len(corpus.dictionary)

    # =============== Generate texts ===============
    with open(args.outfile, 'w') as outfile:
        start = torch.randint(n_tokens, (1, 1), dtype=torch.long).to(device)
        with torch.no_grad():
            for i in range(args.words):
                output = model(start)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                start.fill_(word_idx)
                word = corpus.dictionary.idx2word[word_idx]

                outfile.write(word + ('\n' if i % 20 == 19 else ' '))
                if i % args.log_interval == 0:
                    print('| Generated {}/{} words'.format(i, args.words))
