from data import *
from model import *
from plot import *

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, input_char_indexes, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        # Modify encoder forward pass method to take in character indexes of the input word
        encoder_output, encoder_hidden = encoder(torch.LongTensor([input_char_indexes[ei]]).to(device),
                                                 input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            # Train using DecoderRNN
            # decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            # Train using AttnDecoderRNN
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, target_tensor[di])

            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # Train using DecoderRNN
            # decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            # Train using AttnDecoderRNN
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_epochs(encoder, decoder, input_lang, output_lang, n_epochs=20, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    # Train over n_epochs
    for epoch in range(n_epochs):

        # Print start of epoch
        print('Epoch %d' % epoch)

        # Shuffle train_pairs
        random.shuffle(train_pairs)

        # Get tensors from pair
        training_pairs = [tensors_from_pair(pair, input_lang, output_lang) for pair in train_pairs]

        # Get character indexes
        training_char_indexes = [char_idx_from_sentence(input_char, pair[0]) for pair in train_pairs]

        # Train all train_pairs
        for i in range(1, len(train_pairs) + 1):
            # for i in range(1, 2):
            training_pair = training_pairs[i - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            input_char_indexes = training_char_indexes[i - 1]

            loss = train(input_tensor, target_tensor, input_char_indexes, encoder, decoder, encoder_optimizer,
                         decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if i % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (
                    time_since(start, i / len(train_pairs)), i, i / len(train_pairs) * 100, print_loss_avg)
                )

            if i % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    show_plot(plot_losses)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        input_char_indexes = char_idx_from_sentence(input_char, sentence)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(torch.LongTensor([input_char_indexes[ei]]).to(device),
                                                     input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_hidden = encoder_hidden

        decoded_batch = beam_decode(decoder_hidden.unsqueeze(1), decoder, encoder_outputs.unsqueeze(1))

        return decoded_batch


if __name__ == '__main__':
    print(device)
    input_lang, output_lang, input_char, train_pairs, test_pairs = prepare_data('eng', 'fra', True)

    hidden_size = 256
    char_embedding_dim = 25
    char_representation_dim = 25

    encoder1 = EncoderRNN(input_char.n_chars, input_lang.n_words, hidden_size, char_embedding_dim,
                          char_representation_dim).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    train_epochs(encoder1, attn_decoder1, input_lang, output_lang)

    from nltk.translate.bleu_score import corpus_bleu

    hypotheses = []
    list_of_references = []

    for pair in test_pairs:
        # Evaluate pair
        print('>', pair[0])
        print('=', pair[1])
        decoded_batch = evaluate(encoder1, attn_decoder1, pair[0])
        output_sentence = ' '.join([output_lang.index2word[index.item()] for index in decoded_batch[0][0][1:-1]])
        print('<', output_sentence)
        print('')

        # Append to corpus
        hypotheses.append(output_sentence.split(' '))
        list_of_references.append([pair[1].split(' ')])

    # BLEU-1
    weights = (1.0 / 1.0,)
    score = corpus_bleu(list_of_references, hypotheses, weights)
    print('BLEU-1: %.4f' % score)

    # BLEU-2
    weights = (1.0 / 2.0, 1.0 / 2.0,)
    score = corpus_bleu(list_of_references, hypotheses, weights)
    print('BLEU-2: %.4f' % score)

    # BLEU-3
    weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,)
    score = corpus_bleu(list_of_references, hypotheses, weights)
    print('BLEU-3: %.4f' % score)

    # BLEU-4
    weights = (1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0,)
    score = corpus_bleu(list_of_references, hypotheses, weights)
    print('BLEU-4: %.4f' % score)
