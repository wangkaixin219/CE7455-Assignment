import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import operator
from queue import PriorityQueue


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1
EOS_char_token = 0


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class EncoderRNN(nn.Module):
    def __init__(self, n_chars, n_words, hidden_size, char_embedding_dim=25, char_representation_dim=25):
        """
        Input parameters:
            n_chars = Number of unique characters in input language
            n_words = Number of unique words in input language
            hidden_size = Dimension of GRU input and output.
            char_embedding_dim = Dimension of the character embeddings
            char_representation_dim = Output dimension from the CNN encoder for character
        """
        super(EncoderRNN, self).__init__()

        # Parameters
        self.n_chars = n_chars
        self.char_embedding_dim = char_embedding_dim
        self.char_representation_dim = char_representation_dim
        self.n_words = n_words
        self.hidden_size = hidden_size

        # Character-level encoder
        self.char_embedding_layer = nn.Embedding(n_chars, char_embedding_dim)
        self.char_cnn3_layer = nn.Conv2d(in_channels=1, out_channels=char_representation_dim, kernel_size=(3, char_embedding_dim), padding=(2, 0))

        # Word embedding layer (Dimension of the word embeddings is automatically derived as hidden_size - char_representation_dim)
        self.word_embedding_layer = nn.Embedding(n_words, hidden_size - char_representation_dim)

        # GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size)

        # Weights
        self.init_embedding(self.char_embedding_layer.weight)

    def forward(self, char_indexes, word_index, hidden):
        # Get char representation
        char_embedding = self.char_embedding_layer(char_indexes).unsqueeze(1)
        char_cnn3 = self.char_cnn3_layer(char_embedding).squeeze(-1).unsqueeze(1)
        char_representation = F.max_pool2d(char_cnn3, kernel_size=(1, char_cnn3.size(-1))).squeeze(-1)

        # Get word embedding
        word_embedding = self.word_embedding_layer(word_index).view(1, 1, -1)

        # Concatenate char representation with word embedding
        combined = torch.cat((char_representation, word_embedding), dim=2)

        # Feed combined and hidden to GRU
        output, hidden = self.gru(combined, hidden)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    @staticmethod
    def init_embedding(input_embedding):
        """
        Initialize embedding
        """
        bias = np.sqrt(3.0 / input_embedding.size(1))
        nn.init.uniform_(input_embedding, -bias, bias)


class BeamSearchNode(object):
    def __init__(self, hidden_state, previous_node, word_id, log_prob, length):
        self.hidden_state = hidden_state
        self.prev_node = previous_node
        self.word_id = word_id
        self.log_p = log_prob
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.log_p / float(self.length - 1 + 1e-6) + alpha * reward


def beam_decode(decoder_hiddens, decoder, encoder_outputs=None):

    beam_width = 5
    top_k = 1  # how many sentences do you want to generate
    decoded_batch = []

    # print(target_tensor.shape)
    # print(decoder_hiddens.shape)
    # print(encoder_outputs.shape)

    # decoding goes sentence by sentence
    for idx in range(decoder_hiddens.size(1)):
        decoder_hidden = decoder_hiddens[:, idx, :]
        encoder_output = encoder_outputs[:, idx, :]

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[SOS_token]]).to(device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((top_k + 1), top_k - len(endnodes))

        # Starting node - hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # Start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # Start beam search
        while True:

            # Give up when decoding takes too long
            if qsize > 2000:
                break

            # Fetch the best node
            score, n = nodes.get()
            decoder_input = n.word_id
            decoder_hidden = n.hidden_state

            if n.word_id.item() == EOS_token and n.prev_node is not None:
                endnodes.append((score, n))
                # If we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # Decode for one step using decoder
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            # print(indexes)
            next_nodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.log_p + log_p, n.length + 1)
                score = -node.eval()
                next_nodes.append((score, node))

            # Put them into queue
            for i in range(len(next_nodes)):
                score, next_node = next_nodes[i]
                nodes.put((score, next_node))
                # increase qsize
            qsize += len(next_nodes) - 1

        # Choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(top_k)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.word_id)
            # back trace
            while n.prev_node is not None:
                n = n.prev_node
                utterance.append(n.word_id)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch

