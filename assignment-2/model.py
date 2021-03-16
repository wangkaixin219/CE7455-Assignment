import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import autograd
import numpy as np


def log_sum_exp(vec):
    """
    This function calculates the score explained above for the forward algorithm
    vec 2D: 1 * tagset_size
    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax(vec):
    """
    This function returns the max index in a vector
    """
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def to_scalar(var):
    """
    Function to convert pytorch tensor to a scalar
    """
    return var.view(-1).data.tolist()[0]


class NERModel(nn.Module):

    def __init__(self, word2idx, tag2idx, char2idx, args, pre_word_embeds=None,
                 char_out_dimension=25, char_embedding_dim=25):
        """
        Input parameters:
        vocab_size= Size of vocabulary (int)
        tag_to_ix = Dictionary that maps NER tags to indices
        embedding_dim = Dimension of word embeddings (int)
        hidden_dim = The hidden dimension of the LSTM layer (int)
        char_to_ix = Dictionary that maps characters to indices
        pre_word_embeds = Numpy array which provides mapping from word embeddings to word indices
        char_out_dimension = Output dimension from the CNN encoder for character
        char_embedding_dim = Dimension of the character embeddings
        use_gpu = defines availability of GPU,when True: CUDA function calls are made
        else: Normal CPU function calls are made
        use_crf = parameter which decides if you want to use the CRF layer for output decoding
        """

        super(NERModel, self).__init__()

        # parameter initialization for the model
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.crf = args.crf
        self.cuda = args.cuda
        self.start = args.start
        self.stop = args.stop
        self.char_encode_mode = args.char_encoder
        self.word_encode_mode = args.word_encoder

        self.vocab_size = len(word2idx)
        self.tag_size = len(tag2idx)
        self.tag2idx = tag2idx
        self.out_channels = char_out_dimension
        self.char_lstm_dim = char_out_dimension

        if char_embedding_dim is not None:
            self.char_embedding_dim = char_embedding_dim

            # Initializing the character embedding layer
            self.char_embeds = nn.Embedding(len(char2idx), char_embedding_dim)
            self.init_embedding(self.char_embeds.weight)

            # Performing LSTM/CNN encoding on the character embeddings
            if self.char_encode_mode == 'LSTM':
                self.char_lstm = nn.LSTM(char_embedding_dim, self.char_lstm_dim, num_layers=1, bidirectional=True)
                self.init_lstm(self.char_lstm)
            elif self.char_encode_mode == 'CNN':
                self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3, char_embedding_dim), padding=(2, 0))
            else:
                raise ValueError('char_encoder value error.')

        # Creating Embedding layer with dimension of ( number of words * dimension of each word)
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(pre_word_embeds)
        else:
            self.pre_word_embeds = False

        self.dropout = nn.Dropout(args.dropout)

        # LSTM Layer:
        if self.word_encode_mode == 'LSTM':
            if self.char_encode_mode == 'LSTM':
                self.lstm = nn.LSTM(self.embedding_dim + self.char_lstm_dim * 2, self.hidden_dim, bidirectional=True)
            else:
                self.lstm = nn.LSTM(self.embedding_dim + self.out_channels, self.hidden_dim, bidirectional=True)
            self.init_lstm(self.lstm)
            self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.tag_size)

        # CNN (One-layer):
        elif self.word_encode_mode == 'CNN':
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.hidden_dim * 2, kernel_size=(1, 1), padding=(0, 0))
            nn.init.xavier_uniform_(self.conv1.weight)
            if self.char_encode_mode == 'LSTM':
                self.max_pool1 = nn.MaxPool2d((1, self.embedding_dim + self.char_lstm_dim * 2))
            if self.char_encode_mode == 'CNN':
                self.max_pool1 = nn.MaxPool2d((1, self.embedding_dim + self.out_channels))

        # CNN (Two-layer):
        elif self.word_encode_mode == 'CNN2':
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.hidden_dim * 2, kernel_size=(1, 1), padding=(0, 0))
            self.conv2 = nn.Conv2d(in_channels=self.hidden_dim * 2, out_channels=self.hidden_dim * 2, kernel_size=(1, 1), padding=(0, 0))

            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)

            self.max_pool1 = nn.MaxPool2d((1, 2))
            self.max_pool2 = nn.MaxPool2d((1, (self.embedding_dim + self.out_channels) // 2))

        # CNN (Three-layer):
        elif self.word_encode_mode == 'CNN3':
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.hidden_dim * 2, kernel_size=(1, 1), padding=(0, 0))
            self.conv2 = nn.Conv2d(in_channels=self.hidden_dim * 2, out_channels=self.hidden_dim * 2, kernel_size=(1, 1), padding=(0, 0))
            self.conv3 = nn.Conv2d(in_channels=self.hidden_dim * 2, out_channels=self.hidden_dim * 2, kernel_size=(1, 1), padding=(0, 0))

            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
            nn.init.xavier_uniform_(self.conv3.weight)

            self.max_pool1 = nn.MaxPool2d((1, 2))
            self.max_pool2 = nn.MaxPool2d((1, 2))
            self.max_pool3 = nn.MaxPool2d((1, (self.embedding_dim + self.out_channels) // 4))

        # CNN (Dilated Three-layer):
        elif self.word_encode_mode == 'CNN_DILATED':
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.hidden_dim * 2, kernel_size=(1, 2), padding=(0, 0), dilation=(1, 1))
            self.conv2 = nn.Conv2d(in_channels=self.hidden_dim * 2, out_channels=self.hidden_dim * 2, kernel_size=(1, 2), padding=(0, 0), dilation=(2, 2))
            self.conv3 = nn.Conv2d(in_channels=self.hidden_dim * 2, out_channels=self.hidden_dim * 2, kernel_size=(1, 2), padding=(0, 0), dilation=(3, 3))

            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
            nn.init.xavier_uniform_(self.conv3.weight)

            self.max_pool1 = nn.MaxPool2d((1, 2))
            self.max_pool2 = nn.MaxPool2d((1, 2))
            self.max_pool3 = nn.MaxPool2d((1, 27))

        else:
            raise ValueError('word_encoder value error.')

        self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.tag_size)
        self.init_linear(self.hidden2tag)

        if self.crf:
            self.transitions = nn.Parameter(torch.zeros(self.tag_size, self.tag_size))
            self.transitions.data[tag2idx[self.start], :] = -10000
            self.transitions.data[:, tag2idx[self.stop]] = -10000

    @staticmethod
    def init_embedding(input_embedding):
        """
        Initialize embedding
        """
        bias = np.sqrt(3.0 / input_embedding.size(1))
        nn.init.uniform_(input_embedding, -bias, bias)

    @staticmethod
    def init_linear(input_linear):
        """
        Initialize linear transformation
        """
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform_(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    @staticmethod
    def init_lstm(input_lstm):
        """
        Initialize lstm

        PyTorch weights parameters:

            weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
                of shape `(hidden_size * input_size)` for `k = 0`. Otherwise, the shape is
                `(hidden_size * hidden_size)`

            weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
                of shape `(hidden_size * hidden_size)`
        """
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind))
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)
            weight = eval('input_lstm.weight_hh_l' + str(ind))
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)

        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
                sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
                nn.init.uniform_(weight, -sampling_range, sampling_range)
                weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
                sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
                nn.init.uniform_(weight, -sampling_range, sampling_range)

        if input_lstm.bias:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind))
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                bias = eval('input_lstm.bias_hh_l' + str(ind))
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

            if input_lstm.bidirectional:
                for ind in range(0, input_lstm.num_layers):
                    bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                    bias.data.zero_()
                    bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                    bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                    bias.data.zero_()
                    bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

    def score_sentences(self, feats, tags):
        """
        tags is ground_truth, a list of ints, length is len(sentence)
        feats is a 2D tensor, len(sentence) * tag_size
        """
        r = torch.LongTensor(range(feats.size()[0]))
        if self.cuda:
            r = r.cuda()
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag2idx[self.start]]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag2idx[self.stop]])])
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.tag2idx[self.start]]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag2idx[self.stop]])])

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])
        return score

    def forward_alg(self, feats):
        """
         This function performs the forward algorithm explained above
        """
        init_alphas = torch.Tensor(1, self.tag_size).fill_(-10000.)
        init_alphas[0][self.tag2idx[self.start]] = 0.
        forward_var = autograd.Variable(init_alphas)
        if self.cuda:
            forward_var = forward_var.cuda()

        # Iterate through the sentence
        for feat in feats:
            emit_score = feat.view(-1, 1)
            tag_var = forward_var + self.transitions + emit_score
            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1)

        terminal_var = (forward_var + self.transitions[self.tag2idx[self.stop]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)

        return alpha

    def get_neg_log_likelihood(self, sentence, tags, chars2, chars2_length, d):
        feats = self.get_features(sentence, chars2, chars2_length, d)

        if self.crf:
            forward_score = self.forward_alg(feats)
            gold_score = self.score_sentences(feats, tags)
            return forward_score - gold_score
        else:
            tags = Variable(tags)
            scores = nn.functional.cross_entropy(feats, tags)
            return scores

    def viterbi_alg(self, feats):
        """
        In this function, we implement the viterbi algorithm explained above.
        A Dynamic programming based approach to find the best tag sequence
        """
        back_pointers = []
        # analogous to forward

        # Initialize the viterbi variables in log space
        init_vars = torch.Tensor(1, self.tag_size).fill_(-10000.)
        init_vars[0][self.tag2idx[self.start]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = Variable(init_vars)
        if self.cuda:
            forward_var = forward_var.cuda()
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tag_size, self.tag_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()  # holds the backpointers for this step
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]  # holds the viterbi variables for this step
            viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
            if self.cuda:
                viterbivars_t = viterbivars_t.cuda()

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = viterbivars_t + feat
            back_pointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2idx[self.stop]]
        terminal_var.data[self.tag2idx[self.stop]] = -10000.
        terminal_var.data[self.tag2idx[self.start]] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(back_pointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag2idx[self.start]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def get_features(self, sentence, chars2, chars2_length, d):
        if self.char_encode_mode == 'LSTM':
            chars_embeds = self.char_embeds(chars2).transpose(0, 1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
            lstm_out, _ = self.char_lstm(packed)
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))

            if self.cuda:
                chars_embeds_temp = chars_embeds_temp.cuda()
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = torch.cat(
                    (outputs[i, index - 1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))
            chars_embeds = chars_embeds_temp.clone()
            for i in range(chars_embeds.size(0)):
                chars_embeds[d[i]] = chars_embeds_temp[i]

        elif self.char_encode_mode == 'CNN':
            chars_embeds = self.char_embeds(chars2).unsqueeze(1)
            chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3, kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)

        if self.word_encode_mode == 'LSTM':
            embeds = self.word_embeds(sentence)
            embeds = torch.cat((embeds, chars_embeds), 1)
            embeds = embeds.unsqueeze(1)
            embeds = self.dropout(embeds)
            lstm_out, _ = self.lstm(embeds)
            lstm_out = lstm_out.view(len(sentence), self.hidden_dim * 2)
            lstm_out = self.dropout(lstm_out)
            feats = self.hidden2tag(lstm_out)
        else:
            embeds = self.word_embeds(sentence)
            embeds = torch.cat((embeds, chars_embeds), 1)
            embeds = embeds.unsqueeze(1).unsqueeze(1)
            cnn_out = self.conv1(embeds)
            cnn_out = self.max_pool1(cnn_out)

            if self.word_encode_mode == 'CNN2':
                cnn_out = self.conv2(cnn_out)
                cnn_out = self.max_pool2(cnn_out)

            if self.word_encode_mode == 'CNN3' or self.word_encode_mode == 'CNN_DILATED':
                cnn_out = self.conv2(cnn_out)
                cnn_out = self.max_pool2(cnn_out)
                cnn_out = self.conv3(cnn_out)
                cnn_out = self.max_pool3(cnn_out)

            cnn_out = cnn_out.squeeze(-1).squeeze(-1)
            feats = self.hidden2tag(cnn_out)

        return feats

    def forward(self, sentence, chars, chars2_length, d):
        """
        The function calls viterbi decode and generates the
        most probable sequence of tags for the sentence
        """
        feats = self.get_features(sentence, chars, chars2_length, d)

        if self.crf:
            score, tag_seq = self.viterbi_alg(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq
