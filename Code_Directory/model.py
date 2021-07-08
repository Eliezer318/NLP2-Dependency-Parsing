from torch import nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class KiperwasserDependencyParser(nn.Module):
    def __init__(self, lstm_num_layers, word_vocab_size, tag_vocab_size, word_embedding_dim, pos_embedding_dim,
                 lstm_hidden_dim, mlp_hidden_dim, dropout=0.6, *args):
        super(KiperwasserDependencyParser, self).__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.pos_embedding = nn.Embedding(tag_vocab_size, pos_embedding_dim)
        self.hidden_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim
        self.encoder = nn.LSTM(input_size=self.hidden_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_num_layers,
                               bidirectional=True, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(nn.Linear(lstm_hidden_dim * 4, mlp_hidden_dim), nn.Tanh(), nn.Dropout(dropout),
                                 nn.Linear(mlp_hidden_dim, 1))

    def forward(self, word_idx_tensor: torch.Tensor, pos_idx_tensor: torch.Tensor) -> torch.Tensor:
        sentence_vec = torch.cat((self.word_embedding(word_idx_tensor), self.pos_embedding(pos_idx_tensor)), dim=2)
        hiddens, _ = self.encoder(sentence_vec)
        N, sen_len, hidden_dim = hiddens.shape
        combinations = torch.cat((hiddens.repeat_interleave(sen_len, dim=1).to(device),
                                  hiddens.repeat_interleave(sen_len, dim=0).reshape(N, -1, hidden_dim).to(device)), dim=2)
        return self.mlp(combinations.squeeze()).view(sen_len, sen_len)


class BeaffineAttention(nn.Module):
    def __init__(self, lstm_hidden_dim, mlp_hidden_dim):
        super(BeaffineAttention, self).__init__()
        self.head_mlp = nn.Sequential(nn.Linear(lstm_hidden_dim, mlp_hidden_dim), nn.ReLU(), nn.Dropout(0.3))
        self.modifier_mlp = nn.Sequential(nn.Linear(lstm_hidden_dim, mlp_hidden_dim), nn.ReLU(), nn.Dropout(0.3))
        self.U_arc = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)

    def forward(self, lstm_output):
        lstm_output = lstm_output.squeeze(0)
        head = self.head_mlp(lstm_output)
        modifier = self.modifier_mlp(lstm_output)
        return self.U_arc(modifier) @ head.T


class Advanced(nn.Module):
    def __init__(self, lstm_num_layers, word_vocab_size, tag_vocab_size, word_embedding_dim, pos_embedding_dim,
                 lstm_hidden_dim, mlp_hidden_dim, dropout=0.6, *args):
        super(Advanced, self).__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.pos_embedding = nn.Embedding(tag_vocab_size, pos_embedding_dim)
        self.hidden_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim
        self.encoder = nn.LSTM(input_size=self.hidden_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_num_layers,
                               bidirectional=True, batch_first=True, dropout=dropout)
        self.beaffine = BeaffineAttention(2 * lstm_hidden_dim, mlp_hidden_dim)

    def forward(self, word_idx_tensor: torch.Tensor, pos_idx_tensor: torch.Tensor, head=None) -> torch.Tensor:
        """
        :param word_idx_tensor: indices of words in the sentence
        :param pos_idx_tensor: indices of pos tags in the sentence
        :param head: labels, will be None in test time
        :return: scores matrix for when (i, j) in the matrix is score for (modifier, head) edge
        """
        sentence_vec = torch.cat((self.word_embedding(word_idx_tensor), self.pos_embedding(pos_idx_tensor)), dim=2)
        hiddens, _ = self.encoder(sentence_vec)
        res = self.beaffine(hiddens)
        # augmenting loss in training time
        if head is not None:
            res += 1
            res[torch.arange(1, res.shape[0]), head] -= 1
        return res
