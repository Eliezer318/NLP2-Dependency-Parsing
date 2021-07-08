import torch
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset

import re
from collections import defaultdict
from collections import Counter
from typing import Tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"  # Optional: this is used to pad a batch of sentences in different lengths.
ROOT_TOKEN = "<root>"  # use this if you are padding your batches and want a special token for ROOT
SPECIAL_TOKENS = [ROOT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]


def get_vocabs(files_path, word_dict, pos_dict):
    for file in files_path:
        with open(file, 'r') as f:
            for line in f:
                if line == '\n': continue
                splited = re.split('\t|\n', line)[:-1]
                token_counter, token, token_pos, token_head = splited[0], splited[1], splited[3], splited[6]
                word_dict[token] += 1
                pos_dict[token_pos] += 1
        return word_dict, pos_dict


class PosDataReader:
    def __init__(self, file, validation_path: str = None):
        self.files = [file, validation_path] if validation_path else [file]
        self.word_dict = defaultdict(int)
        self.pos_dict = defaultdict(int)
        get_vocabs(self.files, self.word_dict, self.pos_dict)
        self.sentences = [[]]
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        for file in self.files:
            with open(file, 'r') as f:
                for line in f:
                    if line != '\n':
                        splited = re.split('\n|\t', line)
                        token_counter, token, token_pos, token_head = splited[0], splited[1], splited[3], splited[6]
                        if token_head == '_':
                            token_head = -1
                        self.sentences[-1].append((token, token_pos, int(token_counter), int(token_head)))
                    else:
                        self.sentences.append([])
        self.sentences = self.sentences[:-1]

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


class PosDataset(Dataset):
    def __init__(self, file_path: str, validation_path: str = None, padding=False, word_embeddings=None, pos_embeddings=None):
        super().__init__()
        self.subset = file_path.split('.')[-2].split('/')[-1]  # One of the following: [train, test]
        self.datareader = PosDataReader(file_path, validation_path)
        self.vocab_size = len(self.datareader.word_dict)
        if word_embeddings:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = word_embeddings
            self.pos_idx_mappings, self.idx_pos_mappings = pos_embeddings
        else:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(self.datareader.word_dict)
            self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(self.datareader.pos_dict)

        self.pad_idx = self.word_idx_mappings.get(PAD_TOKEN)
        self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        self.unknown_pos_idx = self.pos_idx_mappings.get(UNKNOWN_TOKEN)
        self.root_idx = self.word_idx_mappings.get(ROOT_TOKEN)

        self.sentence_lens = [len(sentence) for sentence in self.datareader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)
        self.training = True

    def __len__(self):
        return len(self.sentences_dataset)

    def sentence_dropout(self, word_idx_tensor: torch.Tensor, frequencies: torch.Tensor):  # TODO make sure it is not happening in evaluation
        if self.training:
            save = word_idx_tensor[0]
            word_idx_tensor[torch.rand(word_idx_tensor.shape) <= (0.25 / frequencies + 0.25)] = self.unknown_idx
            word_idx_tensor[0] = save

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        word_embed_idx, frequencies, pos_embed_idx, modifier_embed_idx, head_embed_idx, sentence_len = self.sentences_dataset[index]
        self.sentence_dropout(word_embed_idx, frequencies)
        return word_embed_idx, pos_embed_idx, head_embed_idx

    @staticmethod
    def init_word_embeddings(word_dict):
        glove = Vocab(Counter(word_dict), vectors="glove.6B.300d", specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    def init_pos_vocab(self, pos_dict):
        idx_pos_mappings = sorted([self.word_idx_mappings.get(token) for token in SPECIAL_TOKENS])
        pos_idx_mappings = {self.idx_word_mappings[idx]: idx for idx in idx_pos_mappings}

        for i, pos in enumerate(sorted(pos_dict.keys())):
            # pos_idx_mappings[str(pos)] = int(i)
            pos_idx_mappings[str(pos)] = int(i + len(SPECIAL_TOKENS))
            idx_pos_mappings.append(str(pos))
        return pos_idx_mappings, idx_pos_mappings

    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self, padding):
        sentence_word_idx_list = list()
        sentence_frequencies_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_modifier_idx_list = list()
        sentence_head_idx_list = list()
        sentence_len_list = list()
        for sentence_idx, sentence in enumerate(self.datareader.sentences):
            words_idx_list = [self.word_idx_mappings.get(ROOT_TOKEN)]
            pos_idx_list = [self.pos_idx_mappings.get(ROOT_TOKEN)]
            modifier_list = []
            head_list = []
            for word, pos, modifier, head in sentence:
                words_idx_list.append(self.word_idx_mappings.get(word, self.unknown_idx))
                pos_idx_list.append(self.pos_idx_mappings.get(pos, self.unknown_idx))
                modifier_list.append(modifier)
                head_list.append(head)
            sentence_len = len(words_idx_list)
            # if padding:
            #     while len(words_idx_list) < self.max_seq_len:
            #         words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
            #         pos_idx_list.append(self.pos_idx_mappings.get(PAD_TOKEN))
            frequencies = [self.datareader.word_dict[self.idx_word_mappings[idx]] for idx in words_idx_list]
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_frequencies_idx_list.append(torch.tensor(frequencies, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_modifier_idx_list.append(torch.tensor(modifier_list, dtype=torch.long, requires_grad=False))
            sentence_head_idx_list.append(torch.tensor(head_list, dtype=torch.long, requires_grad=False))
            sentence_len_list.append(sentence_len)

        # if padding:
        #     all_sentence_word_idx = torch.tensor(sentence_word_idx_list, dtype=torch.long)
        #     all_sentence_pos_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long)
        #     all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
        #     return TensorDataset(all_sentence_word_idx, all_sentence_pos_idx, all_sentence_len)
        # [[1, 2, 3], [4, 5, 6]]
        return {i: sample_tuple for i, sample_tuple in enumerate(zip(
            sentence_word_idx_list, sentence_frequencies_idx_list, sentence_pos_idx_list, sentence_modifier_idx_list,
            sentence_head_idx_list, sentence_len_list))}
