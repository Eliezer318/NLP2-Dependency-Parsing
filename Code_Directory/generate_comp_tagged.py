import torch

from main import get_loaders, write_comp, BasicModel, Advanced

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lstm_num_layers = 3
word_embedding_dim = 100
pos_embedding_dim = 40
lstm_hidden_dim = 300
mlp_hidden_dim = 130
dropout = 0.25
train_loader, eval_loader, comp_loader, word_vocab_size, tag_vocab_size = get_loaders()
model = BasicModel(lstm_num_layers, word_vocab_size, tag_vocab_size, word_embedding_dim, pos_embedding_dim, lstm_hidden_dim, mlp_hidden_dim, dropout)
model.load_state_dict(torch.load(f'cache/basic.pkl'))
model = model.to(device)
write_comp(model, comp_loader)

lstm_num_layers = 4
word_embedding_dim = 100
pos_embedding_dim = 40
lstm_hidden_dim = 400
mlp_hidden_dim = 200
dropout = 0.5
model = Advanced(lstm_num_layers, word_vocab_size, tag_vocab_size, word_embedding_dim, pos_embedding_dim,
                       lstm_hidden_dim, mlp_hidden_dim, dropout)
model.load_state_dict(torch.load(f'cache/advanced.pkl'))
write_comp(model, comp_loader, 'advanced')
