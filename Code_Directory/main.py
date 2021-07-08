import torch
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import re

from data import PosDataset
from model import KiperwasserDependencyParser as BasicModel, Advanced
from train import train, get_comp_tags

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_loaders(train_on_validation=False):
    train_dataset = PosDataset('data/train.labeled')
    eval_dataset = PosDataset('data/test.labeled',
                              word_embeddings=train_dataset.get_word_embeddings(),
                              pos_embeddings=train_dataset.get_pos_vocab()) if not train_on_validation else None

    comp_dataset = PosDataset('data/comp.unlabeled',
                              word_embeddings=train_dataset.get_word_embeddings(),
                              pos_embeddings=train_dataset.get_pos_vocab())

    word_vocab_size = len(train_dataset.datareader.word_dict)
    tag_vocab_size = len(train_dataset.datareader.pos_dict)
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=6)
    eval_loader = DataLoader(eval_dataset, shuffle=False) if not train_on_validation else None
    comp_loader = DataLoader(comp_dataset, shuffle=False)
    return train_loader, eval_loader, comp_loader, word_vocab_size, tag_vocab_size


def write_comp(model: torch.nn.Module, comp_loader: DataLoader, model_type='basic'):
    comp_tags = get_comp_tags(comp_loader, model)
    model.eval()
    idx = 0
    print(f'tagging competition with {model_type} model')
    m = 'm1' if type == 'basic' else 'm2'
    with open(f'comp_{m}.labeled', 'w') as comp_labeled:
        with open('data/comp.unlabeled', 'r') as unlabeled:
            for line in unlabeled:
                if line == '\n':
                    comp_labeled.write('\n')
                    continue
                splited = re.split('\t', line)
                splited[6] = str(int(comp_tags[idx]))
                comp_labeled.write('\t'.join(splited))
                idx += 1


def visualize_results(stats: dict):
    # visualize loss and accuracy along the training time
    train_loss = stats['train_loss']
    test_loss = stats['test_loss']
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Loss as function of epochs')
    # plt.show()

    train_UAS = stats['train_UAS']
    test_UAS = stats['test_UAS']
    plt.figure()
    plt.plot(train_UAS, label='Train UAS')
    plt.plot(test_UAS, label='Test UAS')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('UAS')
    plt.title('UAS as function of epochs')
    # plt.show()


def base_model():
    # model parameters
    lstm_num_layers = 3
    word_embedding_dim = 100
    pos_embedding_dim = 40
    lstm_hidden_dim = 300
    mlp_hidden_dim = 130
    dropout = 0.25

    # train parameters
    epochs = 15
    acumulate_grad_steps = 50
    lr = 1e-2
    lr_decay = 0.8
    weight_decay = 0
    model_type = 'basic'

    train_loader, eval_loader, comp_loader, word_vocab_size, tag_vocab_size = get_loaders()
    model = BasicModel(lstm_num_layers, word_vocab_size, tag_vocab_size, word_embedding_dim, pos_embedding_dim,
                       lstm_hidden_dim, mlp_hidden_dim, dropout)
    stats = train(train_loader, eval_loader, model, epochs, acumulate_grad_steps, lr, lr_decay, weight_decay,
                  model_type, True)

    visualize_results(stats)

    train_loader, eval_loader, comp_loader, word_vocab_size, tag_vocab_size = get_loaders(True)
    model = BasicModel(lstm_num_layers, word_vocab_size, tag_vocab_size, word_embedding_dim, pos_embedding_dim,
                       lstm_hidden_dim, mlp_hidden_dim, dropout)
    train(train_loader, eval_loader, model, epochs, acumulate_grad_steps, lr, lr_decay, weight_decay, model_type, False)
    write_comp(model, comp_loader)


def advanced_model():
    # model parameters
    lstm_num_layers = 4
    word_embedding_dim = 100
    pos_embedding_dim = 40
    lstm_hidden_dim = 400
    mlp_hidden_dim = 200
    dropout = 0.5

    # train parameters
    epochs = 20
    acumulate_grad_steps = 50
    lr = 2e-3
    lr_decay = 0.8
    weight_decay = 1e-7
    model_type = 'advanced'

    train_loader, eval_loader, comp_loader, word_vocab_size, tag_vocab_size = get_loaders()
    model = Advanced(lstm_num_layers, word_vocab_size, tag_vocab_size, word_embedding_dim, pos_embedding_dim,
                     lstm_hidden_dim, mlp_hidden_dim, dropout)
    stats = train(train_loader, eval_loader, model, epochs, acumulate_grad_steps, lr, lr_decay, weight_decay,
                  model_type, True)

    visualize_results(stats)

    train_loader, eval_loader, comp_loader, word_vocab_size, tag_vocab_size = get_loaders(True)
    model = Advanced(lstm_num_layers, word_vocab_size, tag_vocab_size, word_embedding_dim, pos_embedding_dim,
                     lstm_hidden_dim, mlp_hidden_dim, dropout)
    train(train_loader, eval_loader, model, epochs, acumulate_grad_steps, lr, lr_decay, weight_decay, model_type, False)
    write_comp(model, comp_loader)


def main():
    base_model()
    advanced_model()


if __name__ == '__main__':
    main()
