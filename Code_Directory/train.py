import torch
from torch import optim
from torch import nn
from torch.utils.data.dataloader import DataLoader
from typing import Tuple
from tqdm import tqdm

from chu_liu_edmonds import decode_mst

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0, y ** -0.5)
        m.bias.data.fill_(0)


def train(train_loader: DataLoader, eval_loader: DataLoader, model: nn.Module, epochs, acumulate_grad_steps=50,
          lr=1e-3, lr_decay=0.9, weight_decay=1e-7, model_type='basic', evaluating=False) -> dict:
    """
    :param train_loader: train data loader
    :param eval_loader: validation data loader. If we are training on validation set will be None
    :param model: model to train on
    :param epochs: number of epcochs of training
    :param acumulate_grad_steps: how many grads to accumulate before making optimzation step
    :param lr: learning rate
    :param lr_decay: multiplicative learning rate decay
    :param weight_decay: weight decay for the optimizer
    :param model_type: is it the basic or the advanced model
    :param evaluating: bool, whether to evaluate train and validation in the end of the epoch
    :return:
    """

    model = model.to(device)
    with torch.no_grad():
        model.apply(weights_init_normal)
    stats = {'train_UAS': [], 'train_loss': [], 'test_UAS': [], 'test_loss': []}
    betas = (0.9, 0.9) if model_type == 'advanced' else (0.9, 0.999)  # basic get the defaults value
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    if model_type == 'basic':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay, milestones=range(epochs))
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 5_000, 1e-5)

    best_UAS = 0
    desc = f'training {model_type} model {"with" if evaluating else "without"} evaluation'
    for epoch in tqdm(range(epochs), desc=desc):
        train_UAS, train_loss = train_epoch(train_loader, model, optimizer, scheduler, acumulate_grad_steps, epoch,
                                            model_type)
        stats['train_loss'].append(train_loss)
        stats['train_UAS'].append(train_UAS)

        update = False
        if evaluating:
            if eval_loader:
                test_UAS, test_loss = evaluate(eval_loader, model)
                stats['test_UAS'].append(test_UAS)
                stats['test_loss'].append(test_loss)
                if best_UAS <= test_UAS:
                    best_UAS = test_UAS
                    update = True
        else:
            update = True
        if update:
            torch.save(model.state_dict(), f"cache/{model_type}/evaluating={evaluating}_weights.pkl")
            torch.save(stats, f"cache/{model_type}/evaluating={evaluating}_stats.pkl")

    return stats


def train_epoch(data_loader, model, optimizer, scheduler, acumulate_grad_steps, epoch, type_model='basic') -> Tuple[float, float]:
    train_UAS, printable_loss = 0, 0
    loss_function = nn.CrossEntropyLoss().to(device)
    data_loader.dataset.__setattr__('training', True)
    model.train()
    count = 0
    for i, (word_idx_tensor, pos_idx_tensor, head) in enumerate(data_loader):
        if type_model == 'basic':
            tag_scores = model(word_idx_tensor.to(device), pos_idx_tensor.to(device))
        else:
            tag_scores = model(word_idx_tensor.to(device), pos_idx_tensor.to(device), head.squeeze(0).to(device))
        tree_hat = decode_mst(tag_scores.detach().detach().cpu(), pos_idx_tensor.shape[1], has_labels=False)[0]
        loss = loss_function(tag_scores[:, 1:].T, head.squeeze(0).to(device))
        loss = loss / acumulate_grad_steps
        printable_loss += loss.item()
        loss.backward()
        if (i + 1) % acumulate_grad_steps == 0:
            nn.utils.clip_grad_value_(model.parameters(), 100 if type_model == 'basic' else 1000)
            optimizer.step()
            model.zero_grad()
            scheduler.step()
        train_UAS += (tree_hat[1:] == head.numpy()).sum()
        count += len(tree_hat) - 1
    return train_UAS / count, (printable_loss * acumulate_grad_steps) / len(data_loader)


@torch.no_grad()
def evaluate(data_loader: DataLoader, model: nn.Module) -> Tuple[float, float]:
    model.eval()
    data_loader.dataset.__setattr__('training', False)  # cancel dropout in training time
    loss_function = nn.CrossEntropyLoss().to(device)
    loss, count, UAS = 0, 0, 0
    for batch_idx, (word_idx_tensor, pos_idx_tensor, head) in enumerate(data_loader):
        tag_scores = model(word_idx_tensor.to(device), pos_idx_tensor.to(device))
        tree_hat = decode_mst(tag_scores.detach().cpu().numpy(), pos_idx_tensor.shape[1], has_labels=False)[0]
        loss += loss_function(tag_scores[:, 1:].T, head.squeeze(0).to(device)).item()
        UAS += (tree_hat[1:] == head.numpy()).sum()

    return UAS / sum(data_loader.dataset.__getattribute__('sentence_lens')), loss / len(data_loader)


@torch.no_grad()
def get_comp_tags(comp_loader: DataLoader, model: nn.Module, type_model='basic') -> list:
    # get unlabeled data and tag it in the order the data was read
    model = model.to(device)
    model.eval()
    comp_loader.dataset.__setattr__('training', False)  # cancel dropout in training time
    tags = []
    for batch_idx, (word_idx_tensor, pos_idx_tensor, _) in enumerate(comp_loader):
        tag_scores = model(word_idx_tensor.to(device), pos_idx_tensor.to(device))
        tree_hat = decode_mst(tag_scores.detach().cpu().numpy(), pos_idx_tensor.shape[1], has_labels=False)[0]
        tags.extend(tree_hat[1:])
    return tags
