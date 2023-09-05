import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader

from transform import JunctionTree
from model import Net
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--no_inter_message_passing', action='store_true')
args = parser.parse_args()
print(args)

root = 'data/ZINC'
transform = JunctionTree()

train_dataset = ZINC(root, subset=True, split='train', pre_transform=transform)
val_dataset = ZINC(root, subset=True, split='val', pre_transform=transform)
test_dataset = ZINC(root, subset=True, split='test', pre_transform=transform)

train_loader = DataLoader(train_dataset, 128, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, 1000, shuffle=False, num_workers=12)
test_loader = DataLoader(test_dataset, 1000, shuffle=False, num_workers=12)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
model = Net(hidden_channels=args.hidden_channels, out_channels=1,
            num_layers=args.num_layers, dropout=args.dropout,
            inter_message_passing=not args.no_inter_message_passing).to(device)


def train(epoch):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = (model(data).squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        total_error += (model(data).squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)


model.reset_parameters()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=10, min_lr=0.00001)

best_val_mae = test_mae = float('inf')
main_loop = tqdm(range(1, args.epochs + 1))
for epoch in main_loop:
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_mae = test(val_loader)
    scheduler.step(val_mae)

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        # test_mae = test(test_loader)
        torch.save('best_val.ckpt',{
            'epoch' : epoch,
            'score' : val_mae,
            'state_dict' : model.state_dict(),
            'loss' : loss,
        })
    main_loop.set_postfix(lr=lr,loss=loss,val=val_mae,best_val=best_val_mae)
    print()

best = torch.load('best_val.ckpt')
best_val_epoch = best['epoch']
best_val_score = best['score']
best_val_loss = best['loss']
model.load_state_dict(best['state_dict'])
best_test_score = test(test_loader)
print("best_val_epoch",best_val_epoch)
print("best_val_score",best_val_score)
print("best_val_loss",best_val_loss)
print("best_val_test_score",best_test_score)

