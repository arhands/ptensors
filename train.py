import torch
from torch.nn.functional import l1_loss
from torch_geometric.loader import DataLoader
from torch.nn import Module, L1Loss
from tqdm import tqdm
from transforms import PreprocessTransform

@torch.no_grad()
def test(model: Module, dataloader: DataLoader, description: str, device: str, position: int = 1) -> float:
    model.eval()
    score_sum = 0
    num_graphs = 0
    loop = tqdm(dataloader,description,total=len(dataloader),leave=False,position=position)
    for batch in loop:
        batch = batch.to(device)
        pred = model(batch)
        score = l1_loss(pred,batch.y).detach().item()
        score_sum += score*batch.num_graphs
        num_graphs += batch.num_graphs
        loop.set_postfix(avg_score=score_sum/num_graphs)
    model.train()
    return score_sum/num_graphs

def check_trainable_grads(module: Module, ntabs: int = 0):
    bad_params = []
    for p in module.parameters(False):
        # if p.requires_grad:
        if p.grad is None:
            bad_params.append((p.size(),p.requires_grad,None))
        elif p.grad.abs().max() == 0:
            bad_params.append((p.size(),p.requires_grad,0))
    bad_params = [str(s) for s in bad_params]
    print("%s%s: %s" % ('\t'*ntabs,module._get_name(),', '.join(bad_params)))
    for c in module.children():
        check_trainable_grads(c,ntabs+1)


def train(model: Module, train_dataloader: DataLoader, val_dataloader: DataLoader, device: str, best_val_path: str = 'best_val.ckpt', num_epochs: int = 400, lr: float = 0.01, patience: int = 40):
    print("trainable num params:",sum(p.numel() for p in model.parameters() if p.requires_grad))
    loss_fn = L1Loss()
    optim = torch.optim.Adam(model.parameters(),lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,'min',0.5,patience=patience)

    train_history = []
    val_history = []

    best_val = torch.inf
    train_loop = tqdm(range(num_epochs),total=num_epochs)
    for epoch in train_loop:
        grad_sum = 0
        loss_sum = 0
        total_graphs = 0
        epoch_loop = tqdm(train_dataloader,'train',total=len(train_dataloader),leave=False,position=1)
        lr = sched.optimizer.param_groups[0]['lr']
        for batch in epoch_loop:
            batch = batch.to(device)
            optim.zero_grad()
            loss : torch.Tensor = loss_fn(model(batch),batch.y)
            loss.backward()
            grad_sum += next(iter(model.atom_encoder.parameters())).grad.abs().sum().item()
            # check_trainable_grads(model)
            # raise NameError("?")
            optim.step()

            loss_float : float = loss.detach().item()
            loss_sum += loss_float * batch.num_graphs
            total_graphs += batch.num_graphs
            epoch_loop.set_postfix(avg_loss = loss_sum/total_graphs)
        loss_float = loss_sum / total_graphs
        sched.step(loss_float,epoch)

        val_score = test(model,val_dataloader,'val',device)
        
        if val_score < best_val:
            torch.save({
                'state_dict' : model.state_dict(),
                'best_val' : best_val,
                'epoch' : epoch
            },best_val_path)
            best_val = val_score

        train_history.append(loss_float)
        val_history.append(val_score)
        train_loop.set_postfix(best_val=best_val,train=loss_float,val=val_score,lr=lr,grad_sum=grad_sum)
    
    state = torch.load(best_val_path)
    best_val_epoch : int = state['epoch']
    best_val_score : float = state['best_val']
    model.load_state_dict(state['state_dict'])
    return model, best_val_epoch, best_val_score, train_history, val_history
        