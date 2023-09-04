from model import Net
from train import train, test
from loader import get_dataloader
from torch.cuda import is_available
from argparse import ArgumentParser
import pandas
from matplotlib import pyplot as plt
from datetime import datetime
from utils import get_run_path

# from faulthandler import enable
# enable()

from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore",category=UserWarning)

parser = ArgumentParser()
parser.add_argument('--hidden_channels',type=int,default=64)
parser.add_argument('--num_layers',type=int,default=6)
parser.add_argument('--residual',action='store_true')
parser.add_argument('--dropout',type=float,default=0.)

parser.add_argument('--patience',type=int,default=40)
parser.add_argument('--num_epochs',type=int,default=400)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--train_batch_size',type=int,default=128)

parser.add_argument('--eval_batch_size',type=int,default=512)
parser.add_argument('--force_use_cpu',action='store_true')

args = parser.parse_args()

ds_path = 'data/ZINC'
run_path = get_run_path('runs')
overview_log_path = f"{run_path}/summary.log"
with open(overview_log_path,'w') as file:
    intital_info = {
        'start date and time' : datetime.now().strftime(r"%d/%m/%Y %H:%M:%S"),
        **vars(args)
    }
    lines = [
        f"{k} : {intital_info[k]}\n\r"
        for k in intital_info
    ]
    file.writelines(lines)

device = 'cpu' if args.force_use_cpu or not is_available() else 'cuda'

model = Net(args.hidden_channels,args.num_layers,args.dropout,'ZINC',args.residual).to(device)


with open(overview_log_path,'a') as file:
    file.writelines([
        'Model Summary',
        str(model)
    ])

train_loader = get_dataloader(ds_path,'train',args.train_batch_size,device)
val_loader = get_dataloader(ds_path,'val',args.eval_batch_size,device)

model, best_val_epoch, best_val_score, train_history, val_history = train(
    model,
    train_loader,
    val_loader,
    device,
    best_val_path=f'{run_path}/best_val.ckpt',
    num_epochs=args.num_epochs,
    lr=args.lr,
    patience=args.patience)


train_loader = get_dataloader(ds_path,'train',args.eval_batch_size,device)
train_score = test(model,train_loader,'train_score',device)
# val_score = test(model,val_loader,'val_score',device,on_to_device_transform)
test_loader = get_dataloader(ds_path,'test',args.eval_batch_size,device)
test_score = test(model,test_loader,'val_score',device)

print(f"Best validation epoch: {best_val_epoch}")
print("Scores:")
print(f"\tvalid: {best_val_score}")
print(f"\ttest : {test_score}")
print(f"\ttrain: {train_score}")

# updating overview log
with open(overview_log_path,'a') as file:
    intital_info = {
        'best validation valid score' : best_val_score,
        'best validation test score' : test_score,
        'best validation train score' : train_score,
    }
    lines = [
        f"{k} : {intital_info[k]}"
        for k in intital_info
    ]
    file.writelines(lines)

# saving raw metrics
hist = pandas.DataFrame.from_dict({
    'epoch' : list(range(1,1 + args.num_epochs)),
    'train_loss' : train_history,
    'val_score' : val_history,
})
hist.to_csv(f'{run_path}/raw_scores.csv')

# making images
plt.plot(train_history)
plt.plot(val_history)
plt.legend(['train_loss','val_score'])
plt.xlabel('epoch')
plt.ylabel('L1')
plt.savefig(f'{run_path}/history.png')


"""Default params:
Best validation epoch: 320                                                                                                                                
Scores:
        valid: 0.16286240208148955
        test : 0.15104008042812347
        train: 0.09995321006774903
Shutting down ptens.
"""