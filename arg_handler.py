from argparse import ArgumentParser, Namespace
from model_handler import lr_scheduler_arg_type_list, loss_arg_type_list, score_arg_type_list, tu_dataset_type_list
from data_handler import dataset_type_list

def get_args() -> Namespace:
  parser = ArgumentParser()
  parser.add_argument('--dataset',choices=dataset_type_list,required=True)

  # general environment arguments
  parser.add_argument('--enable_model_summary',action="store_true")
  parser.add_argument('--show_epoch_progress_bar',type=bool,default=True)
  parser.add_argument('--device',type=str,default=None)
  parser.add_argument('--eval_batch_size',type=int,default=512)

  # preprocessing arguments
  parser.add_argument('--max_cycle_size',type=int,default=None)

  # model design arguments parameters
  parser.add_argument('--dropout',type=float,default=0.)
  parser.add_argument('--hidden_channels',type=int,default=128)
  parser.add_argument('--num_layers',type=int,default=4)
  parser.add_argument('--ptensor_reduction',type=str,default='mean')
  parser.add_argument('--readout',type=str,default='sum')
  parser.add_argument('--include_cycle2cycle',action='store_true')
  parser.add_argument('--feature_encoder',choices=['OGB','Embedding'],required=True)

  # training/evaluation arguments
  parser.add_argument('--task_type',choices=['classification','single-target-regression'],required=True)
  parser.add_argument('--mode',choices=['min','max'],required=True)
  ## scheduler
  parser.add_argument('--lr_scheduler',choices=lr_scheduler_arg_type_list,default=None)
  parser.add_argument('--patience',type=int,
                      help='For Reduce LR on plateau, it is patience, and for StepLR, it is the time between steps.')
  parser.add_argument('--min_lr',type=float,default=None)


  parser.add_argument('--num_folds',type=int,default=None)

  parser.add_argument('--lr',type=float,default=0.001)
  parser.add_argument('--num_epochs',type=int,default=1000)
  parser.add_argument('--batch_size',type=int,default=128)
  parser.add_argument('--seed',type=int,required=True)
  parser.add_argument('--optimizer',type=str,default='adam')
  parser.add_argument('--eval_metric',choices=score_arg_type_list,required=True)
  parser.add_argument('--loss',choices=loss_arg_type_list,required=True)

  parser.add_argument('--bn_momentum',type=float,default=0.1)

  args: Namespace = parser.parse_args()
  # validating 
  if args.dataset in tu_dataset_type_list:
    assert args.num_folds is not None, "The number of folds must be specified for TUDatasets."
  if args.lr_scheduler is not None and args.lr_scheduler == 'ReduceLROnPlateau':
    assert args.min_lr is not None, "min_lr must be specified for ReduceLROnPlateau."
  return args
