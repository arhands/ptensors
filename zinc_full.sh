python3 main.py             \
  --dataset=ZINC-Full       \
  --hidden_channels=128     \
  --num_layers=4            \
  --dropout=0.              \
  --min_lr=1E-5             \
  --train_batch_size=512    \
  --bn_momentum=0.05        \
  --lr_decay=0.5            \
  --ptensor_reduction=sum   \
  --readout=sum             \
  --patience=80             \
  --show_epoch_progress_bar \
  --num_trials=4            \
  --include_cycle2cycle
  # --wandb_project_name=ZINC_full