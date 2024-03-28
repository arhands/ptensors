python3 main.py             \
  --dataset=ZINC            \
  --hidden_channels=128     \
  --num_layers=4            \
  --dropout=0.              \
  --min_lr=1E-5             \
  --train_batch_size=128    \
  --bn_momentum=0.05        \
  --lr_decay=0.5            \
  --ptensor_reduction=mean  \
  --readout=sum             \
  --patience=30             \
  --show_epoch_progress_bar
  # --wandb_project_name=ZINC_subset