python3 main.py             \
  --dataset=ZINC            \
  --hidden_channels=128     \
  --num_layers=4            \
  --dropout=0.              \
  --min_lr=2E-5             \
  --train_batch_size=256    \
  --bn_momentum=0.1         \
  --lr_decay=0.5            \
  --lr=0.002                \
  --ptensor_reduction=mean  \
  --readout=sum             \
  --patience=60             \
  --enable_model_summary    \
  --show_epoch_progress_bar
  # --wandb_project_name=ZINC_subset