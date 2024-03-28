python3 main.py                 \
  --dataset=ogbg-moltox21       \
  --hidden_channels=256         \
  --num_layers=3                \
  --dropout=0.7                 \
  --train_batch_size=128        \
  --ptensor_reduction=mean      \
  --readout=mean                \
  --optimizer=adam              \
  --show_epoch_progress_bar     \
  --bn_momentum=0.01            \
  --num_epochs=300
  # --wandb_project_name=moltox21