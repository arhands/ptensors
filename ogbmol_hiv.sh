python3 main.py             \
  --dataset=ogbg-molhiv     \
  --hidden_channels=64      \
  --num_layers=2            \
  --dropout=0.5             \
  --train_batch_size=128    \
  --ptensor_reduction=mean  \
  --readout=mean            \
  --optimizer=asam          \
  --show_epoch_progress_bar \
  --lr=0.0001               \
  --num_epochs=200          \
  --max_cycle_size=6
  # --wandb_project_name=ogbg_molhiv