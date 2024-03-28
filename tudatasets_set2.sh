# max cycle size not set
python3 main.py                                                         \
  --dataset=[MUTAG,NCI1,PTC_MR]                                         \
  --hidden_channels=[16,32]                                             \
  --num_layers=4                                                        \
  --dropout=0.                                                          \
  --train_batch_size=[32,128]                                           \
  --show_epoch_progress_bar                                             \
  --num_epochs=350                                                      \
  --patience=50                                                         \
  --lr=[0.01,0.001]                                                     \
  --ptensor_reduction=sum