python3 main.py                                                         \
  --dataset=[MUTAG,PROTEINS,NCI1,NCI109,PTC_MR,IMDB-BINARY,IMDB-MULTI]  \
  --hidden_channels=[16,32]                                             \
  --num_layers=4                                                        \
  --dropout=0.                                                          \
  --train_batch_size=[32,128]                                           \
  --max_cycle_size=10                                                   \
  --show_epoch_progress_bar                                             \
  --num_epochs=350                                                      \
  --ptensor_reduction=mean