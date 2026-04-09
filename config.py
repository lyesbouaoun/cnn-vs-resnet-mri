from datatset_mean_std import  mean_and_std

MEAN,STD = mean_and_std("train")

BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001
NUM_CLASSES = 4



DEVICE = "cuda"