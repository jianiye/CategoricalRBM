# Processes movie lens data
import pandas as pd
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

def getVandMask(path = "rating_1m.csv"):

    df = pd.read_csv(path)

    print(df.head())
    print(len(df))

    umin, umax = df['user'].min(), df['user'].max()
    mmin, mmax = df['movie'].min(), df['movie'].max()
    K = len(df['rating'].unique())

    Visible = np.zeros((umax, mmax))
    Mask = np.zeros((umax, mmax))
    countones = 0
    for u,m,r in zip(df['user'], df['movie'], df['rating']):
        Visible[u-1][m-1] = r
        Mask[u-1][m-1] = 1
        countones += 1

    # One-hot-lize visible
    Visible = torch.nn.functional.one_hot(torch.tensor(Visible.astype(int)))
    Visible = Visible.numpy()
    Visible = Visible[:,:,1:6]
    return Visible, Mask

class ratingdataset(Dataset):
  def __init__(self, V, Mask):
    self.V = V
    self.Mask = Mask

  def __len__(self):
    return len(self.V[:,0])

  def __getitem__(self, idx):
    return [self.V[idx], self.Mask[idx]]
