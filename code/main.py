import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

class DigesterDataSet(Dataset):
    def __init__(self, csvpath, mode='train'):
        self.mode = mode
        df = pd.read_csv(csvpath)
        le = LabelEncoder()
    
        if self.mode == 'train':
            df = df.dropna()
            self.inp = df.iloc[:,1:].values
            self.oup = df.iloc[:,0].values.reshape(891,1)
        else:
            self.inp = df.values
    def __len__(self):
        return len(self.inp)
    def __getitem__(self, idx):
        if self.mode == 'train':
            inpt  = torch.Tensor(self.inp[idx])
            oupt  = torch.Tensor(self.oup[idx])
            return { 'inp': inpt,
                     'oup': oupt,
            }
        else:
            inpt = torch.Tensor(self.inp[idx])
            return { 'inp': inpt
            }

def display_graph(dataset):
    for header in list(dataset.columns.values):
        truc = dataset[[header]]
        truc.plot()
    plt.show()

if __name__ == "__main__":
    filepath = "data\digester_data.csv"
    dataset = pd.read_csv(filepath, index_col=0 , parse_dates=True)
    display_graph(dataset)
    
    # train test split
