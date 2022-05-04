from statistics import mode
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import torch
import tensorflow as tf


# class DigesterDataSet(Dataset):
#     def __init__(self, file_path, train = True):
#         self.data = pd.read_csv(file_path)
#         self.samples = []

#         self.samples = [tuple(x) for x in self.data.values]
#         self.X = []
#         self.y = []      
#         for value in self.data.values:
#             row = tuple(value)
#             #print(row)
#             self.y.append(row[-4])
#             #print(row[:-4] + row[-3:])
#             self.X.append(row[:-4] + row[-3:])
        
#         x_train, x_test, y_train, y_test = train_test_split(
#             self.X, self.y, test_size=0.2, train_size=0.8, random_state=4)

#         # two modes - train and test
#         if train:
#             self.x_data, self.y_data = x_train, y_train
#         else:
    #         self.x_data, self.y_data = x_test, y_test

    # def __len__(self):
    #     return len(self.samples)

    # def __getitem__(self, index):
    #     return self.samples[index]


def display_graph(dataset):
    for header in list(dataset.columns.values):
        truc = dataset[[header]]
        truc.plot()
    plt.show()


if __name__ == "__main__":
    filepath = "data\digester_data.csv"
    dataset = pd.read_csv(filepath, index_col=0, parse_dates=True)

    # dataset_train = DigesterDataSet(filepath)
    # dataset_test = DigesterDataSet(filepath, train=False)
    # #print(len(dataset))
    # #print(dataset[1])
    
    # data_train = torch.utils.data.DataLoader(
    #     dataset, batch_size=50, shuffle=False)
    # data_test = torch.utils.data.DataLoader(
    #     dataset, batch_size=50, shuffle=False)

    # model = Sequential([
    #     Dense(100, activation='relu', input_shape=(10,)),
    #     Dense(10, activation='relu'), ])

    # model.summary()

    # model.compile(
    #     optimizer='adam',
    #     loss='mse',
    #     metrics=['accuracy'],
    # )
    
    # history = model.fit(
    #     dataset_train.x_data,
    #     to_categorical(dataset_train.y_data),
    #     epochs=10,
    #     batch_size=32,
    #     validation_data=(dataset_test.x_data,
    #                      to_categorical(dataset_test.y_data))
    # )
