import pandas as pd
import matplotlib.pyplot as plt



if __name__ == "__main__":
    filepath = "data\digester_data.csv"
    dataset = pd.read_csv(filepath, index_col=0 , parse_dates=True)
    #dataset.plot()
    #plt.show()
    for header in list(dataset.columns.values):
        truc = dataset[[header]]
        truc.plot()
    plt.show()
    #print(dataset)
    #dataset.plot()
    
    
