from cgi import test
import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 成交股數 成交金額 開盤 最高 最低 收盤 價差 成交筆數
# FEATURES = ['shares','amount','open','high','low','close','change','turnover']
FEATURES = ['close','change']

DATA_SIZE = 4
VAL_SIZE = 0.2

class Regression(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def get_train_data(mode='train',data_size=DATA_SIZE,features=FEATURES):
    df = pd.read_csv("./Dataset/train.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.asfreq('d')
    df['pred'] = df['close'].shift(-1)
    df = df.astype('float32')

    x_data = []
    y_data = []
    np_min = np.min(df,axis=0)
    np_max = np.max(df,axis=0)

    for _, sized_df in df.groupby(pd.Grouper(freq=f'{data_size}d')):
        if sized_df.isnull().values.any():
            continue
        sized_df = (sized_df - np_min) / (np_max - np_min)
        x_data.append(sized_df[features].to_numpy().flatten())
        y_data.append(sized_df['pred'][-1])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    length = len(x_data)
    start = 0 if mode=='train' else int(length*(1-VAL_SIZE))
    end = int(length*(1-VAL_SIZE)) if mode == 'train' else length
    return x_data[start:end], y_data[start:end], np_min, np_max

def get_test_data(np_min,np_max,features=FEATURES):
    df = pd.read_csv("./Dataset/test.csv")
    df = df.astype('float32')
    x_data = []
    # np_min = np.min(df,axis=0)
    # np_max = np.max(df,axis=0)
    for _, sized_df in df.groupby('id'):
        sized_df = (sized_df - np_min) / (np_max - np_min)
        x_data.append(sized_df[features].to_numpy().flatten())
    return np.array(x_data),np_min,np_max

### Train ###

models = {0.1:Regression(DATA_SIZE*len(FEATURES), 4)}
rates = {0.1:[]}
epoch = 1000
for rate, loss_list in rates.items():
    x,y,np_min,np_max = get_train_data()
    model = models[rate]
    loss_fc = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    for i in range(epoch):
        y_predicted = model(torch.tensor(x))
        loss = loss_fc(y_predicted.flatten(), torch.tensor(y))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())

        if i % 10 == 0:
            print(f'epoch: {i}, loss = {loss.item(): .7f}')


# ## Draw epoch-loss ###
# for rate, loss_list in rates.items():
#     plt.plot(range(1,epoch+1),loss_list,label=f'rate={rate}')
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.legend()
# plt.grid(True)
# plt.show()

### Test ###
final_model = (None,99,99)
ys = []
yp = []
for rate,model in models.items():
    x,y,_,_ = get_train_data('val')
    loss_fc = nn.MSELoss()
    y_predicted = model(torch.tensor(x))
    loss = loss_fc(y_predicted.flatten(), torch.tensor(y))
    ys.append(y*(np_max['close'] - np_min['close'])+np_min['close'])
    yp.append(y_predicted.detach().flatten()*(np_max['close'] - np_min['close'])+np_min['close'])
    final_model = (model,loss.item(),rate) if final_model[1] > loss.item() else final_model
    print(rate,loss.item())

# ### Draw real-val ###
# fig = plt.figure()
# a=fig.add_subplot(1,1,1)
# a.plot(ys[0],ys[0],label='correct')
# a.plot(ys[0],yp[0],'o',label='pred')
# a.legend()
# a.grid(True)
# plt.ylabel("predict_stock")
# plt.xlabel("real_stock")
# plt.show()


### Predict ###
test_data,_,_ = get_test_data(np_min,np_max)
write_data = "id,result\n"
y_predicted = final_model[0](torch.tensor(test_data))

for i,t in enumerate(y_predicted.flatten()*(np_max['close'] - np_min['close'])+np_min['close']):
    write_data += f'{i},{t.item()}\n'

with open('./submit.csv','w') as f:
    f.write(write_data)
# print(write_data)

