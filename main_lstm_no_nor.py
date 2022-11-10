from cProfile import label
from re import T
import torch
from torch import nn
import pandas as df
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 成交股數 成交金額 開盤 最高 最低 收盤 價差 成交筆數
# FEATURES = ['shares','amount','open','high','low','close','change','turnover'] # 14.9 loss val=0.00059
# FEATURES = ['shares', 'open','high','low','close','turnover'] #12 val=0.00056
# FEATURES = ['open','high','low','close','change'] #14 val loss=0.00032
FEATURES = ['amount', 'open','high','low','close'] #11 3min val loss=0.00044
# FEATURES = ['open','high','low','close'] # val loss=0.00038
# FEATURES = ['close','change']

DATA_SIZE = 4

class Regression(nn.Module):
    def __init__(self,hidden_size=16):
        super(Regression, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=DATA_SIZE*len(FEATURES),hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.out = nn.Linear(hidden_size, 1)


    def forward(self, x):
        out,_ = self.lstm(x,None)
        out = self.out(out)
        return out


def get_data(file, type):
    data = df.read_csv(file)
    assert type in ('train', 'val', 'all')
    if type=='train':
        data = data.iloc[:7*len(data)//8,:].copy()
    elif type == 'val':
        data = data.iloc[7*len(data)//8:,:].copy()
    data = data[FEATURES]
    data = data.astype('float32')
    data['close'] = scaler.fit_transform(data['close'].values.reshape(-1,1))
    return data


scaler = MinMaxScaler(feature_range=(-1, 1))

### Train ###
train_data = get_data('./Dataset/train.csv','train')
train_y = train_data['close'].shift(-1)[DATA_SIZE-1:].dropna().values
# print(train_y)
# exit()
train_x = np.array([train_data.iloc[i:i+DATA_SIZE].to_numpy().flatten() for i in range(len(train_data)-DATA_SIZE)])

# rates = {0.3:[],0.1:[],0.05:[],0.01:[]}
rates = {0.01:[]}
# models = {0.3:Regression(),0.1:Regression(),0.05:Regression(),0.01:Regression()}
models = {0.01:Regression()}
# rates = {0.01:[]}
epoch = 200
for rate, loss_list in rates.items():
    model = models[rate]
    loss_fc = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    print(model)
    for i in range(epoch):
        y_predicted = model(torch.tensor(train_x))
        loss = loss_fc(y_predicted.flatten(), torch.tensor(train_y))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if i % 10 == 0:
            print(f'epoch: {i}, loss = {loss.item(): .7f}')

### Draw ###
epoch_loss = plt.figure(1)
for rate, loss_list in rates.items():
    plt.plot(range(1,epoch+1),loss_list,label=f'rate={rate}')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
epoch_loss.show()

real_pred = plt.figure(2)
plt.plot(range(2152),y_predicted.flatten().detach().numpy(),label='pred')
plt.plot(range(2152),train_y,label='real')
plt.legend()
plt.grid(True)
real_pred.show()
input()



### Test ###
val_data = get_data('./Dataset/train.csv','val')
val_y = val_data['close'].shift(-1)[DATA_SIZE-1:].dropna().values
val_x = np.array([val_data.iloc[i:i+DATA_SIZE].to_numpy().flatten() for i in range(len(val_data)-DATA_SIZE)])
final_model = (None,9999999,99)
for rate,model in models.items():
    loss_fc = nn.MSELoss()
    y_predicted = model(torch.tensor(val_x))
    loss = loss_fc(y_predicted.flatten(), torch.tensor(val_y))
    final_model = (model,loss.item(),rate) if final_model[1] > loss.item() else final_model
    print(rate,loss.item())

### Predict ###

test_data = get_data('./Dataset/test.csv','all')

write_data = "id,result\n"
test_x = np.array([test_data.iloc[i*DATA_SIZE:i*DATA_SIZE+DATA_SIZE].to_numpy().flatten() for i in range(len(test_data)//DATA_SIZE)])

y_predicted = final_model[0](torch.tensor(test_x))


for i,t in enumerate(scaler.inverse_transform(y_predicted.detach().numpy())):
    
    write_data += f'{i},{t.item()}\n'

with open('./submit.csv','w') as f:
    f.write(write_data)
