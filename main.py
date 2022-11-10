from cProfile import label
from re import T
import torch
from torch import nn
import pandas as df
import numpy as np
import matplotlib.pyplot as plt

# 成交股數 成交金額 開盤 最高 最低 收盤 價差 成交筆數
# FEATURES = ['shares','amount','open','high','low','close','change','turnover']
FEATURES = ['open','high','low','close']
DATA_SIZE = 4

class Regression(nn.Module):
    def __init__(self,input_size=DATA_SIZE*len(FEATURES),hidden_size=64):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def get_data(file, type,d_min=None,d_max=None):
    assert type in ('train', 'val', 'all')
    data = df.read_csv(file)
    if type=='train':
        data = data.iloc[:7*len(data)//8,:].copy()
    elif type == 'val':
        data = data.iloc[7*len(data)//8:,:].copy()
    data = data[FEATURES]
    data = data.astype('float32')
    d_min = data.min() if d_min is None else d_min
    d_max = data.max() if d_max is None else d_max
    data = (data - d_min) / (d_max - d_min)
    return data,d_min,d_max


### Train ###
train_data,train_min,train_max = get_data('./Dataset/train.csv','train')
train_y = train_data['close'].shift(-1)[DATA_SIZE-1:].dropna().values
train_x = np.array([train_data.iloc[i:i+DATA_SIZE].to_numpy().flatten() for i in range(len(train_data)-DATA_SIZE)])

rates = {0.3:[],0.1:[],0.05:[],0.01:[]}
models = {0.3:Regression(),0.1:Regression(),0.05:Regression(),0.01:Regression()}
epoch = 300
for rate, loss_list in rates.items():
    model = models[rate]
    loss_fc = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    for i in range(epoch):
        y_predicted = model(torch.tensor(train_x))
        loss = loss_fc(y_predicted.flatten(), torch.tensor(train_y))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())

        if i % 50 == 0:
            print(f'epoch: {i}, loss = {loss.item(): .7f}')

### Draw ###
for rate, loss_list in rates.items():
    plt.plot(range(1,epoch+1),loss_list,label=f'rate={rate}')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.show()

### Validate ###
val_data,_,_ = get_data('./Dataset/train.csv','val')
val_y = val_data['close'].shift(-1).iloc[DATA_SIZE-1:].dropna().values
val_x = np.array([val_data.iloc[i:i+DATA_SIZE].to_numpy().flatten() for i in range(len(val_data)-DATA_SIZE)])
final_model = (None,99,99)
for rate,model in models.items():
    loss_fc = nn.MSELoss()
    y_predicted = model(torch.tensor(val_x))
    loss = loss_fc(y_predicted.flatten(), torch.tensor(val_y))
    final_model = (model,loss.item(),rate) if final_model[1] > loss.item() else final_model
    print(rate,loss.item())


### Predict ###
pred_data,p_min,p_max = get_data('./Dataset/test.csv','all')
pred_x = np.array([pred_data.iloc[i*DATA_SIZE:i*DATA_SIZE+DATA_SIZE].to_numpy().flatten() for i in range(len(pred_data)//DATA_SIZE)])
pred_y = final_model[0](torch.tensor(pred_x))

write_data = "id,result\n"
for i,t in enumerate(pred_y.flatten()*(p_max['close'] - p_min['close'])+p_min['close']):
    write_data += f'{i},{t.item()}\n'

with open('./submit.csv','w') as f:
    f.write(write_data)
