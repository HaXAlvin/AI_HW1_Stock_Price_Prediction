from cProfile import label
from re import T
import torch
from torch import nn
import pandas as df
import numpy as np
import matplotlib.pyplot as plt

# 成交股數 成交金額 開盤 最高 最低 收盤 價差 成交筆數
# FEATURES = ['shares','amount','open','high','low','close','change','turnover'] # 14.9 loss val=0.00059
# FEATURES = ['shares', 'open','high','low','close','turnover'] #12 val=0.00056
# FEATURES = ['open','high','low','close','change'] #14 val loss=0.00032
# FEATURES = ['amount', 'open','high','low','close'] #11 3min val loss=0.00044
FEATURES = ['open','high','low','close'] # val loss=0.00038
# FEATURES = ['close','change']

DATA_SIZE = 4

class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(DATA_SIZE*len(FEATURES), 1024)
        self.fc2 = nn.Linear(1024, 1)
        # self.fc3 = nn.Linear(8, 1)
        

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        # x = torch.relu(x)
        # x = self.fc3(x)
        return x

def get_data(file, type):
    data = df.read_csv(file)
    assert type in ('train', 'val', 'all')
    if type=='train':
        data = data.iloc[:7*len(data)//8,:].copy()
    elif type == 'val':
        data = data.iloc[7*len(data)//8:,:].copy()
    data = data[FEATURES]
    data = data.astype('float32')
    close = data['close'].shift(-1)[DATA_SIZE-1:].dropna().values
    four_day_datas = []
    # four_day_nor = []
    for i in range(len(data)-DATA_SIZE):
        four_day_datas.append(data.iloc[i:i+DATA_SIZE].to_numpy().flatten())
        # d_min = four_day_datas[-1].min()
        # d_max = four_day_datas[-1].max()
        # four_day_datas[-1] = (four_day_datas[-1] - d_min) / (d_max - d_min)
        # four_day_nor.append(lambda x,d_max=d_max,d_min=d_min: x*(d_max-d_min)+d_min)
    return torch.tensor(np.array(four_day_datas)),torch.tensor(close)
    # ,four_day_nor


### Train ###
# train_data,_,_ = get_data('./Dataset/train.csv','train')
# train_y = train_data['close'].shift(-1)[DATA_SIZE-1:].dropna().values
# train_x = np.array([train_data.iloc[i:i+DATA_SIZE].to_numpy().flatten() for i in range(len(train_data)-DATA_SIZE)])
train_x,train_y = get_data('./Dataset/train.csv','train')

rates = {0.3:[],0.1:[],0.05:[],0.01:[]}
models = {0.3:Regression(),0.1:Regression(),0.05:Regression(),0.01:Regression()}
# rates = {0.01:[]}
epoch = 1000
for rate, loss_list in rates.items():
    model = models[rate]
    loss_fc = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    for i in range(epoch):
        
        y_predicted = model(train_x)
        # print(y_predicted)
        # print(y_predicted.flatten())
        # print(train_nor[0](y_predicted.flatten()[0]))
        # y_predicted = torch.Tensor(list(map(lambda x:x[1](x[0]),zip(y_predicted,train_nor))))
        # y_predicted = torch.Tensor(list(map(lambda pred: train_nor[pred[0]](pred[1]), enumerate(y_predicted.flatten()))))
        # print(y_predicted)
        loss = loss_fc(y_predicted.flatten(), train_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())

        if i % 10 == 0:
            print(f'epoch: {i}, loss = {loss.item(): .7f}, real={train_y[0]}, pred={y_predicted[0].item()}')

### Draw ###
for rate, loss_list in rates.items():
    plt.plot(range(1,epoch+1),loss_list,label=f'rate={rate}')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.show()


### Test ###
val_x,val_y = get_data('./Dataset/train.csv','val')
print(val_x)
exit()
# val_y = val_data['close'].shift(-1)[DATA_SIZE-1:].dropna().values
# val_x = np.array([val_data.iloc[i:i+DATA_SIZE].to_numpy().flatten() for i in range(len(val_data)-DATA_SIZE)])
final_model = (None,999999999,99999999)
for rate,model in models.items():
    loss_fc = nn.MSELoss()
    y_predicted = model(val_x)
    # y_predicted = list(map(lambda pred: val_nor[pred[0]](pred[1]), enumerate(y_predicted.flatten())))
    loss = loss_fc(y_predicted.flatten(), val_y)
    final_model = (model,loss.item(),rate) if final_model[1] > loss.item() else final_model
    print(rate,loss.item())

### Predict ###

test_x,_ = get_data('./Dataset/test.csv','all')

write_data = "id,result\n"
# test_x = np.array([test_data.iloc[i*DATA_SIZE:i*DATA_SIZE+DATA_SIZE].to_numpy().flatten() for i in range(len(test_data)//DATA_SIZE)])
y_predicted = final_model[0](test_x)


for i,t in enumerate(y_predicted.flatten()):
    write_data += f'{i},{t.item()}\n'

with open('./submit.csv','w') as f:
    f.write(write_data)
