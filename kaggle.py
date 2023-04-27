#run on colab
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 讀入 train.csv 檔案
train_data = pd.read_csv('train.csv')
# 定義模型的神經網路
#123
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.Lrelu = nn.LeakyReLU()
      self.relu = nn.ReLU()
      self.elu = nn.ELU()
      self.tanh = nn.Tanh()
      Neurons = [6,117,27,2,520,47,9,222,1440,404,31,62,53,15,6,10,101,39,7,1]
      self.start = nn.Linear(2, Neurons[0])
      self.h1 = nn.Linear(Neurons[0], Neurons[1])
      self.h2 = nn.Linear(Neurons[1], Neurons[2])
      self.h3 = nn.Linear(Neurons[2], Neurons[3])
      self.h4 = nn.Linear(Neurons[3], Neurons[4])
      self.h5 = nn.Linear(Neurons[4], Neurons[5])
      self.h6 = nn.Linear(Neurons[5], Neurons[6])
      self.h7 = nn.Linear(Neurons[6], Neurons[7])
      self.h8 = nn.Linear(Neurons[7], Neurons[8])
      self.h9 = nn.Linear(Neurons[8], Neurons[9])
      self.h10 = nn.Linear(Neurons[9], Neurons[10])
      self.h11 = nn.Linear(Neurons[10], Neurons[11])
      self.h12 = nn.Linear(Neurons[11], Neurons[12])
      self.h13 = nn.Linear(Neurons[12], Neurons[13])
      self.h14 = nn.Linear(Neurons[13], Neurons[14])
      self.h15 = nn.Linear(Neurons[14], Neurons[15])
      self.h16 = nn.Linear(Neurons[15], Neurons[16])
      self.h17 = nn.Linear(Neurons[16], Neurons[17])
      self.h18 = nn.Linear(Neurons[17], Neurons[18])
      self.final = nn.Linear(Neurons[18], 1)
    def forward(self, x):
      x = self.start(x)
      x = self.tanh(x) #
      x = self.h1(x)
      x = self.Lrelu(x) ##
      x = self.h2(x)
      x = self.tanh(x) #
      x = self.h3(x)
      x = self.tanh(x) #
      x = self.h4(x)
      x = self.Lrelu(x) ##
      x = self.h5(x)
      x = self.tanh(x) #
      x = self.h6(x)
      x = self.h7(x)
      x = self.h8(x)
      x = self.h9(x)
      x = self.tanh(x) #
      x = self.h10(x)
      x = self.h11(x)
      x = self.h12(x)
      x = self.Lrelu(x) ##
      x = self.elu(x)
      x = self.h13(x)
      x = self.tanh(x) #
      x = self.h14(x)
      x = self.h15(x)
      x = self.h16(x)
      x = self.h17(x)
      x = self.h18(x)
      x = self.final(x)
      x = self.Lrelu(x) ##
      return x
# 建立模型
model = Net().to(DEVICE)
# 定義損失函數和優化器
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000064)
# 定義一個空列表，用於存儲每個epoch的loss值，繪製趨勢圖用
loss_list = []
# 訓練模型
train_time = 3001 #訓練次數
print_dis = 500 #多少次印一次
for epoch in tqdm(range(train_time)):
  inputs = torch.tensor(train_data[['x1', 'x2']].values, device=DEVICE, dtype=torch.float32)
  labels = torch.tensor(train_data[['y']].values, device=DEVICE, dtype=torch.float32)
  optimizer.zero_grad()
  outputs = model(inputs)
  loss = criterion(outputs, labels)
  loss.backward() #逆傳播
  optimizer.step() #丟回去
  loss_list.append(loss.item()) #繪圖用
  if epoch % print_dis == 0:
      print('epoch {}, loss {}'.format(epoch, loss.item()))
# 讀入 test.csv 檔案 (開始預測)
test_data = pd.read_csv('test.csv')
# 預測 test.csv 的 y 值
with torch.no_grad():
    test_inputs = torch.tensor(test_data[['x1', 'x2']].values, device=DEVICE, dtype=torch.float32)
    predicted_y = model(test_inputs).detach().cpu().numpy()
# 將預測結果存入 ans.csv 檔案中
result = pd.DataFrame({'id': range(1, len(test_data)+1), 'y': predicted_y.flatten()})
result.to_csv('ans.csv', index=False)
#簡單測試
my_best = pd.read_csv('k0_01643.csv')
now_test = pd.read_csv('ans.csv')
my_best = my_best.iloc[:, -1].values
now_test = now_test.iloc[:, -1].values
print("MSE:", np.mean((my_best - now_test)**2))
# 繪製訓練趨勢圖
plt.plot(range(train_time), loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Trend')
plt.show()

