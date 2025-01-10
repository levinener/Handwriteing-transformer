import torch
import torch.nn as nn
import torch.optim as optim

class LayerNetwork(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):##这是LayerNetwork类的构造函数
        super(LayerNetwork,self).__init__() ##调用父类nn.Module的构造函数，确保能够正确初始化
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(hidden_size,output_size)

    def forward(self,input):
        output = self.fc1(input)
        output = self.relu(output)
        output = self.fc2(output)
        return output

input_size=1
hidden_size=5
output_size=1
model = LayerNetwork(input_size,hidden_size,output_size)

###
critertion= nn.MSELoss()#定义损失函数
optimizer =optim.Adam(model.parameters(), lr=0.01)#定义优化器

x = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y = torch.tensor([[2.0],[3.0],[4.0],[5.0]])

for epoch in range(20):
    outputs=model(x) #(nn.Module)中定义的__call__方法会自动调用forward()方法
    loss= critertion(outputs,y)

    optimizer.zero_grad()#清除之前梯度
    loss.backward()#计算梯度

    optimizer.step()#更新权重

    print(f'Epoch [{epoch+1}/25], Loss: {loss.item():.4f}')
