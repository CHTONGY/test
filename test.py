# # -*- coding: utf-8 -*-
import detectron2
import torch
import torchvision

from torch import nn
import argparse

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        在构造函数中，我们实例化了两个nn.Linear模块，并将它们作为成员变量。
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear2 = nn.DataParallel(self.linear2)
        self.linear3 = torch.nn.Linear(H, D_out)
        self.linear3 = nn.DataParallel(self.linear3)
        

    def forward(self, x):
        """
        在前向传播的函数中，我们接收一个输入的张量，也必须返回一个输出张量。
        我们可以使用构造函数中定义的模块以及张量上的任意的(可微分的）操作。
        """
        h_relu = self.linear1(x).clamp(min=0)
        h_relu = self.linear2(h_relu).clamp(min=0)
        y_pred = self.linear3(h_relu)
        return y_pred

def main(args):
    if args.data_path == None:
        print("the path to the training data is None")
        return

    # N是批大小； D_in 是输入维度；
    # H 是隐藏层维度； D_out 是输出维度
    N, D_in, H, D_out = 8, 100, 100, 10

    if args.pre_trained_model == None:
        # 通过实例化上面定义的类来构建我们的模型
        model = TwoLayerNet(D_in, H, D_out).to(torch.device("cuda")) 
    else:
        # 加载训练好的模型
        model = torch.load(args.pre_trained_model).to(torch.device("cuda"))

    # 产生输入和输出的随机张量
    x = torch.ones((N, D_in)).to(torch.device("cuda"))
    y = torch.ones((N, D_out)).to(torch.device("cuda"))
    
    # 构造损失函数和优化器
    # 构造损失函数和优化器
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    
    print("---------------------start to train----------------------")

    for t in range(50):
        # 前向传播：通过向模型传递x计算预测值y
        y_pred = model(x)

        # 计算并输出loss
        loss = criterion(y_pred, y)

        print("iter:", t, "loss:", loss.item())

        # 清零梯度，反向传播，更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ("------------------start to predict-----------------------")
    # test阶段
    x_test = torch.ones((N, D_in)).to(torch.device("cuda"))
    y_test = model(x_test).to(torch.device("cuda"))
    
    model_save_path = args.save_path + "model.pkl"
    torch.save(model, model_save_path)

    print(y_test)

if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--data_path", type = str, help="the path to training data")
    parser.add_argument("--pre_trained_model", default=None, help="the preTrained model")
    parser.add_argument("--save_path", default="./", help="the path to save the model and metrics")
    
    args = parser.parse_args()

    main(args)
