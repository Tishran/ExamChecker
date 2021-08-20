import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch import nn

torch.manual_seed(242)

class NNForMnist(nn.Module):
  def __init__(self):
        super(NNForMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

net = NNForMnist()
net.load_state_dict(torch.load('./model_dict.pth'))
net.train()


def get_predictions_num(list_of_images):
    with torch.no_grad():
        output = net(list_of_images)

        result = []
        for i in range(len(output)):
            result.append(str(output.data.max(1, keepdim=True)[1][i].item()))

    return ''.join(result)
