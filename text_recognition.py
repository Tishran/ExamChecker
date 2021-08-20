import numpy as np
import torch
import torchvision

from torch import nn
import torch.nn.functional as F

#torch.manual_seed(242)

TARGETS = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

class NNForMnist(nn.Module):
  def __init__(self):
        super(NNForMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4*4*128, 100)
        self.fc2 = nn.Linear(100, 33)

  def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv4_bn(self.conv3(x)), 2)) # dropout
        x = x.view(-1, 4*4*128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

net = NNForMnist()
network_state_dict = torch.load('./model_new_text.pth')
net.load_state_dict(network_state_dict)
net.eval()

def get_predictions_text(list_of_images):
    with torch.no_grad():
        output = net(list_of_images)
        result = []
        for i in range(len(output)):
            result.append(TARGETS[output.data.max(1, keepdim=True)[1][i].item()])

    return ''.join(result)

