import torch
import torch.nn.functional as F

class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
                        
        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                  out_channels=16,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=1) 
        self.conv_2 = torch.nn.Conv2d(in_channels=16,
                              out_channels=32,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=1) 
        self.conv_3 = torch.nn.Conv2d(in_channels=32,
                              out_channels=64,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=1)
        
        
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        hidden_dim = 64*124*1
        self.linear_1 = torch.nn.Linear(hidden_dim, 256)

        self.linear_out = torch.nn.Linear(256, num_classes)

        
    def embedding(self, x):
        # x [ B, F = 13, T]
        x = x.unsqueeze(1)
        out = self.conv_1(x)
        out=self.max_pool(out)
        out = F.relu(out)
        
        out = self.conv_2(out)
        out=self.max_pool(out)
        out = F.relu(out)
        
        out = self.conv_3(out)
        out=self.max_pool(out)
        out = F.relu(out)
        out = out.view(out.shape[0], -1)
        
        out = self.linear_1(out)

        
        return out
        
        
    def forward(self, x):

        out = self.embedding(x)
        out = F.relu(out)
        logits = self.linear_out(out)
        
        probas = F.softmax(logits, dim=1)
        
        return logits, probas
    
