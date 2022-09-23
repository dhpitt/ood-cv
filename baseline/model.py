import torch
from torch import nn

from torchvision import models

from load_geirhos_model import load_model

class SpecifiedResNet(nn.Module):
    '''
    A model that classifies azimuth.
    '''

    def __init__(self, *, out_bins):
        super(SpecifiedResNet, self).__init__()
        self.net = load_model("resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN")
        self.net.fc = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, out_bins)
        )

    def forward(self, batch):
        x, y = batch
        logits = self.net(x)
        #p_hat = nn.functional.softmax(x, dim=1)
        #y_hat = torch.argmax(p_hat, dim=1)

        return nn.functional.cross_entropy(input=logits, target=torch.tensor(y).cuda())
    
    def classify(self, batch):
        x, y = batch
        logits = self.net(x)
        p_hat = nn.functional.softmax(logits, dim=1)
        y_hat = torch.argmax(p_hat, dim=1)
        return (y_hat == torch.tensor(y).cuda()).sum()/len(y)
    
    def unlabeled_inference(self, batch):
        x, names, labels = batch
        logits = self.net(x)
        p_hat = nn.functional.softmax(logits, dim=1)
        y_hat = torch.argmax(p_hat, dim=1)
        #print(y_hat, names)
        return y_hat.cpu().numpy()[0], names[0], labels[0]
        

if __name__ == "__main__":
    rn = SpecifiedResNet(out_bins = 12)
    print(rn.net.fc)
