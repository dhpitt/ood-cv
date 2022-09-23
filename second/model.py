import math

import torch
from torch import nn
from torchvision import models
import torchvision.transforms as tvt

from load_geirhos_model import load_model

class GaussianNoiseTransform(object):
    '''Custom transform to add Gaussian noise to an image'''
    def __init__(self, mu=0, sigmasq=0.05):
        self.mu = mu
        self.sigmasq = sigmasq

    def __repr__(self):
        return 'GaussianNoiseTransform object'

    def __call__(self, img):
        '''
        Applies gaussian noise to a 1 or 3 channel image
        img: torch.Tensor of shape 1 || 3 x h x w
        '''
        noise = torch.randn(size=img.size()) * math.sqrt(self.sigmasq)
        img += noise.cuda()
        return img

contrastive_augs = tvt.Compose([
    tvt.RandomApply([tvt.GaussianBlur(kernel_size=(5, 5), sigma=(1,1))]),
    tvt.RandomApply([tvt.ColorJitter(brightness=.5, hue=.5)]),
    tvt.RandomApply([GaussianNoiseTransform()])
])

class SpecifiedResNet(nn.Module):
    '''
    A model that classifies azimuth.
    '''

    def __init__(self, *, out_bins):
        super(SpecifiedResNet, self).__init__()
        self.net = load_model("resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN")
        self.net.fc = nn.Identity()
        self.classifier = nn.Linear(2048, out_bins)

    def forward(self, batch):
        self.train()
        x, y = batch
        loss = 0

        # add a contrastive learning loss term
        view1, view2 = contrastive_augs(x), contrastive_augs(x)
        z1, z2 = nn.functional.normalize(self.net(view1)), nn.functional.normalize(self.net(view2))
        similarity = z1 @ z2.T
        target = torch.arange(len(similarity)).cuda()
        loss += nn.functional.cross_entropy(similarity, target, reduction='mean')

        # add classification accuracy loss term
        features = self.net(x)
        logits = self.classifier(features)
        
        loss += nn.functional.cross_entropy(input=logits, target=torch.tensor(y).cuda())

        return loss
    
    def classify(self, batch):
        self.eval()
        x, y = batch
        features = self.net(x)
        logits = self.classifier(features)
        p_hat = nn.functional.softmax(logits, dim=1)
        y_hat = torch.argmax(p_hat, dim=1)
        return (y_hat == torch.tensor(y).cuda()).sum()/len(y)
    
    def unlabeled_inference(self, batch):
        self.eval()
        x, names, labels = batch
        logits = self.net(x)
        p_hat = nn.functional.softmax(logits, dim=1)
        y_hat = torch.argmax(p_hat, dim=1)
        #print(y_hat, names)
        return y_hat.cpu().numpy()[0], names[0], labels[0]
        

if __name__ == "__main__":
    rn = SpecifiedResNet(out_bins = 12)
    print(rn.net.fc)

    img = torch.randn(size=[3,224,224])
    xform = GaussianNoiseTransform()
    print(xform(img))
