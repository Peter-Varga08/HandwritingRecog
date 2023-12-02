import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn

from character_recognition.utils import resize_pad


class CharacterRecognizer(nn.Module):
    def __init__(self):
        super(CharacterRecognizer, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 15, 5, 1, 0),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.lin_layers = nn.Sequential(
            nn.Linear(7 * 7 * 15, 300),
            nn.ReLU(),
            nn.Linear(300, 27),
            nn.LogSoftmax(dim=1)
        )
        self.opt = torch.optim.Adam(params=self.parameters(), weight_decay=0.002)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 7 * 7 * 15)
        x = self.lin_layers(x)

        return x

    def pre_process(self, img):
        """
        Pre-processes the image for the model.
        """
        img = resize_pad(img, 40, 40)
        img = np.reshape(img, [1, 1, 40, 40])
        img = torch.Tensor(img)
        return img

    def predict(self, input):
        """
        Feed 40x40 sized image input to model and return (label-idx, and prob) thereof.
        """
        input = self.pre_process(input)
        if input.shape != (1, 1, 40, 40):
            raise ValueError('Input shape must be (1, 1, 40, 40)')
        out = self(input)
        pred_label = torch.argmax(out).detach().numpy()
        out = out.detach().numpy()
        # output of model is log softmax -> this makes it probability distribution
        char_probs = np.exp(out) / (np.exp(out)).sum()

        return pred_label, char_probs[0][pred_label]

    def load_checkpoint(self, ckpt_path, map_location=torch.device('cpu')):
        ckpt = torch.load(ckpt_path, map_location=map_location)
        print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
        return ckpt

    def save_checkpoint(self, state, save_path):
        torch.save(state, save_path)

    def load_model(self, checkpoint: str):
        ckpt = self.load_checkpoint(checkpoint)
        self.epoch = ckpt['epoch']
        self.load_state_dict(ckpt['weights'])
        self.opt.load_state_dict(ckpt['optimizer'])
