import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F





class PerceptualNet(nn.Module):
	def __init__(self):
		super(PerceptualNet, self).__init__()
		block = [torchvision.models.vgg16(pretrained=True).features[:15].eval()]

		for p in block[0]:
			p.requires_grad = False
		self.block = torch.nn.ModuleList(block)
		self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
		self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

	def forward(self, x):
		x = (x - self.mean) / self.std
		x = F.interpolate(x, size=(224, 224), mode='bilinear',align_corners=True)
		for block in self.block:
			x = block(x)
		return x