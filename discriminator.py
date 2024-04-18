import torch
import torch.nn as nn



class Conv2dlayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
	             activation='elu', norm='none', sn=False):
		super(Conv2dlayer, self).__init__()

		# pading initialize
		if pad_type == 'reflect':
			self.pad = nn.ReflectionPad2d(padding=padding)
		elif pad_type == 'replicate':
			self.pad = nn.ReplicationPad2d(padding=padding)
		elif pad_type == 'zero':
			self.pad = nn.ZeroPad2d(padding=padding)
		else:
			assert 0, 'unexpected padding type:{}'.format(pad_type)

		# normalizeaton initialize

		if norm == 'bn':
			self.norm = nn.BatchNorm2d(out_channels)
		elif norm == 'in':
			self.norm = nn.InstanceNorm2d(out_channels)
		elif norm == 'la':
			self.norm = nn.LayerNorm(out_channels)
		elif norm == 'none':
			self.norm = None
		else:
			assert 0, 'unexpected normalization type:{}'.format(norm)

		# activation func initialize

		if activation == 'relu':
			self.activation = nn.ReLU(inplace=True)
		elif activation == 'lrelu':
			self.activation = nn.LeakyReLU(0.2, inplace=True)
		elif activation == 'elu':
			self.activation = nn.ELU(inplace=True)
		elif activation == 'selu':
			self.activation = nn.LeakyReLU(inplace=True)
		elif activation == 'tanh':
			self.activation = nn.Tanh()
		elif activation == 'sigmoid':
			self.activation = nn.Sigmoid()
		elif activation == 'none':
			self.activation = None
		else:
			assert 0, 'unexpected activation func type:{}'.format(activation)

		# spectral normlization + conv2d

		if sn:
			self.conv = nn.utils.spectral_norm(
				nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))

		else:
			self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

	def forward(self, x):
		x = self.pad(x)
		x = self.conv(x)
		if self.norm:
			x = self.norm(x)
		if self.activation:
			x = self.activation(x)
		return x



class PatchDiscriminate(nn.Module):
	def __init__(self, opt):
		super(PatchDiscriminate, self).__init__()
		# down sampling
		self.layer1 = Conv2dlayer(6, opt.latent_channels, 7, 1, 3, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
		self.layer2 = Conv2dlayer(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
		self.layer3 = Conv2dlayer(opt.latent_channels * 2, opt.latent_channels * 4, 4, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
		self.layer4 = Conv2dlayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
		self.layer5 = Conv2dlayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
		self.layer6 = Conv2dlayer(opt.latent_channels * 4, 1, 4, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)

	def forward(self, img, intensity):
		x = torch.cat((img, intensity), 1)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.layer6(x)
		return x


class PatchDiscriminate_2(nn.Module):
	def __init__(self, opt):
		super(PatchDiscriminate_2, self).__init__()
		# down sampling
		self.layer1 = Conv2dlayer(3, opt.latent_channels, 7, 1, 3, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
		self.layer2 = Conv2dlayer(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
		self.layer3 = Conv2dlayer(opt.latent_channels * 2, opt.latent_channels * 4, 4, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
		self.layer4 = Conv2dlayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
		self.layer5 = Conv2dlayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
		self.layer6 = Conv2dlayer(opt.latent_channels * 4, 1, 4, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)

	def forward(self, img):
		x = img
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.layer6(x)
		return x