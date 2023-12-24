
from utils import *


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


class TransposeConv2dLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
	             activation='lrelu', norm='none', sn=True, scale_factor=2):
		super(TransposeConv2dLayer, self).__init__()
		self.scalar_factor = scale_factor
		self.conv = Conv2dlayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation,
		                        norm, sn)

	def forward(self, x):
		x = F.interpolate(x, scale_factor=self.scalar_factor, mode='bilinear')
		x = self.conv(x)
		return x


class GatedConv2d(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
	             activation='elu', norm='none', sn=False):
		super(GatedConv2d, self).__init__()

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
			self.conv_mask = nn.utils.spectral_norm(
				nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))

		else:
			self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
			self.conv_mask = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		x = self.pad(x)
		conv = self.conv(x)
		mask = self.conv_mask(x)
		gate_mask = self.sigmoid(mask)
		if self.activation:
			conv = self.activation(conv)
		x = conv * gate_mask

		return x


class TransposeGatedConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
	             activation='lrelu', norm='none', sn=True, scale_factor=2):
		super(TransposeGatedConv2d, self).__init__()
		self.scalar_factor = scale_factor
		self.conv = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation,
		                        norm, sn)

	def forward(self, x):
		x = F.interpolate(x, scale_factor=self.scalar_factor, mode='bilinear')
		x = self.conv(x)
		return x








