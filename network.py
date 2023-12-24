import torch.nn.init as init
import torchvision
from model_utils import *

from argument import arg
import torch
from torchvision.models import resnet34
from einops import rearrange

def weights_init(net, init_type='kaiming', init_gain=0.02):
	"""Initialize network weights"""

	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and classname.find('Conv') != -1:
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, 0.02)
			init.constant_(m.bias.data, 0.0)
		elif classname.find('Linear') != -1:
			init.normal_(m.weight, 0, 0.01)
			init.constant_(m.bias, 0)

	# Apply the initialization function
	net.apply(init_func)


def bilinear_kernel(in_channels, out_channels, kernel_size):
	factor = (kernel_size + 1) // 2
	if kernel_size % 2 == 1:
		center = factor - 1
	else:
		center = factor - 0.5
	og = np.ogrid[:kernel_size, :kernel_size]
	filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
	weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
	weight[range(in_channels), range(out_channels), :, :] = filt
	return torch.from_numpy(weight)


class CoarseNetwork(nn.Module):
	def __init__(self, opt):
		super(CoarseNetwork, self).__init__()
		# encoder
		self.coarse1 = nn.Sequential(
			GatedConv2d(opt.in_channels, opt.latent_channels, 5, 1, 2, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 3, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),

		)
		# dilation
		self.coarse2 = nn.Sequential(
			GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation=2, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation=4, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation=8, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation=16, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),

		)
		# decoder
		self.coarse3 = nn.Sequential(
			TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels, opt.latent_channels // 2, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels // 2, opt.out_channels, 3, 1, 1, pad_type=opt.pad_type, activation='none', norm=opt.norm),
			nn.Tanh()
		)

	def forward(self, first_in):
		first_out = self.coarse1(first_in)
		first_out = self.coarse2(first_out)

		first_out = self.coarse3(first_out)

		return first_out


class GateGenerator(nn.Module):
	def __init__(self, opt):
		super(GateGenerator, self).__init__()



		### Coarse Network
		self.coarse = CoarseNetwork(opt)

		### Refin Network
		self.refine1 = nn.Sequential(
			GatedConv2d(opt.in_channels, opt.latent_channels, 5, 1, 2, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels, opt.latent_channels, 3, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			)
		self.refine2 = nn.Sequential(
			GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			)
		self.refine3 = nn.Sequential(
			GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			)
		self.refine4 = nn.Sequential(
			GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation=2, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation=4, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			)
		self.refine5 = nn.Sequential(
			GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation=8, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation=16, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm)

		)

		self.refine6 = nn.Sequential(
			GatedConv2d(opt.latent_channels * 8, opt.latent_channels * 4, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
		)

		self.refine7 = nn.Sequential(
			TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels * 1, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			GatedConv2d(opt.latent_channels * 1, opt.out_channels, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
			nn.Tanh()
		)


		self.conv_P1 = nn.Sequential(
			GatedConv2d(opt.latent_channels * 1, opt.latent_channels * 1, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
		)
		self.conv_P2 = nn.Sequential(
			GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
		)
		self.attention_trans = nn.Sequential(
			GatedConv2d(opt.in_channels, opt.out_channels, 3, 1, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
		)

		### Hightlight
		self.HFE = HighlightFeatureExtractor(opt.n_class)

	def forward(self, img):
		b,c,h,w = img.shape
		self.patchnum = h//8

		HFE = self.HFE(img)
		HFE = torch.mean(HFE,dim=1,keepdim=True)
		HFE = torch.cat([HFE,HFE,HFE],dim=1)
		## coarse
		coarse_hi_img =(1 - HFE) * img + HFE * img
		# in [b, 6, h, w]
		coarse_in = torch.cat((coarse_hi_img, HFE), dim=1)
		# out [b, 3, h, w]
		coarse_out = self.coarse(coarse_in)
		## refine
		refine_hi_img = img * (1 - HFE) + coarse_out * HFE
		refine_in = torch.cat((refine_hi_img, HFE), dim=1)  # [b, 6, h, w]
		P1 = self.refine1(refine_in)  # [b, 48, 112, 112]
		refine_out = self.refine2(P1) # [b, 96, 56, 56]
		refine_out = self.refine3(refine_out) # [b, 192, 56, 56]
		refine_out = self.refine4(refine_out) + refine_out # [b, 192, 56, 56]
		P2 = self.refine5(refine_out) + refine_out # [b, 192, 56, 56]

		patch_fb = self.cal_HFE_feature(self.patchnum, HFE, h) # [b, 3, 28, 28]
		att = self.compute_attention(P2,patch_fb) #[b, 6, 784, 784]
		att = self.attention_trans(att)  #[b, 3, 784, 784]
		transP2 =self.conv_P2(self.attention_transfer(P2, att))  #[b, 192, 56, 56]
		second_out = torch.cat((P2, transP2), dim=1) #[b, 384, 56, 56]
		second_out = self.refine6(second_out)

		transP1 = self.conv_P1(self.attention_transfer(P1,att)) #[b, 48, 112, 112]

		second_out = torch.cat((second_out, transP1), dim=1)

		second_out = self.refine7(second_out)  #[b, 3, 224, 224]
		second_out = torch.clamp(second_out,0,1)
		return coarse_out, second_out, HFE

	def cal_HFE_feature(self, patch_num, intensity, raw_size):

		pool = nn.MaxPool2d(raw_size // patch_num)  # patch_num=28
		patch_fb = pool(intensity)  # [b, 3, 28, 28]
		return patch_fb

	def compute_attention(self, feature, patch_fb):  # [b, 192, 56, 56]
		patchnum = self.patchnum
		try:
			convtrans = nn.ConvTranspose2d(3,192,kernel_size=3,padding=1).cuda()
		except Exception as e:
			convtrans = nn.ConvTranspose2d(3,192,kernel_size=3,padding=1)

		patch_fb2 = convtrans(patch_fb)
		feature = F.interpolate(feature, scale_factor=0.5, mode='bilinear')  # [b, 192, 28, 28]
		# highlight region
		feature_h = patch_fb2 * feature
		# non-highlight region
		feature_n = (1-patch_fb2) * feature

		p_fb = rearrange(patch_fb,'b c h w -> b c (h w) 1')

		p_matrix_h = torch.matmul(p_fb, p_fb.permute([0,1, 3, 2]))
		p_matrix_n = torch.matmul((1-p_fb), (1 - p_fb).permute([0,1, 3, 2]))
		feature_h = rearrange(feature_h,'b (c k) h w -> b c (h w) k',c=3)
		feature_n = rearrange(feature_n,'b (c k) h w -> b c (h w) k',c=3)

		cos = self.cosine_Matrix(feature_h, feature_n)

		c_h = cos * p_matrix_h

		c_n = cos * p_matrix_n
		s_h = F.softmax(c_h, dim=2) * p_matrix_h
		s_n = F.softmax(c_n, dim=2) * p_matrix_n
		return torch.cat((s_h,s_n),dim=1)

	def attention_transfer(self, feature, attention):  # feature: [B, C, H, W]
		b_num, c, h, w = feature.shape
		patch_num =self.patchnum
		f = self.extract_image_patches(feature)
		f = rearrange(f,'b hn wn p1 p2 (c k) -> b c (hn wn) (p1 p2 k) ',c=3)
		f = torch.matmul(attention, f)
		f = rearrange(f,'b c (hn wn) (p1 p2 k) -> b (c k) (hn p1) (wn p2) ',hn=patch_num,p1=h//patch_num,p2=h//patch_num)

		return f

	def extract_image_patches(self, img):
		patch_num = self.patchnum
		img = rearrange(img,'b c (p1 hn) (p2 wn) -> b hn wn p1 p2 c',hn=patch_num,wn=patch_num)

		return img

	def cosine_Matrix(self, _matrixA, _matrixB):
		_matrixA_matrixB = torch.matmul(_matrixA, _matrixB.permute([0,1, 3, 2]))
		_matrixA_norm = torch.sqrt((_matrixA * _matrixA).sum(axis=3)).unsqueeze(dim=3)
		_matrixB_norm = torch.sqrt((_matrixB * _matrixB).sum(axis=3)).unsqueeze(dim=3)
		return _matrixA_matrixB / torch.matmul(_matrixA_norm, _matrixB_norm.permute([0,1, 3, 2]))

# ----------------------------------
# Discriminator
# ----------------------------------
# input:generated image/ground truth and HFE
# output: patch based region

class PatchDiscriminate(nn.Module):
	def __init__(self, opt):
		super(PatchDiscriminate, self).__init__()
		# down sampling
		self.layer1 = Conv2dlayer(opt.in_channels, opt.latent_channels, 7, 1, 3, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm, sn=True)
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


# ----------------------------------------
#            Perceptual Network
# ----------------------------------------


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
		x = F.interpolate(x, size=(224, 224), mode='bilinear')
		for block in self.block:
			x = block(x)
		return x


# ----------------------------------------
#          HighlightFeatureExtractor
# ----------------------------------------
class HighlightFeatureExtractor(nn.Module):
	def __init__(self, num_classes=1):
		super(HighlightFeatureExtractor, self).__init__()
		pretrained_net = resnet34(pretrained=True)
		# downsample
		self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
		self.stage2 = list(pretrained_net.children())[-4]
		self.stage3 = list(pretrained_net.children())[-3]

		# unified channel
		self.scores1 = nn.Conv2d(512, num_classes, 1)
		self.scores2 = nn.Conv2d(256, num_classes, 1)
		self.scores3 = nn.Conv2d(128, num_classes, 1)

		# upsample
		self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
		self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)
		self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
		self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)
		self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
		self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)

	def forward(self, x):
		s1 = self.stage1(x)  # [b, 128, 28, 28]
		s2 = self.stage2(s1)  # [b, 256, 14, 14]
		s3 = self.stage3(s2)  # [b, 512, 7, 7]

		s3 = self.upsample_2x(self.scores1(s3))  # [b, 3, 14, 14]
		s2 = self.upsample_4x(self.scores2(s2) + s3)  # [b, 3, 28, 28]
		s1 = self.upsample_8x(self.scores3(s1) + s2)  # [b, 3, 224, 224]
		return s1

