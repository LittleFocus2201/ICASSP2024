import os

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

import cv2
from torchvision import transforms
from tqdm import tqdm

import network as network
from argument import arg

def save_png(image, name, pixel_max_cnt=255):
	# Save image one-by-one

	img = image * 255
	# Process img_copy and do not destroy the data of img
	img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
	img_copy = np.clip(img_copy, 0, pixel_max_cnt)
	img_copy = img_copy.astype(np.uint8)
	img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
	cv2.imwrite(name, img_copy)


def test(opt):
	generator = network.GateGenerator(opt).to("cuda:0")


	def load_model(net, checkpoint):
		state_dict = torch.load(checkpoint,map_location=torch.device('cuda:0'))
		net.load_state_dict(state_dict['model'])


	checkpoint = opt.infer_model


	load_model(generator, checkpoint)

	tf_aug = transforms.Compose([
		lambda x: Image.open(x).convert('RGB'),
		transforms.ToTensor()
	])


	def infer(input, output, imgname):
		img = tf_aug(input+imgname).to("cuda:0")


		model = 'bicubic'
		img = img.unsqueeze(0)
		b,c,h,w = img.shape

		if img.shape[2] < 250:
			img = F.interpolate(img,size=[224,224],mode=model,align_corners=True)
		else:
			img = F.interpolate(img,size=[448,448],mode=model,align_corners=True)


		first_out, second_out, HF = generator(img)

		second_out = img * (1 - HF) + second_out * HF

		final = second_out
		final = F.interpolate(final,size=[h,w],mode=model,align_corners=True)
		HF = F.interpolate(HF,size=[h,w],mode=model,align_corners=True)

		save_png(final, output+imgname[:-4]  +'-removed' +'.png')
		save_png(HF, output+imgname[:-4]+'-HFE' +'.png')
	input = opt.input_dir
	output = opt.output_dir

	imglist = os.listdir(input)

	for img in tqdm(imglist):
		print(img,'done!')

		infer(input,output,img)


if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	opt = arg()
	test(opt)
