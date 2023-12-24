import torch
import numpy as np
from PIL import Image
import csv
import os
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import glob


class SecularDataset(Dataset):
	def __init__(self, transform, opt, mode='train'):
		super(SecularDataset, self).__init__()
		self.root = opt.dataroot
		self.transform = transform
		self.raw_image_dir = opt.train_dir
		self.test_data_dir = opt.test_dir
		if mode == 'train':
			self.specular_image, self.raw_image = self.load_csv('image.csv', self.raw_image_dir)
		else:
			self.specular_image, self.raw_image = self.load_csv('image_test.csv', self.test_data_dir)

	def load_csv(self, filename, path):
		if not os.path.exists(os.path.join(self.root, filename)):

			raw_image = glob.glob(os.path.join(self.root, path, '*D.png'))
			specular_image = glob.glob(os.path.join(self.root, path, '*A.png'))

			raw_image.sort()
			specular_image.sort()
			raw_image = np.asarray(raw_image)
			specular_image = np.asarray(specular_image)

			# shuffle
			indices = np.arange(0, len(raw_image))
			np.random.shuffle(indices)
			specular_image = specular_image[indices]
			raw_image = raw_image[indices]

			# write to csv
			with open(os.path.join(self.root, filename), mode='w', newline='') as f:
				writer = csv.writer(f)
				for i in range(len(raw_image)):
					writer.writerow([specular_image[i], raw_image[i]])
				print('writen into csv file : ', filename)

		# read csv
		specular_image, raw_image = [], []
		with open(os.path.join(self.root, filename)) as f:
			reader = csv.reader(f)
			for row in reader:
				img, label = row
				specular_image.append(img)
				raw_image.append(label)

		assert len(specular_image) == len(raw_image)
		return specular_image, raw_image

	def __len__(self):
		return len(self.specular_image)

	def __getitem__(self, index):

		specular_image, raw_image = self.specular_image[index], self.raw_image[index]
		# transform
		tf = self.transform
		tf_img = transforms.Compose([
			lambda x: Image.open(x).convert('RGB'),
			transforms.Resize([224, 224]),
			transforms.ToTensor()])

		specular_image = tf_img(specular_image)
		raw_image = tf_img(raw_image)

		aug_img = torch.cat([specular_image, raw_image], dim=0)
		aug_img = tf(aug_img)

		specular_image, raw_image = torch.split(aug_img, [3, 3], dim=0)
		return specular_image, raw_image






