import argparse


def arg():
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_path', type=str, default='./runs', help='saving folder')
	parser.add_argument('--pretrain_path', type=str, default='./pretrain', help='pretrain saving folder')
	parser.add_argument('--training_path', type=str, default='./runs', help='pretrain saving folder')

	parser.add_argument('--runs_name', type=str, default='multi_scale', help='logs and checkpoint saving folder')
	parser.add_argument('--sample_path', type=str, default='samples', help='training samples path that is a folder')

	parser.add_argument('--gan_type', type=str, default='WGAN', help='the type of GAN for training')
	parser.add_argument('--multi_gpu', type=bool, default=False, help='nn.Parallel needs or not')
	parser.add_argument('--ngpu', type=int, default=1, help='gpu nums')
	parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='True for unchanged input data type')
	parser.add_argument('--mixed_precision', type=bool, default=True, help='mixed_precision')

	parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between model checkpoints')
	parser.add_argument('--checkpoint', type=str, default='runs/old_train/checkpoint/GateGenerator_epoch2_batchsize16.mdl', help='load checkpointr')

	parser.add_argument('--load_name', type=str, default='', help='load model name')
	# Training parameters

	parser.add_argument('--epochs', type=int, default=100, help='number of epochs of training')
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--resume_epoch', type=int, default=2)
	parser.add_argument('--batch_size', type=int, default=3, help='size of the batches')
	parser.add_argument('--n_class', type=int, default=3, help='number class of HIE module')
	parser.add_argument('--lr_g', type=float, default=1e-4, help='Adam: generator learning rate')
	parser.add_argument('--lr_h', type=float, default=1e-4, help='Adam: generator learning rate')

	parser.add_argument('--lr_d', type=float, default=1e-4, help='Adam: discriminator learning rate')
	parser.add_argument('--b1', type=float, default=0.5, help='Adam: beta 1')
	parser.add_argument('--b2', type=float, default=0.999, help='Adam: beta 2')
	parser.add_argument('--weight_decay', type=float, default=0, help='Adam: weight decay')
	parser.add_argument('--lambda_content', type=float, default=10, help='the parameter of content loss')
	parser.add_argument('--lambda_perceptual', type=float, default=1, help='the parameter of perceptual loss')
	parser.add_argument('--lambda_gan', type=float, default=1, help='the parameter of gan loss of AdaReconL1Loss')
	parser.add_argument('--lambda_ssim', type=float, default=0.1, help='the parameter of gan loss of AdaReconL1Loss')

	parser.add_argument('--num_workers', type=int, default=16, help='number of cpu threads to use during batch generation')
	# Network parameters
	parser.add_argument('--in_channels', type=int, default=6, help='input RGB image + 1 channel mask')
	parser.add_argument('--out_channels', type=int, default=3, help='output RGB image')
	parser.add_argument('--latent_channels', type=int, default=48, help='latent channels')
	parser.add_argument('--pad_type', type=str, default='zero', help='the padding type')
	parser.add_argument('--activation', type=str, default='lrelu', help='the activation type')
	parser.add_argument('--norm', type=str, default='in', help='normalization type')
	parser.add_argument('--init_type', type=str, default='xavier', help='the initialization type')
	parser.add_argument('--init_gain', type=float, default=0.02, help='the initialization gain')
	# Dataset parameters
	parser.add_argument('--dataroot', type=str, default='SHIQ_tiny/', help='the data root')
	parser.add_argument('--train_dir', type=str, default='train', help='the training data folder')
	parser.add_argument('--test_dir', type=str, default='test', help='the test data folder')

	# infer
	parser.add_argument('--input_dir', type=str, default='test/inp/', help='the test data folder')
	parser.add_argument('--output_dir', type=str, default='test/out/', help='the test data folder')
	parser.add_argument('--infer_model', type=str, default='./savedmodel/Nature.mdl', help='the test data folder')

	opt = parser.parse_args()
	return opt
