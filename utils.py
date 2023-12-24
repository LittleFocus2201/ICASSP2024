import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import nn
import skimage
from skimage.measure import compare_ssim

    
# ----------------------------------------
#             PATH processing
# ----------------------------------------
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_names(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = img * 255
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.jpg'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = compare_ssim(target, pred, multichannel = True)
    return ssim


## for contextual attention

def extract_patch(patch_num,intensity):
    kenel_size = intensity.shape[2]/patch_num
    patches = nn.MaxPool2d(kenel_size)
    return patches










def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images

def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


# for highlight contextual attention
def cal_intensity_feature(patch_num, intensity, raw_size):
    pool = nn.MaxPool2d(raw_size // patch_num)  # patch_num=28
    patch_fb = pool(intensity)  # [b, 3, 28, 28]
    return patch_fb


def compute_attention(feature, patch_fb):  # [b, 192, 56, 56]
    convtrans = nn.ConvTranspose2d(3, 192, kernel_size=3, padding=1).cuda()
    patch_fb2 = convtrans(patch_fb)
    b = feature.shape[0]
    feature = F.interpolate(feature, scale_factor=0.5, mode='bilinear')  # [b, 192, 28, 28]
    # highlight region
    feature_h = patch_fb2 * feature
    # non-highlight region
    feature_n = (1 - patch_fb2) * feature
    p_fb = torch.reshape(patch_fb, [b, 3, 28 * 28, 1])
    p_matrix_h = torch.matmul(p_fb, p_fb.permute([0, 1, 3, 2]))
    p_matrix_n = torch.matmul((1 - p_fb), (1 - p_fb).permute([0, 1, 3, 2]))
    feature_h = feature_h.permute([0, 2, 3, 1]).reshape([b, 3, 28 * 28, 64])
    feature_n = feature_n.permute([0, 2, 3, 1]).reshape([b, 3, 28 * 28, 64])

    cos = cosine(feature_h, feature_n)
    c_h = cos * p_matrix_h
    c_n = cos * p_matrix_n
    s_h = F.softmax(c_h, dim=2) * p_matrix_h
    s_n = F.softmax(c_n, dim=2) * p_matrix_n
    s = torch.cat((s_h, s_n), dim=1)
    return s


def Attention_implementation(feature, attention):  # feature: [B, C, H, W]
    b_num, c, h, w = feature.shape
    f = extract_image_patches(feature, 28)
    f = torch.reshape(f, [b_num, 3, f.shape[1] * f.shape[2], -1])
    f = torch.matmul(attention, f)
    f = torch.reshape(f, [b_num, 28, 28, h // 28, w // 28, c])
    f = f.permute([0, 5, 1, 3, 2, 4])
    f = torch.reshape(f, [b_num, c, h, w])
    return f


def extract_image_patches(img, patch_num):
    b, c, h, w = img.shape
    img = torch.reshape(img, [b, c, patch_num, h // patch_num, patch_num, w // patch_num])
    img = img.permute([0, 2, 4, 3, 5, 1])
    return img


def cosine(tensor_A, tensor_B):
    Mul_tensor = torch.matmul(tensor_A, tensor_B.permute([0, 1, 3, 2]))
    tensor_A_norm = torch.sqrt((tensor_A * tensor_A).sum(axis=3)).unsqueeze(dim=3)
    tensor_B_norm = torch.sqrt((tensor_B * tensor_B).sum(axis=3)).unsqueeze(dim=3)
    return Mul_tensor / torch.matmul(tensor_A_norm, tensor_B_norm.permute([0, 1, 3, 2]))








class Colorloss(nn.Module):
    def __init__(self):
        super(Colorloss, self).__init__()

        self.loss_fn = nn.MSELoss()

    def forward(self,img,gt):
        yuv_img = self.rgb2yuv(img)
        yuv_gt = self.rgb2yuv(gt)

        return self.loss_fn(yuv_img[1:],yuv_gt[1:])




    def rgb2yuv(self,rgb):
        rgb_ = rgb.transpose(1, 3)  # input is b*3*n*n   default


        A = torch.tensor([[0.299, -0.14714119, 0.61497538],
                          [0.587, -0.28886916, -0.51496512],
                          [0.114, 0.43601035, -0.10001026]])  # from  Wikipedia

        try:
            A = A.cuda()
        except Exception as e:
            pass
        yuv = torch.tensordot(rgb_, A, 1).transpose(1, 3)
        return yuv

    def yuv2rgb(self,yuv):
        yuv_ = yuv.transpose(1,3)                              # input is b*3*n*n   default
        A = torch.tensor([[1., 1.,1.],
                          [0., -0.39465, 2.03211],
                          [1.13983, -0.58060, 0]])             # from  Wikipedia
        try:
            A = A.cuda()
        except Exception as e:
            pass
        rgb = torch.tensordot(yuv_,A,1).transpose(1,3)
        return rgb


if __name__ == '__main__':
    t = torch.rand(2,3,256,256)
    model = Colorloss()

    u = model(t)

