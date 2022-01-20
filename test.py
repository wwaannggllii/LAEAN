# -*- coding: utf-8 -*-
import argparse, time, os, random
import scipy.misc as misc
import imageio
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
import cv2

import torch

import options.options as option
from utils import util
from models.SRModel import SRModel
from data import create_dataloader
from data import create_dataset
from data.common import rgb2ycbcr

BENCHMARK = ['Set5', 'Set14', 'Urban100', 'Manga109','B100']


def main():
    # os.environ['CUDA_VISIBLE_DEVICES']='1' # You can specify your GPU device here. I failed to perform it by `torch.cuda.set_device()`.
    parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)
    opt = option.dict_to_nonedict(opt)

    # Initialization
    scale = opt['scale']
    dataroot_HR = opt['datasets']['test']['dataroot_HR']
    network_opt = opt['networks']['G']
    if network_opt['which_model'] == "feedback":
        model_name = "%s_f%dt%du%ds%d"%(network_opt['which_model'], network_opt['num_features'], network_opt['num_steps'], network_opt['num_units'], network_opt['num_stages'])
    else:
        model_name = network_opt['which_model']

    bm_list = [dataroot_HR.find(bm)>=0 for bm in BENCHMARK]
    bm_idx = bm_list.index(True)
    bm_name = BENCHMARK[bm_idx]

    # create test dataloader
    dataset_opt = opt['datasets']['test']
    if dataset_opt is None:
        raise ValueError("test dataset_opt is None!")
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)

    if test_loader is None:
        raise ValueError("The training data does not exist")

    if opt['mode'] == 'sr':
        solver = SRModel(opt)
    elif opt['mode'] == 'fi':
        solver = SRModel(opt)
    elif opt['mode'] == 'msan':
        solver = SRModel(opt)

    else:
        raise NotImplementedError

    # load model
    model_pth = os.path.join(solver.model_pth, 'best_checkpoint.pth')
    # model_pth = os.path.join(solver.model_pth, 'epoch', 'checkpoint.pth')
    # model_pth = solver.model_pth

    if model_pth is None:
        raise ValueError("model_pth' is required.")
    print('[Loading model from %s...]'%model_pth)
    model_dict = torch.load(model_pth, map_location='cpu')

    if 'state_dict' in model_dict.keys():
        solver.model.load_state_dict(model_dict['state_dict'])
    else:
        if model_name == "rcan_ours":
            new_model_dict = OrderedDict()
            for key, value in model_dict.items():
                new_key = 'module.'+key
                new_model_dict[new_key] = value
            model_dict = new_model_dict

        solver.model.load_state_dict(model_dict)
    print('=> Done.')
    print('[Start Testing]')



    # we only forward one epoch at test stage, so no need to load epoch, best_prec, results from .pth file
    # we only save images and .pth for evaluation. Calculating statistics are handled by matlab.
    # do crop for efficiency
    save_txt_path = os.path.join(solver.model_pth, '%s_x%d.csv' % (bm_name, scale))
    test_bar = tqdm(test_loader)
    sr_list = []
    path_list = []
    psnr_list = []
    ssim_list = []
    uiqm_list = []


    total_psnr = 0.
    total_ssim = 0.
    total_uiqm = 0.
    start_time = time.time()

    for iter, batch in enumerate(test_bar):
        solver.feed_data(batch)
        solver.test(opt['chop'])
        visuals = solver.get_current_visual()   # fetch current iteration results as cpu tensor

        sr_img = np.transpose(util.quantize(visuals['SR'], opt['rgb_range']).numpy(), (1, 2, 0)).astype(np.uint8)
        gt_img = np.transpose(util.quantize(visuals['HR'], opt['rgb_range']).numpy(), (1, 2, 0)).astype(np.uint8)

        # calculate PSNR
        crop_size = opt['scale']
        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
        ####################################################################################
        cropped_sr_img = cropped_sr_img / 255.
        cropped_gt_img = cropped_gt_img / 255.
        cropped_sr_img = rgb2ycbcr(cropped_sr_img).astype(np.float32)
        cropped_gt_img = rgb2ycbcr(cropped_gt_img).astype(np.float32)
        psnr = util.calc_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
        ssim = util.compute_ssim1(cropped_sr_img * 255, cropped_gt_img * 255)
        #########################################################################################
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        total_psnr += psnr
        total_ssim += ssim


        sr_list.append(sr_img)
        path_list.append(os.path.splitext(os.path.basename(batch['HR_path'][0]))[0])



        data_frame = pd.DataFrame(data={'Image name': path_list,
                                        'PSNR': psnr_list,
                                        'SSIM': ssim_list,
                                        },
                                  index=range(1, len(path_list) + 1)
                                  )
    data_frame.to_csv(save_txt_path,index_label="Len")

    print("=====================================")
    time_elapse = start_time - time.time()
    print('time_elapse:', time_elapse)


    line_list = []
    line = "Method : %s\nTest set : %s\nScale : %d "%(model_name, bm_name, scale)
    line_list.append(line+'\n')
    print(line)

    for value,value1,img_name in zip(psnr_list,ssim_list, path_list):
        line = "Image name : %s PSNR = %.5f SSIM = %.5f" %(img_name, value,value1)
        line_list.append(line + '\n')
        line_list.append(line)
        print(line)

    line = "Average PSNR is %.5f"%(total_psnr/len(test_bar))
    line_list.append(line)
    print(line)
    line = "Average SSIM is %.5f" % (total_ssim / len(test_bar))
    line_list.append(line)
    print(line)
    save_img_path = os.path.join('./eval/SR/BI', model_name, bm_name, "x%d"%scale)
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    for img, img_name in zip(sr_list, path_list):
        imageio.imsave(os.path.join(save_img_path, img_name.replace('HR', model_name)+'.png'), img)

    test_bar.close()



if __name__ == '__main__':
    main()