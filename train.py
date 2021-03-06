# -*- coding: utf-8 -*-
import argparse, time, os
import random

import torch

import pandas as pd
from tqdm import tqdm

import options.options as option
from utils import util
from models.SRModel import SRModel
from data import create_dataloader
from data import create_dataset
from data.common import rgb2ycbcr
import numpy as np

# os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_IS"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def main():
    # os.environ['CUDA_VISIBLE_DEVICES']="1" # You can specify your GPU device here. I failed to perform it by `torch.cuda.set_device()`.
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)

    if opt['train']['resume'] is False:
        util.mkdir_and_rename(opt['path']['exp_root'])  # rename old experiments if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'exp_root' and \
                     not key == 'pretrain_G' and not key == 'pretrain_D'))
        option.save(opt)
        opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    else:
        opt = option.dict_to_nonedict(opt)
        if opt['train']['resume_path'] is None:
            raise ValueError("The 'resume_path' does not declarate")

    if opt['exec_debug']:
        NUM_EPOCH = 100
        opt['datasets']['train']['dataroot_HR'] = opt['datasets']['train']['dataroot_HR_debug']
        opt['datasets']['train']['dataroot_LR'] = opt['datasets']['train']['dataroot_LR_debug']

    else:
        NUM_EPOCH = int(opt['train']['num_epochs'])

    # random seed
    seed = opt['train']['manual_seed'] #0
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print('Number of train images in [%s]: %d' % (dataset_opt['name'], len(train_set)))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('Number of val images in [%s]: %d' % (dataset_opt['name'], len(val_set)))
        elif phase == 'test':
            pass
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    if train_loader is None:
        raise ValueError("The training data does not exist")

    # TODO: design an exp that can obtain the location of the biggest error
    if opt['mode'] == 'sr':
        solver = SRModel(opt)
    elif opt['mode'] == 'srgan':
        solver = SRModelGAN(opt)


    solver.summary(train_set[0]['LR'].size())
    solver.net_init()
    print('[Start Training]')

    start_time = time.time()

    start_epoch = 1
    if opt['train']['resume']:
        start_epoch = solver.load()

    for epoch in range(start_epoch, NUM_EPOCH + 1):
        # Initialization
        solver.training_loss = 0.0
        epoch_loss_log = 0.0


        if opt['mode'] == 'sr' or opt['mode'] == 'srgan' or opt['mode'] == 'sr_curriculum':
            training_results = {'batch_size': 0, 'training_loss': 0.0}
        else:
            pass    # TODO
        train_bar = tqdm(train_loader)


        # Train model
        for iter, batch in enumerate(train_bar):
            solver.feed_data(batch)
            iter_loss = solver.train_step()
            epoch_loss_log += iter_loss.item()
            batch_size = batch['LR'].size(0)
            training_results['batch_size'] += batch_size

            if opt['mode'] == 'sr':
                training_results['training_loss'] += iter_loss * batch_size
                train_bar.set_description(desc='[%d/%d] Loss: %.4f ' % (
                    epoch, NUM_EPOCH, iter_loss))
            elif opt['mode'] == 'srgan':
                training_results['training_loss'] += iter_loss * batch_size
                train_bar.set_description(desc='[%d/%d] Loss: %.4f ' % (
                    epoch, NUM_EPOCH, iter_loss))
            elif opt['mode'] == 'sr_curriculum':
                training_results['training_loss'] += iter_loss.data * batch_size
                train_bar.set_description(desc='[%d/%d] Loss: %.4f ' % (
                    epoch, NUM_EPOCH, iter_loss))
            else:
                pass    # TODO

        solver.last_epoch_loss = epoch_loss_log / (len(train_bar))

        train_bar.close()
        time_elapse = time.time() - start_time
        start_time = time.time()
        print('Train Loss: %.4f' % (training_results['training_loss'] / training_results['batch_size']))

        # validate
        val_results = {'batch_size': 0, 'val_loss': 0.0, 'psnr': 0.0, 'ssim': 0.0}

        if epoch % solver.val_step == 0 and epoch != 0:
            print('[Validating...]')
            start_time = time.time()
            solver.val_loss = 0.0

            vis_index = 1

            for iter, batch in enumerate(val_loader):
                visuals_list = []

                solver.feed_data(batch)
                iter_loss = solver.test(opt['chop'])
                batch_size = batch['LR'].size(0)
                val_results['batch_size'] += batch_size

                visuals = solver.get_current_visual()   # float cpu tensor

                sr_img = np.transpose(util.quantize(visuals['SR'], opt['rgb_range']).numpy(), (1,2,0)).astype(np.uint8)
                gt_img = np.transpose(util.quantize(visuals['HR'], opt['rgb_range']).numpy(), (1,2,0)).astype(np.uint8)

                # calculate PSNR
                crop_size = opt['scale']
                cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]

                cropped_sr_img = cropped_sr_img / 255.
                cropped_gt_img = cropped_gt_img / 255.
                cropped_sr_img = rgb2ycbcr(cropped_sr_img).astype(np.float32)
                cropped_gt_img = rgb2ycbcr(cropped_gt_img).astype(np.float32)


                val_results['val_loss'] += iter_loss * batch_size

                val_results['psnr'] += util.calc_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                val_results['ssim'] += util.compute_ssim1(cropped_sr_img * 255, cropped_gt_img * 255)

                if opt['mode'] == 'srgan':
                    pass    # TODO

            avg_psnr = val_results['psnr']/val_results['batch_size']
            avg_ssim = val_results['ssim']/val_results['batch_size']
            print('Valid Loss: %.4f | Avg. PSNR: %.4f | Avg. SSIM: %.4f | Learning Rate: %f'%(val_results['val_loss']/val_results['batch_size'], avg_psnr, avg_ssim, solver.current_learning_rate()))

            time_elapse = start_time - time.time()

            #if epoch%solver.log_step == 0 and epoch != 0:
            # tensorboard visualization
            solver.training_loss = training_results['training_loss'] / training_results['batch_size']
            solver.val_loss = val_results['val_loss'] / val_results['batch_size']

            solver.tf_log(epoch)

            # statistics
            if opt['mode'] == 'sr' or opt['mode'] == 'srgan' or opt['mode'] == 'sr_curriculum' or opt['mode'] == 'fi'or opt['mode'] == 'msan':
                solver.results['training_loss'].append(solver.training_loss.cpu().data.item())
                solver.results['val_loss'].append(solver.val_loss.cpu().data.item())
                solver.results['psnr'].append(avg_psnr)
                solver.results['ssim'].append(avg_ssim)
            else:
                pass    # TODO

            is_best = False
            if solver.best_prec < solver.results['psnr'][-1]:
                solver.best_prec = solver.results['psnr'][-1]
                is_best = True
            solver.save(epoch, is_best)

        # update lr
        solver.update_learning_rate(epoch)

    data_frame = pd.DataFrame(
        data={'training_loss': solver.results['training_loss']
            , 'val_loss': solver.results['val_loss']
            , 'psnr': solver.results['psnr']
            , 'ssim': solver.results['ssim']
              },
        index=range(1, NUM_EPOCH+1)
    )
    data_frame.to_csv(os.path.join(solver.results_dir, 'train_results.csv'),
                      index_label='Epoch')


if __name__ == '__main__':
    main()

