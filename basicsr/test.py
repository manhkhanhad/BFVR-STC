import logging
import torch
from os import path as osp
import time
import datetime

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import (
    get_env_info,
    get_root_logger,
    get_time_str,
    make_exp_dirs,
    init_tb_logger,
)
from basicsr.utils.options import dict2str
from basicsr.train import parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt = parse_options(root_path, is_train=False)
    opt["path"]["pretrain_network_g"] = "experiments/20250611_235449_stage2_wo_noise/models/net_g_48000.pth"

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize tensorboard logger
    tb_logger = None
    if opt['logger'].get('use_tb_logger'):
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)
    
    # start testing
    start_time = time.time()
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=tb_logger, save_img=opt['val']['save_img'])

    # end of testing
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of testing. Time consumed: {consumed_time}')
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)