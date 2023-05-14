import os
from os import path
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2

from model.network import XMem
from model.propagation.prop_net import PropagationNetwork
from model.fusion_net import FusionNet
from model.s2m.s2m_network import deeplabv3plus_resnet50 as S2M
from dataset.davis_test_dataset import DAVISTestDataset
from davis_processor import DAVISProcessor

from davisinteractive.session.session import DavisInteractiveSession
import warnings
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# torch.backends.cudnn.enabled = False
"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--prop_model', default='saves/propagation_model.pth')
parser.add_argument('--fusion_model', default='saves/fusion.pth')
parser.add_argument('--s2m_model', default='saves/s2m.pth')
parser.add_argument('--davis', default='../DAVIS')
parser.add_argument('--output',default='../output')
parser.add_argument('--save_mask', action='store_true')


parser.add_argument('--model', default='./saves/XMem.pth')





parser.add_argument('--save_all', action='store_true',
                    help='Save all frames. Useful only in YouTubeVOS/long-time video', )

parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')

# Long-term memory options
parser.add_argument('--disable_long_term', default=True, action='store_true')
parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time',
                    type=int, default=10000)
parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)

# Multi-scale options
parser.add_argument('--save_scores', action='store_true')
parser.add_argument('--flip', action='store_true')
parser.add_argument('--size', default=480, type=int,
                    help='Resize the shorter side to this size. -1 to use original resolution. ')

args = parser.parse_args()
config = vars(args)
config['enable_long_term'] = False
config['enable_long_term_count_usage'] = False
print(config['enable_long_term'])

if args.output is None:
    args.output = f'../output/{args.dataset}_{args.split}'
    print(f'Output path not provided. Defaulting to {args.output}')

davis_path = args.davis
out_path = args.output
save_mask = args.save_mask

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(davis_path + '/trainval/Annotations/480p/blackswan/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Setup Dataset
test_dataset = DAVISTestDataset(davis_path+'/trainval', imset='2017/val.txt')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

images = {}
num_objects = {}
# Loads all the images
for data in test_loader:
    rgb = data['rgb']
    k = len(data['info']['labels'][0])
    name = data['info']['name'][0]
    images[name] = rgb
    num_objects[name] = k
print('Finished loading %d sequences.' % len(images))

# Load our checkpoint
# prop_saved = torch.load(args.prop_model)
# prop_model = PropagationNetwork().cuda().eval()
# prop_model.load_state_dict(prop_saved)
xmem = XMem(config, args.model).cuda().eval()
if args.model is not None:
    model_weights = torch.load(args.model)
    xmem.load_weights(model_weights, init_as_zero_if_needed=True)

fusion_saved = torch.load(args.fusion_model)
fusion_model = FusionNet().cuda().eval()
fusion_model.load_state_dict(fusion_saved)

s2m_saved = torch.load(args.s2m_model)
s2m_model = S2M().cuda().eval()
s2m_model.load_state_dict(s2m_saved)

total_iter = 0
user_iter = 0
last_seq = None
pred_masks = None
with DavisInteractiveSession(davis_root=davis_path+'/trainval', report_save_dir='../output', max_nb_interactions=8, max_time=8*30) as sess:
    while sess.next():
        sequence, scribbles, new_seq = sess.get_scribbles(only_last=True)
        # if sequence[0] not in 'ijklmnopqrstuvwxyz':
        #     T= images[sequence].shape[1]
        #     H,W = images[sequence].shape[-2:]
        #     sess.submit_masks(np.ones([T,H,W]), [1])
        #     print(sequence)
        #     continue

        if new_seq:
            if 'processor' in locals():
                # Note that ALL pre-computed features are flushed in this step
                # We are not using pre-computed features for the same sequence with different user-id
                del processor # Should release some juicy mem
            processor = DAVISProcessor(args, config, xmem, fusion_model, s2m_model, images[sequence], num_objects[sequence])
            print(sequence)

            # Save last time
            if save_mask:
                if pred_masks is not None:
                    seq_path = path.join(out_path, str(user_iter), last_seq)
                    os.makedirs(seq_path, exist_ok=True)
                    for i in range(len(pred_masks)):
                        img_E = Image.fromarray(pred_masks[i])
                        img_E.putpalette(palette)
                        img_E.save(os.path.join(seq_path, '{:05d}.png'.format(i)))

                if (last_seq is None) or (sequence != last_seq):
                    last_seq = sequence
                    user_iter = 0
                else:
                    user_iter += 1

        pred_masks, next_masks, this_idx = processor.interact(scribbles)
        sess.submit_masks(pred_masks, next_masks)

        total_iter += 1

    report = sess.get_report()
    summary = sess.get_global_summary(save_file=path.join(out_path, 'summary.json'))
