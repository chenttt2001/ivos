from argparse import ArgumentParser
from model.fusion_net import FusionNet
from dataset.gui_read_images import get_tensor_images
from inference_core import InferenceCore
from model.network import XMem
from model.s2m.s2m_network import deeplabv3plus_resnet50 as S2M

import torch

def get_masks(config,args,images,mask_proba,interactived_idx):
    xmem = XMem(config, args.model).cuda().eval()
    if args.model is not None:
        model_weights = torch.load(args.model)
        xmem.load_weights(model_weights, init_as_zero_if_needed=True)

    s2m_model_path = r'saves/s2m.pth'
    s2m_saved = torch.load(s2m_model_path)
    s2m_model = S2M().cuda().eval()
    s2m_model.load_state_dict(s2m_saved)

    fusion_saved = torch.load(args.fusion_model)
    fusion_model = FusionNet().cuda().eval()
    fusion_model.load_state_dict(fusion_saved)

    processor = InferenceCore(config, xmem, fusion_model, images.unsqueeze(0), config['num_objects'], mem_profile=0,
                              device='cuda:0')

    processor.interact(mask_proba, interactived_idx[-1])