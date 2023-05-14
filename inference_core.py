"""
Heart of most evaluation scripts (DAVIS semi-sup/interactive, GUI)
Handles propagation and fusion
See eval_semi_davis.py / eval_interactive_davis.py for examples
"""

import math
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from model.propagation.prop_net import PropagationNetwork
from model.fusion_net import FusionNet
from model.aggregate import aggregate_wbg

from util.tensor_util import pad_divide_by
from model.network import XMem
from xm_inference_core import XmemInferenceCore
from inference.data.mask_mapper import MaskMapper

class InferenceCore:
    """
    images - leave them in original dimension (unpadded), but do normalize them.
            Should be CPU tensors of shape B*T*3*H*W

    mem_profile - How extravagant I can use the GPU memory.
                Usually more memory -> faster speed but I have not drawn the exact relation
                0 - Use the most memory
                1 - Intermediate, larger buffer
                2 - Intermediate, small buffer
                3 - Use the minimal amount of GPU memory
                Note that *none* of the above options will affect the accuracy
                This is a space-time tradeoff, not a space-performance one

    mem_freq - Period at which new memory are put in the bank
                Higher number -> less memory usage
                Unlike the last option, this *is* a space-performance tradeoff
    """

    def __init__(self, config, XMem: XMem, fuse_net: FusionNet, images, num_objects,
                 mem_profile=0, device='cuda:0'):
        #image: 1 T 3 h w

        XMem.to(device, non_blocking=True)
        self.xmprocessor = XmemInferenceCore(XMem, config=config)
        # self.prop_net = prop_net.to(device, non_blocking=True)
        if fuse_net is not None:
            self.fuse_net = fuse_net.to(device, non_blocking=True)
        self.mem_profile = mem_profile
        # self.mem_freq = mem_freq
        self.device = device

        if mem_profile == 0:
            self.data_dev = device
            self.result_dev = device
            self.i_buf_size = -1  # no need to buffer image
        elif mem_profile == 1:
            self.data_dev = 'cpu'
            self.result_dev = device
            self.i_buf_size = 105
        elif mem_profile == 2:
            self.data_dev = 'cpu'
            self.result_dev = 'cpu'
            self.i_buf_size = 3
        else:
            self.data_dev = 'cpu'
            self.result_dev = 'cpu'
            self.i_buf_size = 1

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]
        self.shape = (h, w)
        self.k = num_objects

        # Pad each side to multiples of 16
        self.images, self.pad = pad_divide_by(images, 16, images.shape[-2:])
        # Padded dimensions
        nh, nw = self.images.shape[-2:]
        self.images = self.images.to(self.data_dev, non_blocking=False)

        # These two store the same information in different formats
        self.masks = torch.zeros((t, 1, nh, nw), dtype=torch.uint8, device=self.result_dev)
        self.np_masks = np.zeros((t, h, w), dtype=np.uint8)

        # Object probabilities, background included
        self.prob = torch.zeros((self.k + 1, t, 1, nh, nw), dtype=torch.float32, device=self.result_dev)
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh // 16
        self.kw = self.nw // 16

        self.image_buf = {}
        self.interacted = set()

        self.mapper = MaskMapper()
        self.enable_set_label = True

    def get_image_buffered(self, idx):
        if self.data_dev == self.device:
            return self.images[:, idx]

        # buffer the .cuda() calls
        if idx not in self.image_buf:
            # Flush buffer
            if len(self.image_buf) > self.i_buf_size:
                self.image_buf = {}
        self.image_buf[idx] = self.images[:, idx].to(self.device)
        result = self.image_buf[idx]

        return result

    def set_label(self,mask_dim):
        self.mapper.convert_mask(mask_dim.cpu().numpy())
        # a = np.unique(mask_dim.cpu().numpy()).astype(np.uint8)
        mask_real = np.zeros((self.k, self.h, self.w))
        for i in range(self.k, 0, -1):
            mask_real[i - 1] = mask_dim.cpu().numpy().copy()
            np.place(mask_real[i - 1], mask_real[i - 1] != i, 0)
            np.place(mask_real[i - 1], mask_real[i - 1] == i, 1)
            # a = np.unique(mask_real[i-1]).astype(np.uint8)
        self.xmprocessor.set_all_labels(list(self.mapper.remappings.values()))

    def do_pass(self, idx, mask_dim, forward=True,step_cb=None):
        """
        Do a complete pass that includes propagation and fusion
        key_k/key_v -  memory feature of the starting frame
        idx - Frame index of the starting frame
        forward - forward/backward propagation
        step_cb - Callback function used for GUI (progress bar) only
        """

        # Determine the required size of the memory bank
        if forward:
            closest_ti = min([ti for ti in self.interacted if ti > idx] + [self.t])
        else:
            closest_ti = max([ti for ti in self.interacted if ti < idx] + [-1])
        # Note that we never reach closest_ti, just the frame before it
        if forward:
            this_range = range(idx + 1, closest_ti)
        else:
            this_range = range(idx - 1, closest_ti, -1)


        # self.mapper.convert_mask(mask_dim.cpu().numpy())
        # # a = np.unique(mask_dim.cpu().numpy()).astype(np.uint8)
        mask_real= np.zeros((self.k,self.h,self.w))
        for i in range(self.k,0,-1):
            mask_real[i-1] = mask_dim.cpu().numpy().copy()
            np.place(mask_real[i - 1],mask_real[i - 1] != i, 0)
            np.place(mask_real[i - 1], mask_real[i - 1] == i, 1)
            # a = np.unique(mask_real[i-1]).astype(np.uint8)
        # self.xmprocessor.set_all_labels(list(self.mapper.remappings.values()))
        mask_real =torch.FloatTensor(mask_real).to(self.data_dev, non_blocking=False)
        self.xmprocessor.step(self.images[:, idx].squeeze(0), mask_real)

        for ti in this_range:

            out_mask = self.xmprocessor.step(self.images[:, ti].squeeze(0))
            out_mask = out_mask.unsqueeze(1)

            # out_mask = F.interpolate(out_mask.unsqueeze(1), self.shape, mode='bilinear', align_corners=False)[:, 0]


            # In-place fusion, maximizes the use of queried buffer
            # esp. for long sequence where the buffer will be flushed
            if (closest_ti != self.t) and (closest_ti != -1):
                # w_buffer_past = self.get_image_buffered(idx)
                # w_buffer_curr = self.get_image_buffered(ti)
                # k16,_,_,_,_,_ = self.xmprocessor.network.encode_key(w_buffer_past)
                # q16,_,_,_,_,_, = self.xmprocessor.network.encode_key(w_buffer_curr)
                self.prob[:, ti] = self.fuse_one_frame(closest_ti, idx, ti, self.prob[:, ti], out_mask,
                                                       ).to(self.result_dev)
            else:
                self.prob[:, ti] = out_mask.to(self.result_dev)

            if step_cb is not None:
                step_cb()
        self.mapper.mv_labels()
        return closest_ti

    def fuse_one_frame(self, tc, tr, ti, prev_mask, curr_mask):
        assert (tc < ti < tr or tr < ti < tc)

        prob = torch.zeros((self.k, 1, self.nh, self.nw), dtype=torch.float32, device=self.device)

        # Compute linear coefficients
        nc = abs(tc - ti) / abs(tc - tr)
        nr = abs(tr - ti) / abs(tc - tr)
        dist = torch.FloatTensor([nc, nr]).to(self.device).unsqueeze(0)
        for k in range(1, self.k + 1):
            attn_map = self.get_attention(self.pos_mask_diff[k:k + 1],
                                                   self.neg_mask_diff[k:k + 1])
            # b, _, h, w =self.pos_mask_diff[k:k + 1].shape
            # attn_map = torch.zeros(b, 2, h, w)

            w = torch.sigmoid(self.fuse_net(self.get_image_buffered(ti),
                                            prev_mask[k:k + 1].to(self.device), curr_mask[k:k + 1].to(self.device),
                                            attn_map, dist))
            prob[k - 1] = w
        return aggregate_wbg(prob, keep_bg=True)

    def interact(self, mask, idx,total_cb=None, step_cb=None):
        """
        Interact -> Propagate -> Fuse

        mask - One-hot mask of the interacted frame, background included
        idx - Frame index of the interacted frame
        total_cb, step_cb - Callback functions for the GUI

        Return: all mask results in np format for DAVIS evaluation
        """
        self.interacted.add(idx)

        mask = mask.to(self.device)
        mask, _ = pad_divide_by(mask, 16, mask.shape[-2:])#在xmeminfer里有
        self.mask_diff = mask - self.prob[:, idx].to(self.device)
        self.pos_mask_diff = self.mask_diff.clamp(0, 1)
        self.neg_mask_diff = (-self.mask_diff).clamp(0, 1)
        mask_dim = torch.argmax(mask.squeeze(1), dim=0)
        # a = np.unique(mask_dim.cpu().numpy()).astype(np.uint8)
        # print(mask.tolist()[0])
        self.prob[:, idx] = mask
        if self.enable_set_label:
            self.set_label(mask_dim)
            self.enable_set_label = False

        if total_cb is not None:
            # Finds the total num. frames to process
            front_limit = min([ti for ti in self.interacted if ti > idx] + [self.t])
            back_limit = max([ti for ti in self.interacted if ti < idx] + [-1])
            total_num = front_limit - back_limit - 2 # -1 for shift, -1 for center frame
            if total_num > 0:
                total_cb(total_num)


        self.do_pass(idx, mask_dim, True,step_cb=step_cb)
        self.do_pass(idx, mask_dim, False,step_cb=step_cb)

        # This is a more memory-efficient argmax
        for ti in range(self.t):
            self.masks[ti] = torch.argmax(self.prob[:, ti], dim=0)
        out_masks = self.masks

        # Trim paddings
        if self.pad[2] + self.pad[3] > 0:
            out_masks = out_masks[:, :, self.pad[2]:-self.pad[3], :]
        if self.pad[0] + self.pad[1] > 0:
            out_masks = out_masks[:, :, :, self.pad[0]:-self.pad[1]]

        self.np_masks = (out_masks.detach().cpu().numpy()[:, 0]).astype(np.uint8)

        return self.np_masks

    def update_mask_only(self, prob_mask, idx):
        """
        Interaction only, no propagation/fusion
        prob_mask - mask of the interacted frame, background included
        idx - Frame index of the interacted frame

        Return: all mask results in np format for DAVIS evaluation
        """
        mask = torch.argmax(prob_mask, 0)
        self.masks[idx] = mask

        # Mask - 1 * H * W
        if self.pad[2] + self.pad[3] > 0:
            mask = mask[:, self.pad[2]:-self.pad[3], :]
        if self.pad[0] + self.pad[1] > 0:
            mask = mask[:, :, self.pad[0]:-self.pad[1]]

        mask = (mask.detach().cpu().numpy()[0]).astype(np.uint8)
        self.np_masks[idx] = mask

        return self.np_masks

    def get_attention(self, pos_mask, neg_mask):
        b, _, h, w = pos_mask.shape
        nh = h // 16
        nw = w // 16

        # W = self.get_W(mk16, qk16)

        pos_map = (F.interpolate(pos_mask, size=(nh, nw), mode='area').view(b, 1, nh * nw))
        neg_map = (F.interpolate(neg_mask, size=(nh, nw), mode='area').view(b, 1, nh * nw))
        attn_map = torch.cat([pos_map, neg_map], 1)
        attn_map = attn_map.reshape(b, 2, nh, nw)
        attn_map = F.interpolate(attn_map, mode='bilinear', size=(h, w), align_corners=False)

        return attn_map

    def get_w(self, mk, qk):
        """
        T=1 only. Only needs to obtain W
        """
        B, CK, _, H, W = mk.shape

        mk = mk.view(B, CK, H * W)
        mk = torch.transpose(mk, 1, 2)  # B * HW * CK

        qk = qk.view(1, CK, H * W).expand(B, -1, -1) / math.sqrt(CK)  # B * CK * HW

        affinity = torch.bmm(mk, qk)  # B * HW * HW
        affinity = F.softmax(affinity, dim=1)

        return affinity

