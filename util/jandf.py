import os
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from math import floor

def db_eval_iou(annotation,segmentation):

    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
 """

    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)

def db_eval_boundary(foreground_mask,gt_mask,bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.

    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask);
    gt_boundary = seg2bmap(gt_mask);

    from skimage.morphology import binary_dilation,disk

    fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg     = np.sum(fg_boundary)
    n_gt     = np.sum(gt_boundary)

    #% Compute precision and recall
    if n_fg == 0 and  n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0  and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match)/float(n_fg)
        recall    = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2*precision*recall/(precision+recall);

    return F

def seg2bmap(seg,width=None,height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]

    Returns:
        bmap (ndarray):	Binary boundary map.

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
 """

    seg = seg.astype(np.bool)
    seg[seg>0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width  = seg.shape[1] if width  is None else width
    height = seg.shape[0] if height is None else height

    h,w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
            'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:,:-1]    = seg[:,1:]
    s[:-1,:]    = seg[1:,:]
    se[:-1,:-1] = seg[1:,1:]

    b        = seg^e | seg^s | seg^se
    b[-1,:]  = seg[-1,:]^e[-1,:]
    b[:,-1]  = seg[:,-1]^s[:,-1]
    b[-1,-1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height,width))
        for x in range(w):
            for y in range(h):
                if b[y,x]:
                    j = 1+floor((y-1)+height / h)
                    i = 1+floor((x-1)+width  / h)
                    bmap[j,i] = 1;

    return bmap



# lab_root = r'F:\dataset\segmentation\DAVIS\2017\DAVIS-2017-trainval-480p\DAVIS\Annotations\480p'
# root = r'C:\Users\chen\Desktop\D17_VAL_I2V'
# cla = os.listdir(root)
# for i in cla:
#     print(i)
# with open(r"C:\Users\chen\Desktop\res\mivos.txt","w") as f:
#     for cl in cla:
#         ann = os.path.join(root, cl)
#         f_num = 0
#         j_val = 0
#         f_val = 0
#
#         for i in range(1,len(glob.glob(ann+'//*.png'))):
#             ann_path = glob.glob(ann+'//*.png')[i]
#             fram = os.path.basename(ann_path)
#             lab_path = os.path.join(lab_root,cl,f'{i*5:06d}.png')
#             ann_img = np.array(Image.open(ann_path).convert('L'))*255
#             av= Image.fromarray(ann_img)
#             if not os.path.exists(os.path.join(r'C:\Users\chen\Desktop\test',cl)):
#                 os.makedirs(os.path.join(r'C:\Users\chen\Desktop\test',cl))
#             av.save(os.path.join(r'C:\Users\chen\Desktop\test',cl,'{:06d}.png'.format(i)))
            # lab_img = np.array(Image.open(lab_path).convert('L'))//255
            # # lab_img = read_external_image(lab_path, size=(480, 853))//255
            # f_num +=1
            # j_val += db_eval_iou(ann_img,lab_img)
            # f_val += db_eval_boundary(ann_img,lab_img)
        # print(f'{cl}\n'
        #       f'Jaccard:{j_val/f_num:.3f}\n'
        #       f'F-score:{f_val/f_num:.3f}')
        # f.write(f'{cl}\n'
        #       f'Jaccard:{j_val/f_num:.3f}\n'
        #       f'F-score:{f_val/f_num:.3f}\n')


# with open(r"C:\Users\chen\Desktop\result\res_davis\i2v_d17_val.txt","w") as f:
# for cl in tqdm(cla):
#     ann = os.path.join(root, cl)
#     f_num = 0
#     j_val = 0
#     f_val = 0
#     for ann_path in glob.glob(ann+'//*.png'):
#         fram = os.path.basename(ann_path)
#         lab_path = os.path.join(lab_root,cl,fram)
#         lab_img = np.array(Image.open(lab_path))
#         ann_img = np.array(Image.open(ann_path))
#         f_num +=1
#         j_val += db_eval_iou(ann_img,lab_img)
#         f_val += db_eval_boundary(ann_img,lab_img)
#         ann_img = Image.fromarray(ann_img)
#             # ann_img.save(os.path.join(a,fram))
#         # f.write(f'{cl}\n'
#         #       f'Jaccard:{j_val/f_num:.3f}\n'
#         #       f'F-score:{f_val/f_num:.3f}\n')
#
