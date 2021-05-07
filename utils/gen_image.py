import argparse
import re
import random
from pycocotools.coco import COCO

from .yolo_utils.general import *
import yolo_utils.imgaug_util as iu
import yolo_utils.plot_util as pu
import yolo_utils.file_util as fu
import yolo_utils.bbox_util as bu

import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug.augmenters as iaa
import boxx


def get_batch_anno(anns_df, imgs_df, img_dir, num_to_paste, weights=None):
    '''Return:
    - image_path : Path
    - segmentation_mask : np.ndarray
    - class_id : int
    '''

    df = anns_df.sample(num_to_paste)
    df['image_path'] = df['image_id'].apply(
        lambda x: img_dir/imgs_df.iloc[x]['file_name'])

    return df[['image_path', 'mask', 'category_id']]


def get_img_lbls(row):
    img = imageio.imread(row['image_path'])
    segmap = iu.get_segmap(img, row['mask'])

    return img, segmap, row['category_id']


def bbox_from_pasted_mask(small_img_bool_mask, big_img_paste_coord):

    # tlbr on small img mask
    s_x1, s_y1, s_x2, s_y2 = bbox_from_mask(small_img_bool_mask)

    # coordinate of where small img mask paste on big img
    b_x1, b_y1 = big_img_paste_coord  # top left xy

    # tlbr of big img mask (pasted with small img mask)
    b_mask_x1 = b_x1 + s_x1
    b_mask_y1 = b_y1 + s_y1
    b_mask_x2 = b_x1 + s_x2
    b_mask_y2 = b_y1 + s_y2

    return [b_mask_x1, b_mask_y1, b_mask_x2, b_mask_y2]


def bbox_from_mask(bool_mask):
    mask_y, mask_x = np.array(np.where(bool_mask))
    x1 = np.min(mask_x)
    y1 = np.min(mask_y)
    x2 = np.max(mask_x)
    y2 = np.max(mask_y)

    return [x1, y1, x2, y2]

def bbox_from_mask2(bool_mask):

    mask_y, mask_x = np.array(np.where(bool_mask))

    x1 = np.min(mask_x)
    y1 = np.min(mask_y)
    x2 = np.max(mask_x) + 1 # unsolve +1
    y2 = np.max(mask_y) + 1 # unsolve +1


    return [x1, y1, x2, y2]

def part_of_img(img, bbox):
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2, :]

def bbox_from_pasted_mask2(small_img_bool_mask, big_img_paste_coord):

    #tlbr on small img mask
    s_x1, s_y1, s_x2, s_y2 = bbox_from_mask2(small_img_bool_mask)
    
    #coordinate of where small img mask paste on big img 
    b_x1, b_y1 = big_img_paste_coord #top left xy
    
    #tlbr of big img mask (pasted with small img mask)
    b_mask_x1 = b_x1 + s_x1
    b_mask_y1 = b_y1 + s_y1
    b_mask_x2 = b_x1 + s_x2
    b_mask_y2 = b_y1 + s_y2
    
    return [b_mask_x1, b_mask_y1,b_mask_x2, b_mask_y2]

def paste_into_img2(front_img, back_img, mid_xy, bool_mask=None):
    
    fih, fiw, _ = front_img.shape
    bih, biw, _ = back_img.shape
    
    x1, y1, x2, y2 = list(map(int, xywh2xyxy([*tuple(mid_xy), fiw, fih])))
    if isinstance(bool_mask, np.ndarray):
        back_img[y1:y2, x1:x2, :][bool_mask] = front_img[bool_mask]
        bbox = bbox_from_pasted_mask2(bool_mask, [x1,y1])
    else:    
        back_img[y1:y2, x1:x2, :] = front_img
        bbox = [x1, y1, x2, y2]
    
    return back_img, bbox

def crop_by_bbox(img, bbox, mask=None):
    x1, y1, x2, y2 = bbox

    if isinstance(mask, np.ndarray):
        return img[y1:y2, x1:x2, :], mask[y1:y2, x1:x2]
    else:
        return img[y1:y2, x1:x2, :]


def xywh2xyxy(bbox):
    x_mid, y_mid, w, h = bbox

    x1 = x_mid - (w/2)
    y1 = y_mid - (h/2)
    x2 = x_mid + (w/2)
    y2 = y_mid + (h/2)

    return [x1, y1, x2, y2]


def paste_into_img(front_img, back_img, mid_xy, bool_mask=None):

    fih, fiw, _ = front_img.shape
    bih, biw, _ = back_img.shape

    x1, y1, x2, y2 = list(map(int, xywh2xyxy([*tuple(mid_xy), fiw, fih])))
    if isinstance(bool_mask, np.ndarray):
        back_img[y1:y2, x1:x2, :][bool_mask] = front_img[bool_mask]
        bbox = bbox_from_pasted_mask(bool_mask, [x1, y1])
    else:
        back_img[y1:y2, x1:x2, :] = front_img
        bbox = [x1, y1, x2, y2]

    return back_img, bbox


def batch_resize(scale_ratio):
    '''
    scale_ratio: Double = (0~1)
    '''
    return iaa.Resize(scale_ratio)


def depth_level_resize(img_num, min_size=0.15, max_size=1.0):
    return [iaa.Resize(resize_ratio) for resize_ratio in np.linspace(min_size, max_size, num=img_num)]


def aug_front_img(img, segmap, aug_seq):
    #     resize_small_img = iaa.Resize({"shorter-side": "keep-aspect-ratio", "longer-side":1080})
    if not isinstance(segmap, ia.augmentables.segmaps.SegmentationMapsOnImage):
        segmap = iu.get_segmap(img, segmap)

    resized_img, resized_segmap = aug_seq(image=img, segmentation_maps=segmap)
    resized_mask = (resized_segmap.get_arr() > 0)
    bbox = bbox_from_mask(resized_mask)

    cropped_img, cropped_mask = crop_by_bbox(
        resized_img, bbox, mask=resized_mask)
    cropped_segmap = iu.get_segmap(cropped_img, cropped_mask)

    return cropped_img, cropped_segmap


def pad_back_img_to_fixedSize(back_img, max_front_img_wh, pad_cval=255, mode='Center'):
    '''
    mode = ['RandomEdge', 'Center']
    '''
    bih, biw, _ = back_img.shape
    siw, sih = max_front_img_wh

    if mode == 'Center':
        return iaa.CenterPadToFixedSize(biw+siw, bih+sih, pad_cval=pad_cval)(image=back_img)
    elif mode == 'RandomEdge':
        return iaa.PadToFixedSize(biw+siw, bih+sih, pad_cval=pad_cval)(image=back_img)


def rand_neg(limit: int, size: int = None):
    '''random with range negative limit until limit

    Args:
    -----

    size: set ndim of array return
        if 0, int number is return, if higher than 0, ndim of array with the given size is return

    Example:
    ->[rand_neg(10) for _ in range(10)]
    ->[2, 3, -4, -1, -5, 1, -1, -1, 2, -4]

    ->rand_neg(10, size=1)
    ->array([-2])

    ->rand_neg(10, size=0)

    rand_neg(10, size=1)
    '''
    if size:
        return np.random.randint(limit*2, size=size) - limit
    else:
        return np.random.randint(limit*2) - limit


def rand_neg_with_positive_higher_chance(limit, chance_increase=0.2):
    '''random with range negative limit until limit
    Example:
    ->[np.random.randint(10) - 5 for _ in range(10)]
    ->[2, 3, -4, -1, -5, 1, -1, -1, 2, -4]
    '''
    return np.random.randint(limit*2) - limit + (limit * chance_increase)


def imgs_wh(imgs):
    """Return the width and height for list of imgs

    Args:
        imgs: List[np.ndarray (w, h, 3)]
    """
    heights, widths, _ = np.array([img.shape for img in imgs]).T
    return widths, heights


def get_rand_offset(back_img, paste_num):
    ih, iw, _ = back_img.shape
    return np.array([[rand_neg(iw/4), rand_neg(ih/4)] for i in range(paste_num)])


def get_cells_coord(big_img, quarter_grid=True):
    if quarter_grid:
        bih, biw, _ = big_img.shape

        mid_x, mid_y = (biw/2, bih/2)  # center of image
        qw, qh = (biw/4, bih/4)       # quarter (width, height)

        center_cells = []
        center_cells.append([mid_x-qw, mid_y-qh])
        center_cells.append([mid_x+qw, mid_y-qh])
        center_cells.append([mid_x-qw, mid_y+qh])
        center_cells.append([mid_x+qw, mid_y+qh])

        return center_cells


def get_rand_cell_coords(back_img, paste_num):
    mid_coords = get_cells_coord(back_img, quarter_grid=True)
    offsets = get_rand_offset(back_img, paste_num)

    round_num = (paste_num // len(mid_coords)) + 1
    rmd = paste_num % len(mid_coords)  # remainders

    round_coords = mid_coords.copy()
    rand_coords = []
    for n in range(round_num):
        np.random.shuffle(round_coords)
        # paste remaining for last round
        round_paste = rmd if n == (round_num-1) else len(round_coords)

        for coord in round_coords[:round_paste]:
            coord = np.array(coord)
            coord = coord + offsets[np.random.randint(10)]
            rand_coords.append(coord)

    return np.array(rand_coords)


def map_instance_mask(fg_mask, bg_mask, mid_xy, map_value):
    fih, fiw = fg_mask.shape
    bih, biw = bg_mask.shape

    x1, y1, x2, y2 = list(map(int, xywh2xyxy([*tuple(mid_xy), fiw, fih])))
    # background = 0, map_value start from 1
    bg_mask[y1:y2, x1:x2][fg_mask] = (map_value+1)

    return bg_mask


def mask_overlaped_area(pasted_mask_area, ori_mask_area):
    '''
    mask_area = bool_mask.sum()
    '''
    return 1 - (pasted_mask_area / ori_mask_area)

def remove_overlaped_bbox(thereshold: float, batch_labeled_imgs: list, rand_coords: Union[list, np.ndarray],
                          bg_img: np.ndarray,
                          ) -> np.ndarray:
    '''
    remove the bbox after pasting sequentially,
    the precedence pasted product is more likely
    to be overlaped.
    Calculate the overlapped mask
    area before pasting and the remaining area
    after all the pasting is done.
    Remove the bbox if the covered area percentage reach the overlap thereshold
    '''

    # find orignial mask areas
    product_mask_areas = []
    bg_mask = np.zeros_like(bg_img[:, :, 0])
    for i, (img, segmap, cls_id) in enumerate(batch_labeled_imgs):
        fg_mask = segmap.get_arr()
        product_mask_areas.append(fg_mask.sum())
        bg_mask = map_instance_mask(fg_mask, bg_mask, rand_coords[i], i)

    # find list of bbox to be remove
    bbox_to_remove = []
    for i, ori_mask_area in enumerate(product_mask_areas):
        pasted_mask_area = (bg_mask == i+1).sum()
        is_overlap_tooMuch = (mask_overlaped_area(
            pasted_mask_area, ori_mask_area) >= thereshold)
        bbox_to_remove.append(is_overlap_tooMuch)

    return np.array(bbox_to_remove)


def append_selective_aug(aug_seqs: List, cls_id, aug_funcs_dict, chance=0.5):
    catId_related_aug = aug_funcs_dict.get(cls_id, None)
    if catId_related_aug:
        [aug_seqs.append(iaa.Sometimes(chance, aug))
             for aug in catId_related_aug]
        
    return aug_seqs

def color_chnl_adjust(chnl, values):

    return iaa.WithChannels(
        chnl,
        iaa.Add(values),
    )

def rand_read_img(img_paths: list):
    img_id = random.randint(0, len(img_paths)-1)
    return imageio.imread(img_paths[img_id])


def read_background_img(img_paths: list):
    valid_back_img = False
    back_img = rand_read_img(img_paths)

    while img_paths and not valid_back_img:
        if back_img.ndim == 3:
            return back_img
        else:
            back_img = rand_read_img(img_paths)
    else:
        raise IndexError("empty img_paths given")


def mkdir_if_notExist(path, parents=True):
    if not path.is_dir():
        path.mkdir(parents=parents)

def bbs_to_yoloList(bbs: BoundingBoxesOnImage) -> List:
    """convert object of BoundingBoxesOnImage into yolo format label"""
    return np.column_stack((
        [bb.label for bb in bbs.bounding_boxes],
        bbs.to_xyxy_array())
    ).astype('float32')


#---motion blur---
def thicken(mask_edge, iterations=2):
    kernel = np.ones((3,3), np.uint8)
    return cv2.dilate(mask_edge.astype('uint8'), kernel, iterations=iterations)

def int_bb_arr(bb):
    '''get int array of BoundingBox'''
    return [bb.x1_int, bb.y1_int, bb.x2_int, bb.y2_int]

def resolve_shape_rounding(a1, a2):
    def until_shape(big_a, small_a):
        return big_a[:small_a.shape[0], :small_a.shape[1]]

    def solve_rounding(a1, a2, idx):
        if a1.shape[idx] > a2.shape[idx]:
            a1 = until_shape(a1, a2)
        elif a1.shape[idx] < a2.shape[idx]:
            a2 = until_shape(a2, a1)
        return a1, a2
    
    if a1.ndim >= 2 and a2.ndim>=2:
        #resolve arr size different
        a1, a2 = solve_rounding(a1, a2, 0)
        a1, a2 = solve_rounding(a1, a2, 1)                    
    return a1, a2

def aug_during_paste(back_img, bbox, bool_mask, aug_func, dilation=1.1, active_chance=0.8):

    # is_active
    if random.choices([True, False], weights=[active_chance, 1-active_chance], k=1)[0]:
    
        bbox = BoundingBox(*bbox)

        if dilation < 1.0:
            raise ValueError('dilation area need to >= 1 to cover the entire mask')
        expanded_mask = iaa.Resize(dilation)(image=bool_mask)



        expanded_bbox = BoundingBox(*bbox_from_mask2(expanded_mask))
        half_w, half_h = (expanded_bbox.width/2), (expanded_bbox.height/2)
        new_bbox = BoundingBox(*[
            bbox.center_x-half_w, bbox.center_y-half_h,  # x1,y1
            bbox.center_x+half_w, bbox.center_y+half_h   # x2,y2
        ]).clip_out_of_image(back_img)

    #     print(f"expanded_mask:  {expanded_mask.shape}")
    #     print(f"expanded_bbox:  {expanded_bbox}")
    #     print(f"new_bbox:  {new_bbox}")
        part_img = part_of_img(back_img, int_bb_arr(new_bbox))


        #solve rounding issue
    #     print('shape_diff:', np.array(part_img.shape)[:1] - np.array(expanded_mask.shape)[:1])
        part_img, expanded_mask = resolve_shape_rounding(part_img, expanded_mask)
    #     print('shape_diff:', np.array(part_img.shape)[:1] - np.array(expanded_mask.shape)[:1])

    #     if (part_img.shape[1] - expanded_mask.shape[1] != 0 or 
    #         part_img.shape[0] - expanded_mask.shape[0] != 0):
    #         st()

        aug_img = aug_func(image=part_img.copy())
        part_img[expanded_mask] = aug_img[expanded_mask]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num2gen', type=int, default=10,
                        help='num of image to generate')
    parser.add_argument('--data', type=str,
                        help='yaml path to load data paths',
                        default='data.yaml')
    parser.add_argument('--hyp', type=str,
                        help='yaml path to load hyp paths',
                        default='hyp.yaml')
    
    
    opt = parser.parse_args()

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    with open(opt.hyp) as f:
        hyp_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # input
    back_img_path = Path(data_dict['back_img'])
    front_img_path = Path(data_dict['front_img'])
    coco_path = Path(data_dict['coco_path'])

    # output
    img_dest = Path(data_dict['img_dest'])
    lbl_dest = Path(data_dict['lbl_dest'])
    
   
    img_file_type = ['.jpg', '.png', '.jpeg']
    print('reading back_img paths...')
    back_img_list = img_list_from(back_img_path, img_file_type)
    print('reading front_img paths...')
    front_img_list = img_list_from(front_img_path, img_file_type)
    small_img_paths_dict = {p.name: p for p in list(map(Path, front_img_list))}

    assert coco_path.is_file() and coco_path.suffix == '.json', \
        f"{coco_path} is not a valid json file"


    coco = COCO(coco_path)
    coco_json = boxx.loadjson(coco_path)
    boxx.tree(coco_json, deep=1)

    coco_df = {}
    coco_df['images'] = pd.DataFrame(coco_json['images'])
    coco_df['annotations'] = pd.DataFrame(coco_json['annotations'])
    coco_df['categories'] = pd.DataFrame(coco_json['categories'])

    allCatIds = coco.getCatIds()
    # filtered_catIds = coco.getCatIds(catNms=product_to_gen)

    # get mask annotation
    annIds = coco.getAnnIds(catIds=allCatIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    anns_df = pd.DataFrame(anns)
    print('Converting polygon points to mask...')
    anns_df['mask'] = [coco.annToMask(ann) for ann in tqdm(anns)]

    mkdir_if_notExist(lbl_dest)
    mkdir_if_notExist(img_dest)


    ## assign tuple for range between (min, max) in imgaug library
    keys_with_range = ['rotate', 'batch_resize']
    for c in 'hsv':
        hyp_dict['hsv_'+c][1] = tuple(hyp_dict['hsv_'+c][1])
    
    m_blur_chance = hyp_dict['motion_blur'][0]
    m_blur_kernel = tuple(hyp_dict['motion_blur'][1])
    m_blur_angle = tuple(hyp_dict['motion_blur'][2])


    for k in keys_with_range:
        hyp_dict[k] = tuple(hyp_dict[k])

    ## init data aug params
    adjust_hsv = iaa.WithColorspace(
        to_colorspace="HSV",
        from_colorspace="RGB",
        children=iaa.Sequential([
            iaa.Sometimes(
                hyp_dict['hsv_h'][0],
                color_chnl_adjust(0, hyp_dict['hsv_h'][1]),
                
            ),
            iaa.Sometimes(
                hyp_dict['hsv_s'][0],
                color_chnl_adjust(1, hyp_dict['hsv_s'][1])
            ),
            iaa.Sometimes(
                hyp_dict['hsv_v'][0],
                color_chnl_adjust(2, hyp_dict['hsv_v'][1])
            ),
        ])
    )

    aug_funcs_dict = {
        0:  None,
        1:  None,
        2:  None,
        3:  None,
        4:  None,
        5:  None,
        6:  None,
        7:  None,
        8:  None,
        9:  None,
        10: None,
        11: None,
        12: None,
        13: None,
        14: None,
    }

    motion_blur = iaa.MotionBlur(
                    k=m_blur_kernel,
                    angle=m_blur_angle
                )

    ## data distribution: num of front imgs in single back img
    paste_dist = hyp_dict['paste_dist']
    max_paste_n = hyp_dict['max_paste_n']
    assert len(paste_dist) == max_paste_n, \
        'length of paste_dist nid to be same with maximum paste num'

    n_products_dist = np.array(random.choices(
        range(1, max_paste_n+1), paste_dist, k=10000))

    pasted_back_img_list = []
    clean_bbs_list = []

    ## start generate
    n_products_per_img = random.choices(n_products_dist, k=int(opt.num2gen))
    for n_product in tqdm(n_products_per_img):

        batch_df = get_batch_anno(anns_df, coco_df['images'], front_img_path, n_product)
        img_segmap_cls = list(batch_df.apply(get_img_lbls, axis=1))

        # augmentation on small_imgs
        resize_seqs = [[
            batch_resize(hyp_dict['batch_resize']),
            depth_resize,
            ] for depth_resize in depth_level_resize(len(img_segmap_cls), min_size=hyp_dict['depth_min_size'])]

        augmented_small_imgs = []
        for resize_seq, (img, segmap, cls_id) in zip(resize_seqs, img_segmap_cls):
            aug_seqs = [
                *resize_seq,
                iaa.Rotate(hyp_dict['rotate']),
                adjust_hsv,
            ]
            aug_seqs = append_selective_aug(aug_seqs, cls_id, aug_funcs_dict, hyp_dict['class_aug'])
            aug_seqs = iaa.Sequential(aug_seqs)
            augmented_img, augmented_segmap = aug_front_img(img, segmap, aug_seqs)
            augmented_small_imgs.append([augmented_img, augmented_segmap, cls_id])


        ## back_img padding
        back_img = read_background_img(back_img_list)
        ws, hs = imgs_wh([img for img, _, _ in augmented_small_imgs])
        padded_back_img = pad_back_img_to_fixedSize(
            back_img, (ws.max(), hs.max()))

        ## generate coords
        rand_coords = get_rand_cell_coords(back_img, 10)
        ph, pw, _ = np.array(padded_back_img.shape) - np.array(back_img.shape)
        padded_rand_coords = rand_coords.copy()
        padded_rand_coords += [pw/2, ph/2]  # size of padding for each side

        ## pasting
        bboxes = []
        for i, (img, segmap, cls_id) in enumerate(augmented_small_imgs):
            bool_mask = (segmap.get_arr()>0)
            padded_back_img, bbox = paste_into_img2(
                img, padded_back_img, padded_rand_coords[i], bool_mask)
            aug_during_paste(padded_back_img, bbox, bool_mask, motion_blur, active_chance=m_blur_chance)
            bboxes.append([*bbox, cls_id])

        ## crop
        bbs = iu.get_bbs(padded_back_img, bboxes)
        bih, biw, _ = back_img.shape
        pasted_back_img, pasted_bbs = iaa.CenterCropToFixedSize(height=bih, width=biw)(image=padded_back_img, bounding_boxes=bbs)
        pasted_bbs = pasted_bbs.clip_out_of_image()

        ## remove overlap bbox
        bbox_to_remove = remove_overlaped_bbox(hyp_dict['overlap_thereshold'], augmented_small_imgs, padded_rand_coords, padded_back_img)
        test_pasted_bbs = np.array(pasted_bbs.bounding_boxes)
        clean_bbs = BoundingBoxesOnImage(
            test_pasted_bbs[~bbox_to_remove], pasted_back_img)
        

        bboxes = bbs_to_yoloList(clean_bbs)
        fu.write_img_and_bboxes(pasted_back_img, bboxes, img_dest, lbl_dest)



if __name__ == '__main__':
    run()
