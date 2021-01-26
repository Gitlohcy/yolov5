from .general import *

def yoloLabel_lines_2_list(img, yolo_lines):
    ih, iw , _ = img.shape
    yolo_lines = np.array(yolo_lines).astype('float')
    yolo_lines[:, 1:5] = (yolo_lines[:, 1:5] * [iw, ih, iw, ih])
    
    return yolo_lines.astype('int')

def f_readlines(fname):
    with open(str(fname), 'r') as f:
        lines = f.read().splitlines()

    return np.array([line.split(',') for line in lines])

def f_readlabels(img, fname):
    yolo_lines = f_readlines(fname)

    labels = yoloLabel_lines_2_list(img, yolo_lines)
    return labels

def lbl_from_img_name(lbl_dest, img_fname):
    fname = Path(img_fname).stem
    lbl_path = lbl_dest/(fname + '.txt')
    return lbl_path