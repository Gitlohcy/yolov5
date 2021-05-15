from .general import *
from .bbox_util import xywh2xyxy, xyxy2xywh
import uuid
import os
import shutil
import threading
import queue


import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug.augmenters as iaa

def mkdir_notExist(p: Path):
    if not p.is_dir():
        p.mkdir(parents=True)

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

#read files
def yoloLabel_lines_2_list(img, yolo_lines):
    ih, iw , _ = img.shape
    yolo_lines = np.array(yolo_lines).astype('float')
    bboxes = yolo_lines[:, 1:5]
    bboxes = bboxes * [iw, ih, iw, ih]
    bboxes = xywh2xyxy(bboxes)
    yolo_lines[:, 1:5] = bboxes

    return yolo_lines.astype('int')

def f_readlines(fname, split_char=None):
    with open(str(fname), 'r') as f:
        lines = f.read().splitlines()

    if split_char:
        return np.array([line.split(split_char) for line in lines])

    return lines

def f_readlabels(img, fname):
    yolo_lines = f_readlines(fname, split_char=' ')
    labels = yoloLabel_lines_2_list(img, yolo_lines)

    return labels

def lbl_from_img_name(lbl_dest, img_fname):
    fname = Path(img_fname).stem
    lbl_path = lbl_dest/(fname + '.txt')
    return lbl_path


# write labels
def uniq_file_name(prefix=None):
    hex_name = uuid.uuid4().hex
    uniq_name = prefix + '_' + hex_name if prefix else  hex_name
    return uniq_name

def list_2_yoloLabel_lines(img, labels_list):
    ih, iw , _ = img.shape
    labels_list = np.array(labels_list).astype('float')
    
    bboxes = labels_list[:, 1:5] 
    bboxes  = (bboxes / [iw, ih, iw, ih]).round(4)
    bboxes = xyxy2xywh(bboxes)
    labels_list[:, 1:5] = bboxes

    return labels_list.astype('str') #str lines


def f_writelines(lines: List[str], fname, join_by=None):
    if join_by is None:
        lines = [str(line) + '\n' for line in lines]
    else:
        lines = [join_by.join(list(line)) +'\n' for line in lines]

    with open(str(fname), 'w') as f:
        f.writelines(lines)

def write_img_and_bboxes(img, labels, img_dest, lbl_dest):
    fname = uniq_file_name('img')
    img_name = fname + '.jpg'
    lbl_name = fname + '.txt'
    
    imageio.imwrite(img_dest/img_name, img)
    yolo_lines = list_2_yoloLabel_lines(img, labels)
    f_writelines(yolo_lines, lbl_dest/lbl_name, join_by=' ')


def img_list_from(img_path, img_file_type):
    """
    Eg:
        img_file_type = ['.jpg', '.png', '.jpeg']
    """

    def from_recursive_dir(img_path):
        print(f'Search for img in dir...')
        return [p for p in tqdm(img_path.rglob('*'))
                if p.suffix in img_file_type]

    def from_paths_in_txt(img_path):
        return f_readlines(img_path)

    if img_path.is_dir():
        img_list = from_recursive_dir(img_path)

    elif img_path.is_file() and img_path.suffix == '.txt':
        img_list = from_paths_in_txt(img_path)

    else:
        raise ValueError(
            'only list of Path and directory of imgs is supported')

    return img_list


def read_color_imgs(img_paths: pd.Series, n=int, verbose=True):
    img_to_read = img_paths.sample(n)
    for p in img_to_read:
        img = imageio.imread(p)
        if img.ndim == 3:
            yield img
        else:
            if verbose:
                print('pass grey scale img')
            yield next(read_color_imgs(img_paths, 1))


# --------- threading copy -------
class FileCopy(threading.Thread):
    def __init__(self, queue_obj, files: List[Path], dirs: List[Path]):
        threading.Thread.__init__(self)
        self._queue = queue_obj
        self.files = [str(f) for f in files]
        self.dirs = [str(d) for d in dirs]
        
        for f in files:
            if not f.is_file():
                raise ValueError(f"{f} does not Exist")
        for d in dirs:
            if not d.is_dir():
                raise ValueError(f"{d} is not a directory")

    def run(self):
        # This puts one object into the queue for each file,
        # plus a None to indicate completion
        try:
            for f in tqdm(self.files):
                try:
                    for d in self.dirs:
                        shutil.copy(f, d)
                except IOError as e:
                    self._queue.put(e)
                else:
                    self._queue.put(f)
        finally:
            self._queue.put(None)  # signal completion


def thread_copyfile(files: List[Path], dir_path: Path, verbose=False):
    '''Eg:
        files = (tpath/'train'/'images').ls()
        dir_path = train_setup_dir/'images'

        thread_copyfile(files, dir_path)
    '''

    print('init filecopy queue...')
    q_ = queue.Queue()
    copythread = FileCopy(q_, files, [dir_path])
    copythread.start()
    
    print('start copy')
    while True:
        x = q_.get()
        if x is None:
            break
        if verbose:
            print(x)
    copythread.join()