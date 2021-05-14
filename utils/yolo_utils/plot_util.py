import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_grid(images, max_rows=3, max_cols=3, figsize=(20,20), show_axis='off'):
    '''
    images : np.ndarray (N, h, w, chnl)
    N >= max_rows * max_cols 
    '''
    fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=figsize)

#     for r in range(max_rows):
#         for c in range(max_cols):
#             axes[r, c].axis('off')
    
    for idx, image in enumerate(images):
        row = idx // max_cols
        col = idx % max_cols
        axes[row, col].axis(show_axis)
        axes[row, col].imshow(image, cmap="gray", aspect="auto")
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.show()

def show_batch(imgs, figsize=(20,20), axis="off", titles=[]):
    '''Example:
    show_batch(augmented_imgs,
          titles=[img.shape for img in augmented_imgs])
    '''
    plt.figure(figsize=figsize)
    img_l = len(imgs)
    sqrt = np.sqrt(img_l)
    row = np.ceil(sqrt).astype('int')
    col = np.floor(sqrt).astype('int')

    show_title = True if len(titles) == img_l else False
    
    for i in range(img_l):
        ax = plt.subplot(row, col, i + 1)
        plt.imshow(imgs[i])
        if show_title:
            plt.title(titles[i])

    plt.axis(axis)


#draw
def draw_circle(image, center_coordinates, radius=5, color=(0, 0, 255), thickness=3, copy=False):
    to_draw = image.copy() if copy else image
    
    center_coordinates = tuple(map(int, center_coordinates)) 
    cv2.circle(to_draw, center_coordinates, radius, color, thickness)
    
    if copy:
        return to_draw
    
def draw_line(image, xyxy, color=(0,0,0), thickness=2, copy=False):
    to_draw = image.copy() if copy else image
    
    x1, y1, x2, y2 = list(map(int, xyxy))
    cv2.line(to_draw, (x1, y1), (x2, y2), color, thickness=thickness)
    
    if copy:
        return to_draw
    