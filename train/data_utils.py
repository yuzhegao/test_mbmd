import numpy as np
from PIL import Image, ImageOps, ImageStat, ImageDraw

def crop_search_region(img, last_gt, win_size, area_scale=4, mean_rgb=128):
    """
    crop the search region, which is four times the size of the target,and centering in gt's center

     img: PIL.Image object
     gt: [xmin, ymin, xmax, ymax]

    """

    bnd_xmin, bnd_ymin, bnd_xmax, bnd_ymax = last_gt
    bnd_w = bnd_xmax - bnd_xmin
    bnd_h = bnd_ymax - bnd_ymin
    cy, cx = (bnd_ymin + bnd_ymax)/2, (bnd_xmin+bnd_xmax)/2
    origin_win_size_h, origin_win_size_w = bnd_h * area_scale, bnd_w * area_scale

    im_size = img.size[1::-1] ##[H,W]
    min_x = np.round(cx - origin_win_size_w / 2).astype(np.int32)
    max_x = np.round(cx + origin_win_size_w / 2).astype(np.int32)
    min_y = np.round(cy - origin_win_size_h / 2).astype(np.int32)
    max_y = np.round(cy + origin_win_size_h / 2).astype(np.int32)

    win_loc = np.array([min_x,min_y])

    unscaled_w, unscaled_h = [max_x - min_x + 1, max_y - min_y + 1] ## before scaled to 300*300
    min_x_win, min_y_win, max_x_win, max_y_win = (0, 0, unscaled_w, unscaled_h)
    ## in search region coordinate

    min_x_im, min_y_im, max_x_im, max_y_im = (min_x, min_y, max_x+1, max_y+1)
    ## in origin img coordinate   (useless)

    img = img.crop([min_x_im, min_y_im, max_x_im, max_y_im]) ## crop the search region
    ## from the code below: if the min/max out of origin img bound, then just padding
    img_array = np.array(img)

    if min_x < 0:
        min_x_win = 0 - min_x
    if min_y < 0:
        min_y_win = 0 - min_y
    if max_x+1 > im_size[1]:
        max_x_win = unscaled_w - (max_x + 1 - im_size[1])
    if max_y+1 > im_size[0]:
        max_y_win = unscaled_h - (max_y + 1 - im_size[0])
    ## after padding

    unscaled_win = np.ones([unscaled_h, unscaled_w, 3], dtype=np.uint8) * np.uint8(mean_rgb)
    unscaled_win[min_y_win:max_y_win, min_x_win:max_x_win] = img_array[min_y_win:max_y_win, min_x_win:max_x_win]
    ## here padding with 128(mean value)

    unscaled_win = Image.fromarray(unscaled_win)
    height_scale, width_scale = np.float32(unscaled_h)/win_size, np.float32(unscaled_w)/win_size
    win_img = unscaled_win.resize([win_size, win_size], resample=Image.BILINEAR)
    ## now resize and get "resize_scale_rate"

    return win_img, win_loc, [width_scale,height_scale]
    # return win, np.array([gt_x_min, gt_y_min, gt_x_max, gt_y_max]), diag, np.array(win_loc)

def draw_box(img,box,img_path='img.jpg'):
    """
     img: PIL Image object
     box: [x1 y1 x2 y2] or a list
    """
    x1, y1, x2, y2 = box
    draw = ImageDraw.Draw(img)
    draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=3, fill='red')
    img.save(img_path)

def draw_mulitbox(img,box_list,img_path='img.jpg'):
    """
     img: PIL Image object
     box: [x1 y1 x2 y2] or a list
    """
    for idx,box in enumerate(box_list):
        x1, y1, x2, y2 = box
        draw = ImageDraw.Draw(img)
        if idx == 0:
            color = 'red'
        elif idx ==1:
            color = 'green'
        else:
            color = 'blue'
        draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=3, fill=color)
    img.save(img_path)

def transform(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
     img:  numpy array (W,H,3)
    """
    mean,std = np.array(mean),np.array(std)
    img_array = (np.array(img)/255.0).astype(np.float32) ##(W,H,3)
    img_array = (img_array - mean)/std

    return img_array.astype(np.float32)

def iou_y1x1y2x2(box1,box2):
    a1,b1,a2,b2 = box1
    c1,d1,c2,d2 = box2

    area1 = (a2-a1)*(b2-b1)
    area2 = (c2-c1)*(d2-d1)
    xmin,ymin = max(b1,d1),max(a1,c1)
    xmax,ymax = min(b2,d2),min(a2,c2)
    intersect = (xmax-xmin)*(ymax-ymin)

    return intersect*1.0/(area1 + area2 - intersect)




