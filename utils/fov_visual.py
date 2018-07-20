import matplotlib.patches as patches
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math as m

from itertools import product, combinations
from scipy.interpolate import RegularGridInterpolator as interp2d
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from try_360dgrid import xy2angle, to_3dsphere, pruned_inf, over_pi
from scipy.interpolate import RegularGridInterpolator as interp2d



def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def func_warp_image(Px, Py, frame, warp_method):

    sphW = Px.shape[1]
    sphH = Px.shape[0]

    minX = max(0, int(np.floor(Px.min())))
    minY = max(0, int(np.floor(Py.min())))

    maxX = min(int(sphW), int(np.ceil(Px.max())))
    maxY = min(int(sphH), int(np.ceil(Py.max())))

    im = frame[minY:maxY, minX:maxX, :]
    Px -= minX
    Py -= minY
    warped_images = []

    y_grid = np.arange(im.shape[0])
    x_grid = np.arange(im.shape[1])
    samples = np.vstack([Py.ravel(), Px.ravel()]).transpose()

    for c in xrange(frame.shape[2]):
        full_image = interp2d((y_grid, x_grid), im[:,:,c],
                               bounds_error=False,
                               method=warp_method,
                               fill_value=None)
        warped_image = full_image(samples).reshape(Px.shape)
        warped_images.append(warped_image)
    warped_image = np.stack(warped_images, axis=2)
    return warped_image


# get warped mask and transformed x y coordinates
def fov_module(in_shape, grid, hang, mid_pos):
    hang = hang*np.pi/180
    #vang = vang*np.pi/180
    mid_pos[0] = mid_pos[0]*np.pi/180
    mid_pos[1] = mid_pos[1]*np.pi/180

    #=== rotation 
    _x,_y,_z = to_3dsphere(grid[0]-mid_pos[0],grid[1],1)
    _m3dmat = np.concatenate((np.concatenate((np.expand_dims(_x,axis=2),np.expand_dims(_y,axis=2)),axis=2),np.expand_dims(_z,axis=2)),axis=2)
    m3dmat = np.dot(_m3dmat,rotation_matrix([0,0,1],mid_pos[1]))
    #m3dmat=_m3dmat

    x=m3dmat[:,:,0]
    y=m3dmat[:,:,1]
    z=m3dmat[:,:,2]

    #=== normal vec of x to distinguish direction
    nx,ny,nz = to_3dsphere(mid_pos[0],mid_pos[1],1)
    norm_mat = np.multiply(x,np.abs(nx))#+np.multiply(y,ny)#+np.multiply(z,nz)
    norm_mask = norm_mat>0
    #print("norm x = {0}",nx)

    rzx = np.linalg.norm(np.stack((x, z), axis=0), axis=0)

    imW=in_shape[1]
    imH=in_shape[0]

    #=== calculate image to sphere transformation

    f = (imW/2) / np.tan((hang)/2)
    Px = f*(z/x)
    d = np.sqrt(np.multiply(Px,Px) + f*f)
    Py = np.multiply(d, y/rzx)
    Px = Px + imW/2 
    Py = -Py + imH/2 

    validMap = (Px<1) | (Px>imW) | (Py<1) | (Py>imH)
    validMap = (~validMap)

    fov_mask=validMap&norm_mask
    fov_mask = np.tile(fov_mask[..., None],[1,1,in_shape[2]])


    return fov_mask, Px, Py 
        
def box_proh(im_in, targ_fov, targ_th, targ_ph, targ_size = [480,960],warp_method='linear'):
    scale_c=1
    out_w = targ_size[1]
    out_h = targ_size[0]

    XX, YY = np.meshgrid(range(out_w),range(out_h)) # for output grid
    XX=XX+0.5
    YY=YY+0.5
    theta, phi = xy2angle(XX, YY, out_w, out_h)

    _x,_y,_z = to_3dsphere(theta,phi,1)
    

    fov_mask, px,py = fov_module(im_in.shape, [theta,phi], targ_fov, [targ_th, targ_ph])
 
   ######### check less than 0 or over 360
    px_valid=np.zeros(px.shape)
    px_valid[fov_mask[:,:,0]] = px[fov_mask[:,:,0]]
    max_px_coord = np.where(px_valid==np.max(px_valid[px_valid<=im_in.shape[1]]))
    min_px_coord = np.where(px_valid==np.min(px_valid[px_valid>=0]))

    out_im = func_warp_image(px, py, im_in, warp_method)
    
    #=== mask outside the fov
    out_im[~fov_mask]=0.

    # return output image and x coord only
    return out_im, fov_mask#, crop_fov_mask

def bbox_proh(im_in, targ_fov, targ_th, targ_ph, warp_method='linear'):
    scale_c=1
    out_w = 1920
    out_h = 960

    XX, YY = np.meshgrid(range(out_w),range(out_h)) # for output grid

    theta, phi = xy2angle(XX, YY, out_w, out_h)

    _x,_y,_z = to_3dsphere(theta,phi,1)

    #fov_mask = ov_module([_x,_y,_z], 90., 65., [10., 10.])

    fov_mask, px,py = fov_module(im_in.shape, [theta,phi], targ_fov, [targ_th, targ_ph])
    inside_fov_mask, px,py = fov_module(im_in.shape, [theta,phi], targ_fov-3, [targ_th, targ_ph])

    #fov_mask, px,py = inver_fov_module(im_in.shape, [theta,phi], targ_fov, [targ_th, targ_ph])
    
    ######### check less than 0 or over 360
    px_valid=np.zeros(px.shape)
    px_valid[fov_mask[:,:,0]] = px[fov_mask[:,:,0]]
    max_px_coord = np.where(px_valid==np.max(px_valid[px_valid<im_in.shape[1]]))
    min_px_coord = np.where(px_valid==np.min(px_valid[px_valid>0]))
    ########

    out_im = func_warp_image(px, py, im_in, warp_method)

    #=== mask outside the fov
    out_im[np.logical_not(fov_mask)]=0.
    out_im[inside_fov_mask]=0.
    
    #out_im[np.logical_and(inside_fov_mask,(np.logical_not(fov_mask)))]=0.
    
    return out_im, fov_mask, inside_fov_mask


# get solid mask of box area
def get_box_mask(im_in_shape,box):
    mask = np.zeros(im_in_shape)
    mask[box[1]:box[3],box[0]:box[2]]=1
    return np.expand_dims(mask[:,:,0],axis=2)


# get tight box, px_coord for determine left right border
def get_tite_box(box_in, px_coord=[100,0]):

    BORDER = 0
    mid_xp = 0
    if px_coord[0]<px_coord[1]:
        BORDER = 1
        mid_xp = (px_coord[1]+px_coord[0])/2

    out_boxes = []

    ycord,xcord,_ = np.where(box_in==1)
    if BORDER:
        idx_l = xcord<mid_xp
        idx_r = xcord>mid_xp
        for idx_arr in [idx_l,idx_r]:
            if np.sum(idx_arr)>0:
                tmp_y = ycord[idx_arr]
                tmp_x = xcord[idx_arr]
                out_boxes.append([np.min(tmp_x),np.min(tmp_y),np.max(tmp_x),np.max(tmp_y)])
    else:
        out_boxes.append([np.min(xcord),np.min(ycord),np.max(xcord),np.max(ycord)])

    return out_boxes
    
# box rectangle shape visualize
def draw_dum_box(out_im,box,color):
    mask = np.zeros(out_im.shape)
    mask = mask[:,:,0]==1
    mask[(box[1]):(box[3]),(box[0]):(box[2])]=True
    mask[(box[1]+5):(box[3]-5),(box[0]+5):(box[2]-5)]=False
    out_im[mask]=color

    return out_im

def draw_cube_fov_box(fov_img, ori_img):

    out_img = ori_img
    for ang in [0.,90.,180.,270.]:
        out_im, fov_mask, fov_in_mask = bbox_proh(fov_img,120., ang, 0.)
        bbox_mask = np.logical_and(fov_mask,np.logical_not(fov_in_mask))
        out_img[bbox_mask[:,:,0]]=[0,1,0]
    for ang in [0.,90.,180.,270.]:
        out_im, fov_mask, fov_in_mask = bbox_proh(fov_img,90., ang, 0.)
        bbox_mask = np.logical_and(fov_mask,np.logical_not(fov_in_mask))
        out_img[bbox_mask[:,:,0]]=[1,0,0]
    return out_img

def main():
    #ori_img = cv2.imread('/Users/Jim/Desktop/ski_test.jpg')/255.
    im_file_name = '/Users/Jim/Desktop/Sample_streetview_pano.jpg'
    #im_file_name = '/Users/Jim/Desktop/2007_000170.jpg'
    gt_box = [180,70,410,300]
    #im_file_name = '/Users/Jim/Desktop/horse_rider.jpg'
    #gt_box = [80,5,270,530]  
    _ori_img = cv2.imread(im_file_name)/255.

    _ori_img = cv2.resize(_ori_img, (1920,960))
    fov_img = np.ones((500,500,3))
    #box_mask = get_box_mask(_ori_img.shape,gt_box)

    ori_img = _ori_img

    cv2.imwrite(im_file_name.split('.')[0]+'_warp.jpg',cv2.resize(outim_with_box*255, (1000,500)))

def draw_cube_faces():

    img_source = '/home/raul/Desktop/my360projection'
    img_name = 'cube_to_equi.jpg'
    ang_cand = [[180.,0.],[0.,-90],[0.,0.],[270.,0.],[90.,0.],[0.,90.]]
    out_based_img = np.zeros((960,1920,3))
    out_imgs = []
    fov_masks = []
    crop_fov_masks = []
    
    for idx in range(6):
        cimg_name = os.path.join(img_source,img_name)
        fov_img = cv2.imread(cimg_name)/255.
        out_img, fov_mask, crop_fov_mask = box_proh(fov_img, 90.,ang_cand[idx][0],ang_cand[idx][1])
        out_imgs.append(out_img)
        fov_masks.append(fov_mask)
        crop_fov_masks.append(crop_fov_mask)

    for step in [5,0,2,3,4]:
        out_based_img[fov_masks[step]] = out_imgs[step][fov_masks[step]]
    for step in [5,0,2,3,4]:
        out_based_img[crop_fov_masks[step]] = out_imgs[step][crop_fov_masks[step]]
   
    cv2.imwrite('/Users/Jim/Desktop/'+img_name.split('.')[0]+'_cubevis.jpg',cv2.resize(out_based_img*255, (1000,500)))


if __name__ == "__main__":
    #main()
    draw_cube_faces()

