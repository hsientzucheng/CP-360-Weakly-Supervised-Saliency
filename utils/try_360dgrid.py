import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2
from itertools import product, combinations
from scipy.interpolate import RegularGridInterpolator as interp2d
import math as m
from pylab import *
'''
rot_ax90 = {'x': np.array([[1,0,0],[0,0,-1],[0,1,0]]),
            'y': np.array([[0,0,1],[0,1,0],[-1,0,0]]),
            'z': np.array([[0,-1,0],[1,0,0],[0,0,1]]),
            '-x': np.array([[1,0,0],[0,0,1],[0,-1,0]]),
            '-y': np.array([[0,0,-1],[0,1,0],[1,0,0]]),
            '-z': np.array([[0,1,0],[-1,0,0],[0,0,1]])}
'''

rad45 = np.pi/4
rad135 = np.pi*3/4
#PHI_BOUND=m.atan(1/m.sqrt(2))

def xy2angle(XX,YY,im_w,im_h):
    _XX = 2*XX/float(im_w)-1
    _YY = 1-2*YY/float(im_h)
    theta = _XX*np.pi
    phi = _YY*np.pi/2
    return theta, phi

def to_3dsphere(theta,phi,R):
    x = R*np.cos(phi)*np.cos(theta)
    y = R*np.sin(phi)
    z = R*np.cos(phi)*np.sin(theta)
    return x,y,z

def face_conditions(theta, phi, face):
    if face=='T':
        return (phi>0)&(np.sin(theta)/np.tan(phi)<1)&(np.sin(theta)/np.tan(phi)>-1)&(np.cos(theta)/np.tan(phi)<1)&(np.cos(theta)/np.tan(phi)>-1)
    elif face=='D':
        return (phi<0)&(np.sin(theta)/np.tan(phi)<1)&(np.sin(theta)/np.tan(phi)>-1)&(np.cos(theta)/np.tan(phi)<1)&(np.cos(theta)/np.tan(phi)>-1)
    elif face=='F':
        return (theta>-rad45)&(theta<rad45)&(np.tan(phi)/np.cos(theta)>-1)&(np.tan(phi)/np.cos(theta)<1)
    elif face=='L':
        return (theta>rad45)&(theta<rad135)&(np.tan(phi)/np.sin(theta)>-1)&(np.tan(phi)/np.sin(theta)<1)
    elif face=='B':
        return ((theta>rad135)|(theta<-rad135))&(np.tan(phi)/np.cos(theta)>-1)&(np.tan(phi)/np.cos(theta)<1)
    elif face=='R':
        return (theta>-rad135)&(theta<-rad45)&(-np.tan(phi)/np.sin(theta)>-1)&(-np.tan(phi)/np.sin(theta)<1)

def pruned_inf(angle):
    float_err = 10e-9
    angle[angle==0.0]=float_err
    angle[angle==np.pi]=np.pi-float_err
    angle[angle==-np.pi]=-np.pi+float_err
    angle[angle==np.pi/2]=np.pi/2-float_err/2
    angle[angle==-np.pi/2]=-np.pi/2+float_err/2
    return angle 

def over_pi(angle):
    while(angle>np.pi):
        angle-=2*np.pi
    while(angle<-np.pi):
        angle+=2*np.pi
    return angle

def to_3dcube(theta,phi,R):

    #print PHI_BOUND
    x,y,z = to_3dsphere(theta,phi,R)

    output_cube = np.array([x,y,z])
    output_cube = np.transpose(output_cube,(1,2,0))

    front_trans = np.array([np.ones(phi.shape),np.tan(phi)/np.cos(theta),np.tan(theta)]) 
    front_trans = np.transpose(front_trans,(1,2,0))
    back_trans = np.array([-np.ones(phi.shape), -np.tan(phi)/np.cos(theta), -np.tan(theta)])
    back_trans = np.transpose(back_trans,(1,2,0))
    left_trans = np.array([1/np.tan(theta), np.tan(phi)/np.sin(theta), np.ones(phi.shape)])
    left_trans = np.transpose(left_trans,(1,2,0))
    right_trans = np.array([-1/np.tan(theta), -np.tan(phi)/np.sin(theta), -np.ones(phi.shape)])
    right_trans = np.transpose(right_trans,(1,2,0))
    top_trans = np.array([np.cos(theta)/np.tan(phi),np.ones(phi.shape), np.sin(theta)/np.tan(phi)]) 
    top_trans = np.transpose(top_trans,(1,2,0))
    down_trans = np.array([-np.cos(theta)/np.tan(phi), -np.ones(phi.shape), -np.sin(theta)/np.tan(phi)])
    down_trans = np.transpose(down_trans,(1,2,0))

    print("coordinate transfer")
    output_cube[face_conditions(theta, phi,'F'),:] = front_trans[face_conditions(theta, phi,'F'),:]
    output_cube[face_conditions(theta, phi,'L'),:] = left_trans[face_conditions(theta, phi,'L'),:]
    output_cube[face_conditions(theta, phi,'B'),:] = back_trans[face_conditions(theta, phi,'B'),:]
    output_cube[face_conditions(theta, phi,'R'),:] = right_trans[face_conditions(theta, phi,'R'),:]
    output_cube[face_conditions(theta, phi,'T'),:] = top_trans[face_conditions(theta, phi,'T'),:]
    output_cube[face_conditions(theta, phi,'D'),:] = down_trans[face_conditions(theta, phi,'D'),:]

    return output_cube

def set_region_mask(t, p, targ_t, targ_p, l_width, angle):
    print("target: {0},{1} angle={2}".format(targ_t,targ_p,angle*180/np.pi))
    r_most = over_pi((targ_t+angle+l_width))
    l_most = over_pi((targ_t-angle-l_width))
    r_in = over_pi((targ_t+angle-l_width))
    l_in = over_pi((targ_t-angle+l_width))

    if r_most>l_most:
        o_region_mask=(t<r_most)&(t>l_most)&(p<(targ_p+angle+l_width))&(p>(targ_p-angle-l_width))
    else:
        o_region_mask=((t<r_most)|(t>l_most))&(p<(targ_p+angle+l_width))&(p>(targ_p-angle-l_width))
    
    if r_in>l_in:
        i_region_mask=(t<r_in)&(t>l_in)&(p<(targ_p+angle-l_width))&(p>(targ_p-angle+l_width))
    else:
        i_region_mask=((t<r_in)|(t>l_in))&(p<(targ_p+angle-l_width))&(p>(targ_p-angle+l_width))

    region_mask = o_region_mask&~i_region_mask
    return region_mask

def solid_region_mask(t, p, targ_t, targ_p, l_width, angle):
    print("target: {0},{1} angle={2}".format(targ_t,targ_p,angle*180/np.pi))
    r_most = over_pi((targ_t+angle+l_width))
    l_most = over_pi((targ_t-angle-l_width))
    #r_in = over_pi((targ_t+angle-l_width)) 
    #l_in = over_pi((targ_t-angle+l_width))

    if r_most>l_most:
        o_region_mask=(t<r_most)&(t>l_most)&(p<(targ_p+angle+l_width))&(p>(targ_p-angle-l_width))
    else:
        o_region_mask=((t<r_most)|(t>l_most))&(p<(targ_p+angle+l_width))&(p>(targ_p-angle-l_width))   

    region_mask = o_region_mask
    return region_mask



def myplot(in_name,newsize):

    VIS_MASK=0
    WRITE_FIG=1
    BOX=1
    scale = 0.5

    R=1
    '''
    targ_th=30
    targ_ph=-5
    view_th=30
    view_ph=-30
    vang_x=50
    vang_y=50
    '''	
    targ_th=-180
    targ_ph=-18
    view_th= 160
    view_ph=55
    vang_x=70

    raw_im = cv2.imread(in_name)

    _im_h, _im_w = raw_im.shape[:2]
    #_resize_im = cv2.resize(raw_im,(int(_im_w*scale), int(_im_h*scale))) ##
    _resize_im = cv2.resize(raw_im,newsize) ##
    #resize_im = cv2.normalize(resize_im, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_resize_im = cv2.normalize(_resize_im, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    bb,gg,rr = cv2.split(norm_resize_im)       # get b,g,r
    resize_im = cv2.merge([rr,gg,bb]) 

    im_h, im_w = resize_im.shape[:2]

    XX, YY = np.meshgrid(range(im_w),range(im_h))

    theta, phi = xy2angle(XX,YY,resize_im.shape[1], resize_im.shape[0]) 

    theta = pruned_inf(theta)
    phi = pruned_inf(phi)

    print("compute coordinate")
    x,y,z = to_3dsphere(theta,phi,R)
    cube_mesh = to_3dcube(theta,phi,1)

    #region_mask = set_region_mask(theta,phi,targ_th/180.0*np.pi,targ_ph/180.0*np.pi,0.05,vang_x/180.0*np.pi)
    region_mask = solid_region_mask(theta,phi,targ_th/180.0*np.pi,targ_ph/180.0*np.pi,0.05,vang_x/180.0*np.pi)
    if VIS_MASK:
        vis_mask = norm_resize_im.copy()*255.
        vis_mask[region_mask,:]=(0,0,255)
        cv2.imwrite(in_name.split('.')[0]+'_mask.jpg',vis_mask)

    fig_sp = plt.figure()

    print("plot sphere")
    ax_sp = fig_sp.add_subplot(1, 2, 1, projection='3d')
    #ax_sp = fig_sp.gca(projection='3d')
    ax_sp.set_aspect("equal")
    ax_sp.set_xlabel('X axis')
    ax_sp.set_ylabel('Y axis')
    ax_sp.set_zlabel('Z axis')
    ax_sp.view_init(view_th,view_ph)
    ax_sp.plot_wireframe(x,y,z,color='b',rstride=10, cstride=10)
    #ax_sp.plot_surface(x,y,z, facecolors=resize_im, shade=False)    
    if(BOX):
        ax_sp.hold(True)
        x[region_mask==False]=np.nan
        y[region_mask==False]=np.nan
        z[region_mask==False]=np.nan
        ax_sp.plot_wireframe(x,y,z,color='r',rstride=6, cstride=6)
        #ax_sp.plot_surface(x,y,z,antialiased=False, linewidth=0, shade=False)

    print("plot cube")
    ax_sp = fig_sp.add_subplot(1, 2, 2, projection='3d')
    ax_sp.set_aspect("equal")
    ax_sp.set_xlabel('X axis')
    ax_sp.set_ylabel('Y axis')
    ax_sp.set_zlabel('Z axis')
    ax_sp.view_init(view_th,view_ph)

    ax_sp.plot_wireframe(cube_mesh[:,:,0], cube_mesh[:,:,1], cube_mesh[:,:,2], rstride=10, cstride=10)
    #ax_sp.plot_surface(cube_mesh[:,:,0], cube_mesh[:,:,1], cube_mesh[:,:,2], facecolors=resize_im, shade=False)
    if(BOX):
        ax_sp.hold(True)
        cube_mesh[region_mask==False,:]=np.nan
        ax_sp.plot_wireframe(cube_mesh[:,:,0], cube_mesh[:,:,1], cube_mesh[:,:,2],color='r',rstride=6, cstride=6)
        #ax_sp.plot_surface(cube_mesh[:,:,0], cube_mesh[:,:,1], cube_mesh[:,:,2], color='r', antialiased=False, linewidth=0, shade=False)
    
    if WRITE_FIG:
        plt.savefig(in_name.split('.')[0]+'_cs'+str(view_th)+'_'+str(view_ph)+'.jpg')
    #plt.show()


def cubeplot(in_dir,view_th,view_ph):

    WRITE_FIG=1
    scale_c=2

    raw_im={}
    for idx in range(6):
        raw_tmp = cv2.imread(in_dir+'{0}.jpg'.format(idx))
        tmp_im = cv2.normalize(cv2.resize(raw_tmp, (int(raw_tmp.shape[0]*scale_c),int(raw_tmp.shape[0]*scale_c))), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        bb,gg,rr = cv2.split(tmp_im)       # get b,g,r
        raw_im[idx] = cv2.merge([rr,gg,bb])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #r = [-1,1]
    r = np.arange(-1, 1, 2.0/raw_im[idx].shape[0])
    X, Y = np.meshgrid(r, r)
    ax.plot_surface(-X,-Y,1, facecolors=raw_im[4], shade=False)
    ax.plot_surface(X,-Y,-1, facecolors=raw_im[3], shade=False)
    ax.plot_surface(-Y,-1,X, facecolors=raw_im[1], shade=False)
    ax.plot_surface(Y,1,X, facecolors=raw_im[5], shade=False)
    ax.plot_surface(1,-Y,X, facecolors=raw_im[2], shade=False)
    ax.plot_surface(-1,-Y,-X, facecolors=raw_im[0], shade=False) 
    ax.set_aspect("equal") 
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(view_th,view_ph)

    if WRITE_FIG:
        plt.savefig(in_dir+'cube_vis'+str(view_th)+'_'+str(view_ph)+'.jpg')

def main():
    source_path = '/Users/Jim/Desktop/360MEETING/2017_03_02_pad_exp/select_f_0301/'
    main_size = (2000,1000)
    sub_size = (1000,500)    

    view_th= 160
    view_ph=55

    myplot('/Users/Jim/Desktop/ski_test.jpg',sub_size)
    plt.show()
    #cubeplot('/Users/Jim/Desktop/test_frames_ll4/cube_',view_th,view_ph)
    #items = ['188','193','208','224']
    items = ['224','193']
    # plot cube map directly 
    #cubeplot(source_path+'ski_test_cube/cube_',view_th,view_ph)
    
    for item in items:
        cubeplot(source_path+'zero_pad/filter_'+item+'_',view_th,view_ph)
        cubeplot(source_path+'cube_pad/filter_'+item+'_',view_th,view_ph)
    # plot cube map directly ^ ^ ^ ^
    #myplot('/Users/Jim/Desktop/ski_test.jpg',main_size)
    for item in items:
        myplot(source_path+'filter_'+item+'_1.jpg',sub_size)
        myplot(source_path+'filter_'+item+'_cp.jpg',sub_size)
        myplot(source_path+'filter_'+item+'_zp.jpg',sub_size)

if __name__ == "__main__":
    main()

