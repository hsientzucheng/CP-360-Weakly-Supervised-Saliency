import os
import numpy as np
import argparse
from PIL import Image
from utils.equi_to_cube import Equi2Cube

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None, help='Test video path.')
    parser.add_argument('--out', type=str, default='./Saliency_Visualization', help='Test video path.')
    args, unparsed = parser.parse_known_args()

    def get_vid_list(in_dir):
        out_list = []
        for item in os.listdir(in_dir):
            out_list.append(item)
        return out_list

    out_list = get_vid_list(args.dir)

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    output_path = os.path.join(args.out,args.dir.split('/')[-1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    TEST_ONLY = True
    if TEST_ONLY:
        fff = open('./test_25.txt', "r")
        ddd_list = fff.readlines()
        data_list = [x.split('\n')[0] for x in ddd_list]
    vid_num=0
    for item in data_list:
        vid_num+=1
        print(item+' '+str(vid_num)+'/'+str(len(data_list)))
        if 'CLSTM' in args.dir:
            vid_path = os.path.join(args.dir,item,'overlay')
        elif 'dataset' in args.dir:
            vid_path = os.path.join(args.dir,item+'.mp4','overlay')
        else:
            vid_path = os.path.join(args.dir,item)
        frame_list = os.listdir(vid_path)
        local_out_path = os.path.join(output_path, item)
        if not os.path.exists(local_out_path):
            os.makedirs(local_out_path)
        for num, frame in enumerate(frame_list):
            try:
                cnt = int(frame.split('.jpg')[0])
            except:
                continue
            frame_path = os.path.join(vid_path, frame)
            equi_img = Image.open(frame_path)
            equi_img = equi_img.convert('RGB')
            equi_img = equi_img.resize((1920,960),resample=Image.LANCZOS)
            if num == 0:
                e2c = Equi2Cube(480, np.array(equi_img))
            output_cubeset = e2c.to_cube(np.array(equi_img))

            out_td_equi = np.zeros((np.array(equi_img).shape[0],np.array(equi_img).shape[1]+480,np.array(equi_img).shape[2]),dtype=np.uint)
            out_td_equi[:480,1920:,:]=output_cubeset[5]
            out_td_equi[480:,1920:,:]=output_cubeset[1]
            out_td_equi[:,:1920,:]=np.array(equi_img)
            '''equi_img'''
            out_td_equi = out_td_equi.astype(np.int)
            out_td_equi_img = Image.fromarray(np.uint8(out_td_equi))
            out_td_equi_img = out_td_equi_img.resize((1440, 480), resample=Image.LANCZOS)

            out_td_equi_img.save(os.path.join(local_out_path,'{0:06}.jpg'.format(cnt)))
            #equi_img.save(os.path.join(local_out_path,'{0:06}.jpg'.format(cnt)))
            print(os.path.join(local_out_path,'{0:06}.jpg'.format(cnt)))


if __name__ == "__main__":
    main()
