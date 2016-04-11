import glob
import numpy as np
import os
import sys

def get_all_colors():
    # background, re, rs, rc, head, lc, ls, le, b
    # background, 'rh', 'lra', 'ura', 'head', 'ula', 'lla', 'lh', 'b'
    #list_colors = [(64,64,64),(0,0,255),(0,191,255),(0,255,128),(64,255,0),(255,255,0),(255,64,0),(255,0,128),(191,0,255)]
    
    label_joints = ['background', 'head', 'torso', 'ula', 'lla','lh','ura','lra','rh']

    dict_colors_label = {}
    dict_colors_label['background'] = [(64,64,64), 0]
    dict_colors_label['rh'] = [(0,0,255), 1]
    dict_colors_label['lra'] = [(0,191,255), 2]
    dict_colors_label['ura'] = [(0,255,128), 3]
    dict_colors_label['head'] = [(64,255,0), 4]
    dict_colors_label['ula'] = [(255,255,0), 5]
    dict_colors_label['lla'] = [(255,64,0), 6]
    dict_colors_label['lh'] = [(255,0,128), 7]
    dict_colors_label['torso'] = [(191,0,255), 8]

    list_colors = []
    list_label_joint = []
    for joint in label_joints:
        list_colors.append(dict_colors_label[joint][0])
        list_label_joint.append(dict_colors_label[joint][1])
    return list_colors, list_label_joint

def transform_cross_gt(images_color, list_colors):
    num_joints = len(list_colors)
    (N,H,W,_) = images_color.shape
    images_gt = np.ones((N,H,W,num_joints), dtype=np.float32)
    offset = 5
    for i in range(1,len(list_colors)):
        # BGR
        color = list_colors[i]
        id0 = (images_color[:,:,:,0] <= color[0]+offset) & (images_color[:,:,:,0] >= color[0]-offset)
        id1 = (images_color[:,:,:,1] <= color[1]+offset) & (images_color[:,:,:,1] >= color[1]-offset)
        id2 = (images_color[:,:,:,2] <= color[2]+offset) & (images_color[:,:,:,2] >= color[2]-offset)
        ids = id0*id1*id2
        #images_gt[ids,i] = 1
        images_gt[np.invert(ids),i] = 0
        images_gt[ids,0] = 0
    print("joints sum")
    print(images_gt.shape)
    return images_gt


def main():
	old_folder = '../old_dataset_stacks/'
	target_folder = '../dataset_stacks/'
	list_files = list(glob.glob(str(old_folder + "*.npy")))
	big_list = sorted(list_files)
	for npy_path in big_list:
		prefix = npy_path.split('/')[-1]
		print(npy_path)
		if prefix.startswith('images_color'):
			y_data = np.load(npy_path)
			lc, _ = get_all_colors()
			y_data = transform_cross_gt(y_data, lc)
			vv = np.transpose(y_data, (0, 3, 1, 2))
			new_path = target_folder + prefix
			np.save(new_path, vv)
		else:
			im = np.load(npy_path)
			im_reshaped = np.transpose(im, (0, 3, 1, 2))
			new_path = target_folder + prefix
			np.save(new_path, im_reshaped)

if __name__ == '__main__':
	main()