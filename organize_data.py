import glob
import numpy as np
import os
import sys

def main():
	old_folder = '../old_dataset_stacks/'
	target_folder = '../dataset_stacks/'
	list_files = list(glob.glob(str(old_folder + "*.npy")))
	big_list = sorted(list_files)
	for npy_path in big_list:
		prefix = npy_path.split('/')[-1]
		print(npy_path)
		if prefix.startswith('images_color'):
			im = np.load(npy_path)
			vv = np.transpose(im, (0, 3, 1, 2))
			new_path = target_folder + npy_path
			np.save(new_path, vv)
		else:
			im = np.load(npy_path)
			vv = np.transpose(im, (0, 3, 1, 2))
			new_path = target_folder + npy_path
			np.save(new_path, vv)

if __name__ == '__main__':
	main()