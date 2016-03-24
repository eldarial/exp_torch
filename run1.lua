
npy4th = require 'npy4th'

-- load file and show size
file_load = '../dataset_stacks_with_cluster_th_1.0_res_112_/' .. 'images_depth_stack_001_.npy'
print(file_load)
depth_data = npy4th.loadnpy(file_load)
print(depth_data:size())

