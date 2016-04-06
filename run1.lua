require 'nn'
require 'optim'
require 'torch'
require 'string'
npy4th = require 'npy4th'

-- adjust size
batch_size = 10
num_classes = 9

-- function to load data
function load_specific_data(index_file)
	local index_string = string.format("%03d", index_file) 
	local file_load = '../dataset_stacks/' .. 'images_depth_stack_' .. index_string .. '_.npy'
	print(file_load)
	local depth_data = npy4th.loadnpy(file_load)
	file_load = '../dataset_stacks/' .. 'images_color_stack_' .. index_string .. '_.npy'
	local color_data = npy4th.loadnpy(file_load)
	local x_batch = torch.Tensor(batch_size, 1, 112, 112)
	local y_batch = torch.Tensor(batch_size, num_classes, 112, 112)
	for t = 1,batch_size,1 do
		local sample = depth_data[t]
		x_batch[t] = depth_data[t]
		y_batch[t] = color_data[t]
	end
	print(depth_data:size())
	return x_batch, y_batch
end


train_data, train_label = load_specific_data(1)
print("** information of batch **")
print(train_data:size())
print(train_label:size())

-- create network
model = nn.Sequential()
model:add(nn.SpatialConvolution(1, 32, 9, 9))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(1, 1, 2, 2))

model:add(nn.SpatialConvolution(32, 64, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(1, 1, 2, 2))

model:add(nn.SpatialConvolution(64, 64, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(1, 1, 2, 2))

model:add(nn.SpatialConvolution(64, 1024, 3, 3))
model:add(nn.ReLU())

model:add(nn.SpatialFullConvolution(1024, num_classes, 14, 14, 14, 14))
model:add(nn.Transpose({2,4}))
model:add(nn.Reshape(batch_size*112*112, num_classes))
fwd1 = model:forward(train_data:double())
model:add(nn.LogSoftMax())
fwd2 = model:forward(train_data:double())


-- test spatial convolutions
print("outputs size")
print(fwd1[{1,{1,3},1,1}])
print(fwd2[{1,{1,3},1,1}])