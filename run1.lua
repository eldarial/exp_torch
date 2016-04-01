require 'nn'
require 'optim'
require 'torch'
require 'string'
npy4th = require 'npy4th'

-- adjust size
batch_size = 10

-- function to load data
function load_specific_data(index_file)
	local index_string = string.format("%03d", index_file) 
	local file_load = '../dataset_stacks/' .. 'images_depth_stack_' .. index_string .. '_.npy'
	print(file_load)
	local depth_data = npy4th.loadnpy(file_load)
	--local mini_batch = {}
	local mini_batch = torch.Tensor(batch_size,1,112,112)
	--local data_batch
	for t = 1,batch_size,1 do
		local sample = depth_data[t]
		mini_batch[t] = depth_data[t]
	end
	print(depth_data:size())
	return mini_batch
end


train_batch = load_specific_data(1)
print(train_batch:size())

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

model:add(nn.SpatialFullConvolution(1024, 8, 14, 14, 14, 14))

conv1 = model:forward(train_batch:double())

-- test spatial convolutions
print(conv1:size())

