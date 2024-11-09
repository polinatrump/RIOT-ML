import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from functools import reduce
# Dictionary to track the re-computation frequency of each intermediate tensor element
recomp_MAC = defaultdict(int)
common_MAC = defaultdict(int)

recomp_freq = defaultdict(int)
common_freq = defaultdict(int)

class FakeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.size = reduce(lambda x,y: x*y, shape)


# HWC format
# Define a base class for layers
class Layer:
    def __init__(self, name, output_channels, kernel_size, stride=1, padding=0, dialation=1):
        self.name = name
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dialation = dialation

        # self.memory_usage = None
        self.MAC_per_element = None
        self.output_tensor_shape = None
        self.output_tensor_compute_freq = None
        self.input_tensor_shape = None
        self.tile_buffer_size = None

        self._common_memory_usage = None
        self._memory_usage = None


    def forward_common(self, input_tensor):
        output_height = (input_tensor.shape[0]  + 2 * self.padding - self.dialation* (self.kernel_size - 1) - 1) // self.stride + 1
        output_width = (input_tensor.shape[1] + 2 * self.padding - self.dialation* (self.kernel_size - 1) - 1) // self.stride + 1
        self.output_tensor_shape = (output_height, output_width, self.output_channels)
        self.input_tensor_shape = input_tensor.shape
        self.output_tensor_compute_freq = np.ones((output_height, output_width, self.output_channels))
        self._common_memory_usage = self.output_tensor_compute_freq.size + input_tensor.size
        self._memory_usage = self._common_memory_usage
        return FakeTensor(self.output_tensor_shape)

    
    def get_tile_output_hw(self, input_tensor,i_h, i_w):
        output_height = input_tensor.shape[0] - (self.kernel_size)

        if i_h == 0:
            output_height += self.padding
        if i_h + input_tensor.shape[0] >= self.input_tensor_shape[0]:
            output_height += self.padding
        output_height = output_height // self.stride + 1

        output_width = input_tensor.shape[1] - (self.kernel_size)
        if i_w == 0:
            output_width += self.padding
        if i_w + input_tensor.shape[1] >= self.input_tensor_shape[1]:
            output_width += self.padding
        output_width = output_width // self.stride + 1

        return output_height, output_width
    
    @property
    def memory_usage(self):
        return self._memory_usage

    @property
    def common_memory_usage(self):
        return self._common_memory_usage
       
    @property
    def common_input_size(self):
        return reduce(lambda x,y: x*y, self.input_tensor_shape)
    
    @property
    def common_output_size(self):
        return reduce(lambda x,y: x*y, self.output_tensor_shape)
    
    @property
    def common_output_shape(self):
        return self.output_tensor_shape
    
    @property
    def common_input_shape(self):
        return self.input_tensor_shape
    
    @property
    def buffer_memory_usage(self):
        return self.tile_buffer_size

    def forward(self, input_tensor):
        return self.forward_common(input_tensor)


# Define a convolutional layer
class ConvLayer(Layer):
    def __init__(self, output_channels, kernel_size, stride, padding=0):
        super().__init__("ConvLayer", output_channels, kernel_size, stride, padding)

    def forward_common(self, input_tensor):
        self.MAC_per_element = input_tensor.shape[2] * (self.kernel_size**2)
        return super().forward_common(input_tensor)
    
    def forward(self, input_tensor,i_h=None, i_w=None, acc_pad=None):
        # Simulate convolution by calculating the output dimensions
        # output_height = (input_tensor.shape[0] - (self.kernel_size)) // self.stride + 1
        # output_width = (input_tensor.shape[1] - (self.kernel_size)) // self.stride + 1

        if i_h is None or i_w is None or acc_pad is None:
            return self.forward_common(input_tensor)
        
        output_height, output_width = self.get_tile_output_hw(input_tensor, i_h, i_w)

        if i_h != 0:
            i_h += acc_pad
        if i_w != 0:
            i_w += acc_pad

        output_tensor = FakeTensor((output_height, output_width, self.output_channels))

        self._memory_usage = output_tensor.size + input_tensor.size
        
        self.tile_buffer_size = output_tensor.size

        self.MAC_per_element = input_tensor.shape[2] * (self.kernel_size**2)

        self.output_tensor_compute_freq[i_h:i_h + output_height, i_w:i_w + output_width, :] += 1
        
        return output_tensor
    
class DepthwiseConv(Layer):
    def __init__(self, output_channels, kernel_size, stride, padding=0):
        super().__init__("DepthwiseConv", output_channels, kernel_size, stride, padding)

    def forward_common(self, input_tensor):
        self.MAC_per_element = (self.kernel_size**2)
        o_tensor = super().forward_common(input_tensor)
        self._common_memory_usage = input_tensor.size
        self._memory_usage = input_tensor.size
        return o_tensor
    
    def forward(self, input_tensor,i_h=None, i_w=None, acc_pad=None):
        # Simulate convolution by calculating the output dimensions

        if i_h is None or i_w is None or acc_pad is None:
            return self.forward_common(input_tensor)

        output_height, output_width = self.get_tile_output_hw(input_tensor, i_h, i_w)

        if i_h != 0:
            i_h += acc_pad
        if i_w != 0:
            i_w += acc_pad

        output_tensor = FakeTensor((output_height, output_width, self.output_channels))

        self._memory_usage = input_tensor.size

        self.tile_buffer_size = output_tensor.size
        self.MAC_per_element = (self.kernel_size**2)

        self.output_tensor_compute_freq[i_h:i_h + output_height, i_w:i_w + output_width, :] += 1
        
        return output_tensor

# Define a pooling layer
class PoolingLayer(Layer):
    def __init__(self, pool_size, stride, padding=0):
        super().__init__("PoolingLayer", None, pool_size, stride, padding)

    def forward_common(self, input_tensor):
        self.output_channels = input_tensor.shape[2]
        self.MAC_per_element = (self.kernel_size**2)
        return super().forward_common(input_tensor)
    

    def forward(self, input_tensor,i_h=None, i_w=None, acc_pad=None):
        # Simulate pooling by calculating output dimensions
        if i_h is None or i_w is None or acc_pad is None:
            return self.forward_common(input_tensor)

        output_height, output_width = self.get_tile_output_hw(input_tensor, i_h, i_w)

        if i_h != 0:
            i_h += acc_pad
        if i_w != 0:
            i_w += acc_pad
        output_tensor = FakeTensor((output_height, output_width, input_tensor.shape[2]))

        self.output_tensor_compute_freq[i_h:i_h + output_height, i_w:i_w + output_width, :] += 1

        self._memory_usage = output_tensor.size + input_tensor.size
        self.tile_buffer_size = output_tensor.size

        self.MAC_per_element = (self.kernel_size**2)

        return output_tensor

# Define a fused block that can contain arbitrary layers
class FusedBlock:
    def __init__(self, layers, input_tensor, block_output_size=1, cache=False):
        self.layers = layers
        self.tile_size = None
        self.stride = None
        self.set_block_output_size(block_output_size)
        self.cache = cache
        self.forward = self.forward_no_cache
        print("fusion tile size:", self.tile_size)
        print("fusion stride:", self.stride)
        # i = 1
        # for l in self.layers:
        #     l.name = f'{l.name}_{i}'
        #     i += 1
        # self.forward_common(input_tensor)

    def get_peak_mem(self):
        return reduce(lambda x, y: max(x, y), [l.common_memory_usage for l in self.layers])
    
    def get_sum_mem(self):
        return reduce(lambda x, y: x + y, [l.common_memory_usage for l in self.layers])

    @property
    def memory_usage(self):
        buffer_sum = reduce(lambda x,y: x+y, [l.buffer_memory_usage for l in self.layers])
        # TODO: mem usage with cache is not correct for some case?
        if self.cache:
            print("cache buffer usage:", buffer_sum)
            return buffer_sum + self.layers[0].common_input_size + self.layers[-1].common_output_size

        else:
            layer_mem_max =max([l.memory_usage for l in self.layers])
            print("layer peak usage:", layer_mem_max, [l.memory_usage for l in self.layers])
            return layer_mem_max + self.layers[0].common_input_size + self.layers[-1].common_output_size
    
    @property
    def aggregated_output_shape(self):
        return self.layers[-1].common_output_shape
    
    @property
    def aggregated_output_size(self):
        return self.layers[-1].common_output_size

    
    def forward_common(self, input_tensor):
        out_tensor = input_tensor
        for layer in self.layers:
            out_tensor = layer.forward_common(out_tensor)
        return out_tensor

    def forward_no_cache(self, input_tensor, tile_size=None, stride=None):
        # Process each tile in the input tensor through the layers in the fused block
        if tile_size is None:
            tile_size = self.tile_size

        if stride is None:
            stride = self.stride
        
        height, width, _ = input_tensor.shape
        for i in range(0, height - tile_size + 1, stride):
            for j in range(0, width - tile_size + 1, stride):
                tile = input_tensor[i:i+tile_size, j:j+tile_size, :]
                s = 1
                p = 0
                for layer in self.layers:
                    s *= layer.stride
                    p += layer.padding
                    tile = layer.forward(tile, i // s, j // s, p)
                    
        return tile
    
    def set_forward_cache_horizon(self):
        self.forward = self.forward_cache_horizon

    def forward_cache_horizon(self, input_tensor, tile_size=None, stride=None):
        # Process each tile in the input tensor through the layers in the fused block
        if tile_size is None:
            tile_size = self.tile_size

        if stride is None:
            stride = self.stride

        height, width, _ = input_tensor.shape
        for i in range(0, height - tile_size + 1, stride):
            tile = input_tensor[i:i+tile_size, :, :]
            s = 1
            p = 0
            for layer in self.layers:
                # breakpoint()
                s *= layer.stride
                p += layer.padding
                tile = layer.forward(tile, i // s, 0, p)
                    
        return tile

    def forward_cache_L_shape(self, input_tensor, tile_size=None, stride=None):
        # Process each tile in the input tensor through the layers in the fused block

        if tile_size is None:
            tile_size = self.tile_size

        if stride is None:
            stride = self.stride

        height, width, _ = input_tensor.shape
        start_h = 0
        start_w = 0
        is_dir_horizon = True
        for i in range(0, height - tile_size + 1, stride):
            tile = input_tensor[i:i+tile_size, i:, :]
            s = 1
            for layer in self.layers:
                # breakpoint()
                s *= layer.stride
                tile = layer.forward(tile, i // s, i // s)

            tile = input_tensor[i+stride:,i:i+tile_size, :]
            s = 1
            for layer in self.layers:
                # breakpoint()
                s *= layer.stride
                tile = layer.forward(tile, (i  + stride)// s, i // s)
                    
        return tile
    
    def set_block_output_size(self, block_output_size):
        self.tile_size, self.stride = self.calculate_tile_size_and_stride(block_output_size)
        return self.tile_size, self.stride
    
    def calculate_tile_size_and_stride(self, block_output_size):
        tile_size = block_output_size
        stride = block_output_size
        for layer in self.layers[::-1]:
            if isinstance(layer, ConvLayer | PoolingLayer | DepthwiseConv):
                tile_size = (tile_size - 1) * layer.stride + layer.kernel_size
                stride *= layer.stride
        return tile_size, stride

# Function to calculate the minimum tile size required to output exactly 1 pixel
def calculate_tile_size_and_stride(layers, block_output_size=1):
    tile_size = block_output_size
    stride = block_output_size
    for layer in layers[::-1]:
        if isinstance(layer, ConvLayer | PoolingLayer | DepthwiseConv):
            tile_size = (tile_size - 1) * layer.stride + layer.kernel_size
            stride *= layer.stride
    return tile_size, stride

# Function to visualize re-computation frequencies as 2D heatmaps for each layer
def visualize_recomp_MAC(layers):
    for layer in layers:
        # Extract the layer's re-computation data from recomp_MAC
        heatmap = layer.output_tensor_compute_freq[:,:,0] * layer.MAC_per_element
        print(f"heatmap size: {heatmap.shape[0]}x{heatmap.shape[1]}")
        
        # Plot the heatmap for this layer
        plt.figure(figsize=(6, 6))
        plt.title(f"Re-computation MAC: {layer.name}")
        plt.imshow(heatmap, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Re-computation MAC Count')
        plt.xlabel("Width")
        plt.ylabel("Height")
    plt.show()

def visualize_recomp_freq(layers):
    for layer in layers:

        # Create a 2D array to hold re-computation frequencies
        heatmap = layer.output_tensor_compute_freq[:,:,0]
        print(f"heatmap size: {heatmap.shape[0]}x{heatmap.shape[1]}")
        
        
        # Plot the heatmap for this layer
        plt.figure(figsize=(6, 6))
        plt.title(f"Re-computation Frequency: {layer.name}")
        plt.imshow(heatmap, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Re-computation Count')
        plt.xlabel("Width")
        plt.ylabel("Height")
    plt.show()

# Example: Define user-configurable layers
poc_layers = [
    ConvLayer(output_channels=64, kernel_size=3, stride=1),
    PoolingLayer(pool_size=2, stride=2),
    ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1)
]


def MBConv(intput_channel, output_channel, expansion=1, stride=1, padding=1, kernel_size=3):
    return [
        ConvLayer(output_channels=intput_channel * expansion, kernel_size=1, stride=1),
        DepthwiseConv(output_channels=intput_channel * expansion, kernel_size=kernel_size, stride=stride, padding=padding),
        ConvLayer(output_channels=output_channel, kernel_size=1, stride=1),
    ]

mobilenetv2_layers = [
    ConvLayer(output_channels=32, kernel_size=3, stride=2, padding=1),
    *MBConv(32, 16, 1, 1),
   
    *MBConv(16, 24, 6, 2),
    *MBConv(24, 24, 6, 1),

    *MBConv(24, 32, 6, 2),
    *MBConv(32, 32, 6, 1),
    *MBConv(32, 32, 6, 1),

    *MBConv(32, 64, 6, 2),
    *MBConv(64, 64, 6, 1),
    *MBConv(64, 64, 6, 1),
    *MBConv(64, 64, 6, 1),

    *MBConv(64, 96, 6, 1),
    *MBConv(96, 96, 6, 1),
    *MBConv(96, 96, 6, 1),

    *MBConv(96, 160, 6, 2),
    *MBConv(160, 160, 6, 1),
    *MBConv(160, 160, 6, 1),

    *MBConv(160, 320, 6, 1),
    ConvLayer(output_channels=1280, kernel_size=1, stride=1),
    # PoolingLayer(pool_size=7, stride=1),
]

resnet34_layers = [
    ConvLayer(output_channels=64, kernel_size=7, stride=2, padding=3),
    
    #conv2_x
    PoolingLayer(pool_size=3, stride=2, padding=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),

    #conv3_x
    ConvLayer(output_channels=128, kernel_size=3, stride=2, padding=1),
    ConvLayer(output_channels=128, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=128, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=128, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=128, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=128, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=128, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=128, kernel_size=3, stride=1, padding=1),

    #conv4_x
    ConvLayer(output_channels=256, kernel_size=3, stride=2, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),


    #conv5_x
    ConvLayer(output_channels=512, kernel_size=3, stride=2, padding=1),
    ConvLayer(output_channels=512, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=512, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=512, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=512, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=512, kernel_size=3, stride=1, padding=1),


]

mcunetv2_vww_5fps = [
    ConvLayer(output_channels=16, kernel_size=3, stride=2, padding=1),
    DepthwiseConv(16,3,1,1),
    ConvLayer(output_channels=8, kernel_size=1, stride=1, padding=0),

    *MBConv(8, 16, 6, 2),
    *MBConv(16, 16, 3, 1),
    *MBConv(16, 16, 3, 1),
    
    *MBConv(16, 24, 3, 2, 3, 7),

    *MBConv(24, 24, 6, 1),
    *MBConv(24, 24, 5, 1, 2, 5),
    *MBConv(24, 40, 6, 2, 3, 7),

    *MBConv(40, 40, 6, 1, 3, 7),

    *MBConv(40, 48, 6, 1, 1, 3),
    *MBConv(48, 48, 4, 1, 1, 3),
    *MBConv(48, 96, 5, 2, 2, 5),

    *MBConv(96, 96, 5, 1, 1, 3),
    *MBConv(96, 96, 4, 1, 1, 3),
    *MBConv(96, 160, 3, 1, 3, 7),
]

layers = mobilenetv2_layers

split_idx_1 = 13
split_idx_2 = 16

block1 = layers[0:split_idx_1]
block2 = layers[split_idx_1:split_idx_2]
block_remain = layers[split_idx_2:]
# Example input tensor (adjust dimensions based on tile size)
input_tensor = np.zeros((224, 224, 3))  # (height, width, channels)

# out_block1 = fused_block1.forward_common(input_tensor)

# print("peak b1 common memory usage:", fused_block1.get_peak_mem())
# print("sum b1 common memory usage:", fused_block1.get_sum_mem())

# # tile_size, stride = calculate_tile_size_and_stride(block1, block_output_size=1)
# # print(f"Calculated tile size: {tile_size}x{tile_size}")
# fused_block1.forward_cache_horizon(input_tensor)

# print("peak b1 memory usage:", fused_block1.memory_usage)
# # print("sum b1 memory usage:", fused_block1.get_sum_mem())

# fused_block2 = FusedBlock(block2, input_tensor)
# out_block2 = fused_block2.forward_common(out_block1)

# print("peak b2 common memory usage:", fused_block2.get_peak_mem())
# print("sum b2 common memory usage:", fused_block2.get_sum_mem())


class Network:
    def __init__(self, layers) -> None:
        self.layers = layers
        # self.input_tensor = None
    
    def forward(self, input_tensor):
        output_tensor = input_tensor
        # self.input_tensor = input_tensor
        for l in self.layers:
            output_tensor = l.forward(output_tensor)
            if isinstance(l, FusedBlock):
                output_tensor = np.zeros(l.aggregated_output_shape)
            else:
                output_tensor = np.zeros(output_tensor.shape)
        return output_tensor
    
    def calc_memory_usage(self, input_tensor, ignore_input=True, ignore_output=False):
        self.forward(input_tensor)
        memory_usage = 0
        for i,l in enumerate(self.layers):
            current_mem_usage = l.memory_usage
            if i == 0 and ignore_input:
                current_mem_usage -= input_tensor.size
            elif i == len(self.layers) - 1 and ignore_output:
                current_mem_usage -= l.aggregated_output_size
            memory_usage = max(memory_usage, current_mem_usage)          
            
            if isinstance(l, Layer):
                print(f"layer {i} mem usage: {current_mem_usage} \t",
                      f"input shape: {l.common_input_shape} \t size: {l.common_input_size} \t",
                      f"output shape: {l.common_output_shape} \t size: {l.common_output_size}")
            else:
                print(f"layer {i} mem usage: {current_mem_usage} \t",
                      f"output shape: {l.aggregated_output_shape} \t size: {l.aggregated_output_size}")
        return memory_usage

# Must run first to get original i/o shape
origin_network = Network(layers)
ori_network_mem = origin_network.calc_memory_usage(input_tensor)
print("Original Network memory usage:", ori_network_mem)

# fused_block1 = FusedBlock(block1, input_tensor, 1)
# fused_block2 = FusedBlock(block2, fused_block1.aggregated_output_shape, 1)

# fusion_network = Network([fused_block1, fused_block2, *block_remain])
# fusion_network_mem = fusion_network.calc_memory_usage(input_tensor)
# print("Fusion Network memory usage:", fusion_network_mem)

def find_minimal_mem_usage_fusion_depth(layers, input_tensor):
    depth = None
    mem_usage_lst = []
    for i,l in enumerate(layers):
        temp_fusion = FusedBlock(layers[0:i+1], input_tensor)
        if temp_fusion.tile_size >= input_tensor.shape[0] or temp_fusion.tile_size >= input_tensor.shape[1]:
            break
        temp_fusion.forward(input_tensor)
        mem_usage_lst.append(temp_fusion.memory_usage)
    if len(mem_usage_lst) != 0:
        depth = np.argmin(mem_usage_lst) + 1       
    return depth

idx = 0
blocks = []
block_input_tensor = input_tensor

while idx < len(layers):
    temp_layers = layers[idx:]
    end_idx = find_minimal_mem_usage_fusion_depth(temp_layers, block_input_tensor)
    if end_idx is not None:
        fusion_block = FusedBlock(layers[idx:idx+end_idx], block_input_tensor, block_output_size=1, cache=True)
        blocks.append(fusion_block)
        block_input_tensor = np.zeros(fusion_block.aggregated_output_shape)
    else:
        end_idx = 1
        blocks = [*blocks, *layers[idx:idx+end_idx]]
        block_input_tensor = np.zeros(blocks[-1].common_output_shape)
    idx += end_idx

origin_network = Network(layers)
ori_network_mem = origin_network.calc_memory_usage(input_tensor)
print("Original Network memory usage:", ori_network_mem)

fusion_network = Network(blocks)
_ = [l.set_forward_cache_horizon() for l in blocks]
fusion_network_mem = fusion_network.calc_memory_usage(input_tensor, ignore_output=True)
print("Fusion Network memory usage:", fusion_network_mem)


# dummy_block = FusedBlock(layers, input_tensor)
# fusion_network = layers
# layer_in_fusion_block = []
# layers_sorted_by_memory_usage = sorted(layers, key=lambda x: x.memory_usage, reverse=True)

# for l in layers_sorted_by_memory_usage:
#     l_idx = layers.index(l)
#     tem_network = fusion_network


# # Create a fused block with user-defined layers
# fused_block = FusedBlock(layers, input_tensor)

# print("peak common memory usage:", fused_block.get_peak_mem())
# print("sum common memory usage:", fused_block.get_sum_mem())

# # Calculate the tile size for exactly 1 pixel output
# tile_size, stride = calculate_tile_size_and_stride(layers, block_output_size=1)
# print(f"Calculated tile size: {tile_size}x{tile_size}")
# # 
# # breakpoint()



# # Run the fused block forward pass
# fused_block.forward(input_tensor, tile_size, stride)
# # fused_block.forward_cache_horizon(input_tensor, tile_size, stride)

# print("peak memory usage:", fused_block.get_peak_mem())
# print("sum memory usage:", fused_block.get_sum_mem())

# 
compute_total = 0
common_compute = 0

for layer in layers:
    compute_total += np.sum(layer.output_tensor_compute_freq) * layer.MAC_per_element
    common_compute += layer.output_tensor_compute_freq.size * layer.MAC_per_element

# common_compute = len(recomp_MAC)
redudant_compute = compute_total - common_compute

print(f'total:{compute_total}, common: {common_compute}, redudant:{redudant_compute}, redudant rate: {redudant_compute / compute_total}, overhead factor: {compute_total / common_compute}')

# Visualize re-computation frequencies
# visualize_recomp_freq(layers)