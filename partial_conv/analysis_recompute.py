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

        self.memory_usage = None
        self.MAC_per_element = None
        self.output_tensor_shape = None
        self.output_tensor_compute_freq = None
        self.input_tensor_shape = None


    def forward_common(self, input_tensor):
        output_height = (input_tensor.shape[0]  + 2 * self.padding - self.dialation* (self.kernel_size - 1) - 1) // self.stride + 1
        output_width = (input_tensor.shape[1] + 2 * self.padding - self.dialation* (self.kernel_size - 1) - 1) // self.stride + 1
        self.output_tensor_shape = (output_height, output_width, self.output_channels)
        self.input_tensor_shape = input_tensor.shape
        self.output_tensor_compute_freq = np.ones((output_height, output_width, self.output_channels))
        self.memory_usage = self.output_tensor_compute_freq.size
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


    def forward(self, input_tensor):
        pass

# Define a convolutional layer
class ConvLayer(Layer):
    def __init__(self, output_channels, kernel_size, stride, padding=0):
        super().__init__("ConvLayer", output_channels, kernel_size, stride, padding)

    def forward_common(self, input_tensor):
        self.MAC_per_element = input_tensor.shape[2] * self.output_channels * (self.kernel_size**2)
        return super().forward_common(input_tensor)

    def forward(self, input_tensor,i_h, i_w, acc_pad):
        # Simulate convolution by calculating the output dimensions
        # output_height = (input_tensor.shape[0] - (self.kernel_size)) // self.stride + 1
        # output_width = (input_tensor.shape[1] - (self.kernel_size)) // self.stride + 1
        
        output_height, output_width = self.get_tile_output_hw(input_tensor, i_h, i_w)

        if i_h != 0:
            i_h += acc_pad
        if i_w != 0:
            i_w += acc_pad

        output_tensor = FakeTensor((output_height, output_width, self.output_channels))

        self.memory_usage = output_tensor.size
        self.MAC_per_element = input_tensor.shape[2] * self.output_channels * (self.kernel_size**2)

        self.output_tensor_compute_freq[i_h:i_h + output_height, i_w:i_w + output_width, :] += 1
        
        return output_tensor
    
class DepthwiseConv(Layer):
    def __init__(self, output_channels, kernel_size, stride, padding=0):
        super().__init__("DepthwiseConv", output_channels, kernel_size, stride, padding)

    def forward_common(self, input_tensor):
        self.MAC_per_element = self.output_channels * (self.kernel_size**2)
        return super().forward_common(input_tensor)

    def forward(self, input_tensor,i_h, i_w, acc_pad):
        # Simulate convolution by calculating the output dimensions
        output_height, output_width = self.get_tile_output_hw(input_tensor, i_h, i_w)

        if i_h != 0:
            i_h += acc_pad
        if i_w != 0:
            i_w += acc_pad

        output_tensor = FakeTensor((output_height, output_width, self.output_channels))

        self.memory_usage = output_tensor.size
        self.MAC_per_element = self.output_channels * (self.kernel_size**2)

        self.output_tensor_compute_freq[i_h:i_h + output_height, i_w:i_w + output_width, :] += 1
        
        return output_tensor

# Define a pooling layer
class PoolingLayer(Layer):
    def __init__(self, pool_size, stride, padding=0):
        super().__init__("PoolingLayer", None, pool_size, stride, padding)

    def forward_common(self, input_tensor):
        self.output_channels = input_tensor.shape[2]
        self.MAC_per_element = input_tensor.shape[2]
        return super().forward_common(input_tensor)

    def forward(self, input_tensor,i_h, i_w, acc_pad):
        # Simulate pooling by calculating output dimensions
        output_height, output_width = self.get_tile_output_hw(input_tensor, i_h, i_w)

        if i_h != 0:
            i_h += acc_pad
        if i_w != 0:
            i_w += acc_pad
        output_tensor = FakeTensor((output_height, output_width, input_tensor.shape[2]))

        self.output_tensor_compute_freq[i_h:i_h + output_height, i_w:i_w + output_width, :] += 1

        self.memory_usage = output_tensor.size
        self.MAC_per_element = input_tensor.shape[2] 

        return output_tensor

# Define a fused block that can contain arbitrary layers
class FusedBlock:
    def __init__(self, layers, input_tensor):
        self.layers = layers
        # i = 1
        # for l in self.layers:
        #     l.name = f'{l.name}_{i}'
        #     i += 1
        self.forward_common(input_tensor)

    def get_peak_mem(self):
        return reduce(lambda x, y: max(x, y), [l.memory_usage for l in self.layers])
    
    def get_sum_mem(self):
        return reduce(lambda x, y: x + y, [l.memory_usage for l in self.layers])
    
    def forward_common(self, input_tensor):
        out_tensor = input_tensor
        for layer in self.layers:
            out_tensor = layer.forward_common(out_tensor)
        return out_tensor

    def forward(self, input_tensor, tile_size, stride):
        # Process each tile in the input tensor through the layers in the fused block
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

    def forward_cache_horizon(self, input_tensor, tile_size, stride):
        # Process each tile in the input tensor through the layers in the fused block
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

    def forward_cache_L_shape(self, input_tensor, tile_size, stride):
        # Process each tile in the input tensor through the layers in the fused block
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
# layers = [
#     ConvLayer(output_channels=64, kernel_size=3, stride=1),
#     PoolingLayer(pool_size=2, stride=2),
#     ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),
#     ConvLayer(output_channels=64, kernel_size=3, stride=1)
# ]


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

layers = mobilenetv2_layers

split_idx = 10

block1 = layers[0:split_idx]
block2 = layers[split_idx:]
# Example input tensor (adjust dimensions based on tile size)
input_tensor = np.zeros((224, 224, 3))  # (height, width, channels)
fused_block1 = FusedBlock(block1, input_tensor)
out_block1 = fused_block1.forward_common(input_tensor)

print("peak b1 common memory usage:", fused_block1.get_peak_mem())
print("sum b1 common memory usage:", fused_block1.get_sum_mem())

tile_size, stride = calculate_tile_size_and_stride(block1, block_output_size=1)
print(f"Calculated tile size: {tile_size}x{tile_size}")
fused_block1.forward_cache_horizon(input_tensor, tile_size, stride)

print("peak b1 memory usage:", fused_block1.get_peak_mem())
print("sum b1 memory usage:", fused_block1.get_sum_mem())

fused_block2 = FusedBlock(block2, input_tensor)
out_block2 = fused_block2.forward_common(out_block1)

print("peak b2 common memory usage:", fused_block2.get_peak_mem())
print("sum b2 common memory usage:", fused_block2.get_sum_mem())

dummy_block = FusedBlock(layers, input_tensor)
fusion_network = layers
layer_in_fusion_block = []
layers_sorted_by_memory_usage = sorted(layers, key=lambda x: x.memory_usage, reverse=True)

for l in layers_sorted_by_memory_usage:
    l_idx = layers.index(l)
    tem_network = fusion_network


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