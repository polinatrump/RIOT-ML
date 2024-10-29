import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from functools import reduce
# Dictionary to track the re-computation frequency of each intermediate tensor element
recomp_MAC = defaultdict(int)
common_MAC = defaultdict(int)

recomp_freq = defaultdict(int)
common_freq = defaultdict(int)

# HWC format
# Define a base class for layers
class Layer:
    def __init__(self, name, output_channels, kernel_size, stride):
        self.name = name
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.memory_usage = None
        self.MAC_per_element = None

    def forward(self, input_tensor):
        pass

# Define a convolutional layer
class ConvLayer(Layer):
    def __init__(self, output_channels, kernel_size, stride):
        super().__init__("ConvLayer", output_channels, kernel_size, stride)

    def forward(self, input_tensor,i_h, i_w):
        # Simulate convolution by calculating the output dimensions
        output_height = (input_tensor.shape[0] - (self.kernel_size)) // self.stride + 1
        output_width = (input_tensor.shape[1] - (self.kernel_size)) // self.stride + 1
        output_tensor = np.zeros((output_height, output_width, self.output_channels))

        self.memory_usage = output_tensor.size
        self.MAC_per_element = input_tensor.shape[2] * self.output_channels * (self.kernel_size**2)
        
        # Track re-computation frequency for each element
        for i in range(output_tensor.shape[0]):
            for j in range(output_tensor.shape[1]):
                for k in range(output_tensor.shape[2]):
                    recomp_MAC[(self.name, i+ i_h, j + i_w, k)] += self.MAC_per_element # Increment re-computation count
                    common_MAC[(self.name, i+ i_h, j + i_w, k)] = self.MAC_per_element

                    recomp_freq[(self.name, i+ i_h, j + i_w, k)] += 1
                    common_freq[(self.name, i+ i_h, j + i_w, k)] = 1
        
        return output_tensor
    
class DepthwiseConv(Layer):
    def __init__(self, output_channels, kernel_size, stride):
        super().__init__("DepthwiseConv", output_channels, kernel_size, stride)

    def forward(self, input_tensor,i_h, i_w):
        # Simulate convolution by calculating the output dimensions
        output_height = (input_tensor.shape[0] - (self.kernel_size)) // self.stride + 1
        output_width = (input_tensor.shape[1] - (self.kernel_size)) // self.stride + 1
        output_tensor = np.zeros((output_height, output_width, self.output_channels))

        self.memory_usage = output_tensor.size
        self.MAC_per_element = self.output_channels * (self.kernel_size**2)
        
        # Track re-computation frequency for each element
        for i in range(output_tensor.shape[0]):
            for j in range(output_tensor.shape[1]):
                for k in range(output_tensor.shape[2]):
                    recomp_MAC[(self.name, i+ i_h, j + i_w, k)] += self.MAC_per_element  # Increment re-computation count
                    common_MAC[(self.name, i+ i_h, j + i_w, k)] = self.MAC_per_element

                    recomp_freq[(self.name, i+ i_h, j + i_w, k)] += 1
                    common_freq[(self.name, i+ i_h, j + i_w, k)] = 1

        return output_tensor

# Define a pooling layer
class PoolingLayer(Layer):
    def __init__(self, pool_size, stride):
        super().__init__("PoolingLayer", None, pool_size, stride)

    def forward(self, input_tensor,i_h, i_w):
        # Simulate pooling by calculating output dimensions
        output_height = (input_tensor.shape[0] - (self.kernel_size)) // self.stride + 1
        output_width = (input_tensor.shape[1] - (self.kernel_size)) // self.stride + 1
        output_tensor = np.zeros((output_height, output_width, input_tensor.shape[2]))

        self.memory_usage = output_tensor.size
        self.MAC_per_element = input_tensor.shape[2] 
        
        # Track re-computation frequency for each element
        for i in range(output_tensor.shape[0]):
            for j in range(output_tensor.shape[1]):
                for k in range(output_tensor.shape[2]):
                    recomp_MAC[(self.name, i + i_h, j + i_w, k)] += self.MAC_per_element   # Increment re-computation count
                    common_MAC[(self.name, i+ i_h, j + i_w, k)] = self.MAC_per_element 

                    recomp_freq[(self.name, i+ i_h, j + i_w, k)] += 1
                    common_freq[(self.name, i+ i_h, j + i_w, k)] = 1

        return output_tensor

# Define a fused block that can contain arbitrary layers
class FusedBlock:
    def __init__(self, layers):
        self.layers = layers
        i = 1
        for l in self.layers:
            l.name = f'{l.name}_{i}'
            i += 1

    def get_peak_mem(self):
        return reduce(lambda x, y: max(x, y), [l.memory_usage for l in self.layers])
    
    def get_sum_mem(self):
        return reduce(lambda x, y: x + y, [l.memory_usage for l in self.layers])

    def forward(self, input_tensor, tile_size, stride):
        # Process each tile in the input tensor through the layers in the fused block
        height, width, _ = input_tensor.shape
        for i in range(0, height - tile_size + 1, stride):
            for j in range(0, width - tile_size + 1, stride):
                tile = input_tensor[i:i+tile_size, j:j+tile_size, :]
                s = 1
                for layer in self.layers:
                    # breakpoint()
                    s *= layer.stride
                    tile = layer.forward(tile, i // s, j // s)
                    
        return tile

    def forward_cache_horizon(self, input_tensor, tile_size, stride):
        # Process each tile in the input tensor through the layers in the fused block
        height, width, _ = input_tensor.shape
        for i in range(0, height - tile_size + 1, stride):
            tile = input_tensor[i:i+tile_size, :, :]
            s = 1
            for layer in self.layers:
                # breakpoint()
                s *= layer.stride
                tile = layer.forward(tile, i // s, 0)
                    
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
        if isinstance(layer, ConvLayer) or isinstance(layer, PoolingLayer):
            tile_size = (tile_size - 1) * layer.stride + layer.kernel_size
            stride *= layer.stride
    return tile_size, stride

# Function to visualize re-computation frequencies as 2D heatmaps for each layer
def visualize_recomp_MAC(layers):
    for layer in layers:
        # Extract the layer's re-computation data from recomp_MAC
        layer_data = defaultdict(int)
        for key, count in recomp_MAC.items():
            if key[0] == layer.name:
                layer_data[(key[1], key[2])] = count
        # breakpoint()
        # Get the dimensions of the output for the layer
        max_i = max(key[0] for key in layer_data.keys()) + 1
        max_j = max(key[1] for key in layer_data.keys()) + 1

        # Create a 2D array to hold re-computation frequencies
        heatmap = np.zeros((max_i, max_j))
        for (i, j), count in layer_data.items():
            heatmap[i, j] = count
        print(f"heatmap size: {max_i}x{max_j}")
        
        

        # Plot the heatmap for this layer
        plt.figure(figsize=(6, 6))
        plt.title(f"Re-computation Frequency: {layer.name}")
        plt.imshow(heatmap, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Re-computation Count')
        plt.xlabel("Width")
        plt.ylabel("Height")
    plt.show()

# Example: Define user-configurable layers
layers = [
    ConvLayer(output_channels=64, kernel_size=7, stride=1),
    PoolingLayer(pool_size=2, stride=2),
    ConvLayer(output_channels=64, kernel_size=3, stride=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1)
]

# Create a fused block with user-defined layers
fused_block = FusedBlock(layers)

# Calculate the tile size for exactly 1 pixel output
tile_size, stride = calculate_tile_size_and_stride(layers, block_output_size=1)
print(f"Calculated tile size: {tile_size}x{tile_size}")

# Example input tensor (adjust dimensions based on tile size)
input_tensor = np.zeros((64, 64, 3))  # (height, width, channels)

# Run the fused block forward pass
fused_block.forward(input_tensor, tile_size, stride)

print("peak memory usage:", fused_block.get_peak_mem())
print("sum memory usage:", fused_block.get_sum_mem())
# fused_block.forward_cache_horizon(input_tensor, tile_size, stride)
# 
compute_total = 0

for key, count in recomp_MAC.items():
    compute_total += count

common_compute = 0
for key, count in common_MAC.items():
    common_compute += count

# common_compute = len(recomp_MAC)
redudant_compute = compute_total - common_compute

print(f'total:{compute_total}, redudant:{redudant_compute}, redudant rate: {redudant_compute / compute_total}, overhead factor: {compute_total / common_compute}')

# Visualize re-computation frequencies
visualize_recomp_MAC(layers)