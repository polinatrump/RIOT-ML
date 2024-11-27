from .building_blocks import ConvLayer, DepthwiseConv, PoolingLayer

# Function to calculate the minimum tile size required to output exactly N pixel
def calculate_tile_size_and_stride(layers, block_output_size=1):
    tile_size = block_output_size
    stride = block_output_size
    for layer in layers[::-1]:
        if isinstance(layer, ConvLayer | PoolingLayer | DepthwiseConv):
            tile_size = (tile_size - 1) * layer.stride + layer.kernel_size
            stride *= layer.stride
    return tile_size, stride