from .building_blocks import ConvLayer, DepthwiseConv, PoolingLayer, Network, Layer, FusedBlock
import math
import numpy as np

class DPOptimizer:
    def __init__(self) -> None:
        self.DP_R = {} # mem usage of sub-networks
        self.DP_p_range = {} # Fused Layer range of R[m, n]
        self.p = {} # mem usage of Fusion block containing layers {L_i,...,L_j}
    
    def _get_fusion_memory_usage(self, m, n):
        if m >= len(self.layers):
            return math.inf
        if m == n:
            return self.layers[m].common_memory_usage
        else:
            block_input_tensor = np.zeros(self.common_input_shapes[m])
            block = FusedBlock(self.layers[m:n+1], block_input_tensor, 1, True)
            if block.tile_size > block_input_tensor.shape[0] or block.tile_size > block_input_tensor.shape[1]:
                return math.inf
            block.forward(block_input_tensor)
            return block.memory_usage

    def get_p(self, m, n):
        if m > n:
            self.p[m, n] = math.inf
        if not (m, n) in self.p:
            self.p[m, n] = self._get_fusion_memory_usage(m, n)
        print(f"get_p: ({m}, {n})", self.p[m , n])
        return self.p[m , n]
        
    def get_dp_r(self, m, n):
        
        if m > n:
            self.DP_R[m, n] = 0
        elif m == n:
            self.DP_R[m, n] = self.get_p(m, n)
        else:
            if not (m, n) in self.DP_R:
                self.DP_R[m, n] = self.R(m, n)
        print("Get DP_R:", self.DP_R[m, n], (m, n))
        return self.DP_R[m, n]

    def R(self, m, n):
        r_min = math.inf
        l_min = None
        for j in range(m, n):
            # print("R:", (m,n,j))
            if j + 1 < n:
                t_r = max(self.get_p(m, j), self.get_dp_r(j + 1, n))
            else:
                t_r = self.get_p(m, j)
            # print("t_r:", t_r, (m, n, j))
            if t_r < r_min:
                r_min = t_r
                l_min = (m, j)
        self.DP_R[m, n] = r_min
        self.DP_p_range[m, n] = l_min
        # print("r_min:", r_min, (m, n))
        return r_min
    
    def optimize(self, layers, input_tensor):
        self.layers = layers
        network = Network(layers)
        network.forward(input_tensor)
        self.common_input_shapes = network.get_all_input_shapes()
        
        N = len(self.layers)
        print("Begin DP Searching...")
        mem_usage = self.R(0, N)
        print("DP Search Finished, min memory usage:", mem_usage)
        opt_setting = []
        layer_mem_usage = []
        
        print("Fetch optimal fusion setting")
        l_min = self.DP_p_range[0, N]
        opt_setting.append(l_min)
        layer_mem_usage.append(self.p[l_min[0], l_min[1]])
        while l_min[1] < N - 1:
            l_min = self.DP_p_range[l_min[1] + 1, N]
            opt_setting.append(l_min)
            layer_mem_usage.append(self.p[l_min[0], l_min[1]])
        print("optimal setting:", opt_setting)
        print("layer_mem_usage:", layer_mem_usage)
        return mem_usage, opt_setting


            


