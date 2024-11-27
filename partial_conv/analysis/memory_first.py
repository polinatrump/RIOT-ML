from .building_blocks import ConvLayer, DepthwiseConv, PoolingLayer, Network
import math

class DPOptimizer:
    def __init__(self, layers) -> None:
        self.DP_R = {} # mem usage of sub-networks
        self.DP_p_range = {} # Fused Layer range of R[m, n]
        self.p = {} # mem usage of Fusion block containing layers {L_i,...,L_j}
        self.layers = layers
    
    def _get_fusion_memory_usage(self, m, n):
        pass

    def get_p(self, m, n):
        if m > n:
            self.p[m, n] = math.inf
            return math.inf
        if (m, n) in self.p:
            return self.p[m, n]
        else:
            self.p[m, n] = self._get_fusion_memory_usage(m, n)
            return self.p[m , n]
        
    def get_dp_r(self, m, n):
        if m > n:
            self.DP_R[m, n] = math.inf
        elif m == n:
            self.DP_R[m, n] = self.get_p(m, n)
        else:
            if not (m, n) in self.DP_R:
                self.DP_R[m, n] = self.R(m, n)

        return self.DP_R[m, n]

    def R(self, m, n):
        r_min = math.inf
        l_min = None
        for j in range(m, n):
            t_r = self.get_p(m, j) + self.get_dp_r(j + 1, n)
            if t_r < r_min:
                r_min = t_r
                l_min = (m, j)
        self.DP_R[m, n] = r_min
        self.DP_p_range[m, n] = l_min
        return r_min
            


