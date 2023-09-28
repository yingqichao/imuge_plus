# from collections import OrderedDict
# class Solution:
#     def eraseOverlapIntervals(self, intervals):
#         intervals = sorted(intervals, key=lambda item: (item[1], item[0]))
#         # print(intervals) # [[1, 3], [2, 3], [1, 4], [3, 4]]
#
#         how_many = OrderedDict()
#         how_many.
#         def helper():
#
#
# if __name__ == '__main__':
#     s = Solution()
#     print(s.eraseOverlapIntervals([[1,4],[2,3],[3,4],[1,3]]))
import torch.nn as nn
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        ## compute the qkv matrices
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        ## shape: B, Head, L, dim
        q,k,v = qkv.unbind(0)
        attn_score = (q@k.transpose(-2,-1))*self.scale
        attn_score = attn_score.softmax(dim=-1)
        ## shape: B, L, Head, dim
        value = (attn_score@v).transpose(1,2).reshape(B,N,C)
