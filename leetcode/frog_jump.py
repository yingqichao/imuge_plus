def jump(codebook, stones, curr_index, prev_jump):
    if prev_jump in codebook[curr_index]:
        return False

    if curr_index >= len(stones) - 1:
        ## final stone, success
        return True

    ## attempts
    for K in [prev_jump - 1, prev_jump, prev_jump + 1]:
        if K > 0 and (stones[curr_index] + K) in stones:
            new_index = stones.index(stones[curr_index] + K)
            status = jump(codebook, stones, new_index, K)
            if status == True:
                return True
    codebook[curr_index][prev_jump] = False
    print(f"reached index {curr_index}...")
    return False


class Solution(object):

    def canCross(self, stones):
        """
        :type stones: List[int]
        :rtype: bool
        """
        ## first step assumption
        if stones[1] != 1:
            return False

        codebook = [{}] * len(stones)
        return jump(codebook, stones, 1, 1)

if __name__ == '__main__':
    s = Solution()
    print(s.canCross(stones=[0,1,3,5,6,8,12,17]))