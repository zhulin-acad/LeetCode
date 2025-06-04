"""单调队列"""
from itertools import accumulate
from typing import List
from collections import deque


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        239.给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。
        你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
        返回 滑动窗口中的最大值 。
        """
        ans = []
        q = deque()
        for i, x in enumerate(nums):
            while q and nums[q[-1]] <= x:
                q.pop()
            q.append(i)
            if i - q[0] >= k:
                q.popleft()
            if i >= k - 1:
                ans.append(nums[q[0]])
        return ans

    def maximumRobots(
            self, chargeTimes: List[int], runningCosts: List[int], budget: int
    ) -> int:
        """
        2398.你有 n 个机器人，给你两个下标从 0 开始的整数数组 chargeTimes 和 runningCosts ，
        两者长度都为 n 。第 i 个机器人充电时间为 chargeTimes[i] 单位时间，
        花费 runningCosts[i] 单位时间运行。再给你一个整数 budget 。
        运行 k 个机器人 总开销 是 max(chargeTimes) + k * sum(runningCosts) ，
        其中 max(chargeTimes) 是这 k 个机器人中最大充电时间，sum(runningCosts) 是这 k 个机器人的运行时间之和。
        请你返回在 不超过 budget 的前提下，你 最多 可以 连续 运行的机器人数目为多少。
        """
        ans = left = sum_ = 0
        n = len(chargeTimes)
        q = deque()
        for i in range(n):
            sum_ += runningCosts[i]
            while q and chargeTimes[q[-1]] <= chargeTimes[i]:
                q.pop()
            q.append(i)
            while q and chargeTimes[q[0]] + (i - left + 1) * sum_ > budget:
                if q[0] == left:
                    q.popleft()
                sum_ -= runningCosts[left]
                left += 1

            ans = max(ans, i - left + 1)
        return ans

    def maxResult(self, nums: List[int], k: int) -> int:
        """
        1696.给你一个下标从 0 开始的整数数组 nums 和一个整数 k 。
        一开始你在下标 0 处。每一步，你最多可以往前跳 k 步，但你不能跳出数组的边界。
        也就是说，你可以从下标 i 跳到 [i + 1， min(n - 1, i + k)] 包含 两个端点的任意位置。
        你的目标是到达数组最后一个位置（下标为 n - 1 ），你的 得分 为经过的所有数字之和。
        请你返回你能得到的 最大得分 。
        """
        # 递归
        # def dfs(i):
        #     if i==0:return nums[0]
        #     return max(dfs(j) for j in range(max(i-k,0),i))+nums[i]
        # return dfs(len(nums)-1)

        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        q = deque([0])
        for i in range(1, n):
            # 直接迭代
            # dp[i] = max(dp[max(i-k, 0): i]) + nums[i]
            # 单调队列优化
            if q[0] < i - k: q.popleft()
            dp[i] = dp[q[0]] + nums[i]
            while q and dp[q[-1]] <= dp[i]:
                q.pop()
            q.append(i)
        return dp[-1]

    def shortestSubarray(self, nums: List[int], k: int) -> int:
        """
        862.给你一个整数数组 nums 和一个整数 k ，找出 nums 中和至少为 k 的 最短非空子数组 ，
        并返回该子数组的长度。如果不存在这样的 子数组 ，返回 -1 。
        """
        n = len(nums)
        # pre_sum = [0] * (n + 1)
        # for i in range(n):
        #     pre_sum[i + 1] = pre_sum[i] + nums[i]
        pre_sum = list(accumulate(nums, initial=0))
        ans = n + 1
        dq = deque()

        for i in range(n + 1):
            # for j in range(i):
            #     if pre_sum[i] - pre_sum[j] >= k:
            #         ans = min(ans, i - j)
            while dq and pre_sum[i] - pre_sum[dq[0]] >= k:
                ans = min(ans, i - dq.popleft())
            while dq and pre_sum[i] <= pre_sum[dq[-1]]:
                dq.pop()
            dq.append(i)

        return ans if ans <= n else -1

    def f(self, nums, k):
        """
        整数数组和整数k，返回p-q的期望，p是均匀随机 k 长度数组的最大值，q则是均匀随机 k 长度数组的最小值。
        """
        n = len(nums)
        P = []
        Q = []
        q = deque()
        for i, x in enumerate(nums):
            while q and nums[q[-1]] <= x:
                q.pop()
            q.append(i)
            if i - q[0] >= k:
                q.popleft()
            if i >= k - 1:
                P.append(nums[q[0]])
        q.clear()
        for j, y in enumerate(nums):
            while q and nums[q[-1]] >= y:
                q.pop()
            q.append(j)
            if j - q[0] >= k:
                q.popleft()
            if j >= k - 1:
                Q.append(nums[q[0]])

        ans = (sum(P) - sum(Q)) / (n - k + 1)  # E(p-q) = E(p) - E(q)
        return ans
