"""单调栈"""
from typing import List


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """
        739.给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，
        其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。
        如果气温在这之后都不会升高，请在该位置用 0 来代替。
        """
        n = len(temperatures)
        ans = [0] * n
        st = []
        # for i in range(n-1,-1,-1):
        #     t=temperatures[i]
        #     while st and temperatures[st[-1]]<=t:
        #         st.pop()
        #     if st:
        #         ans[i]=st[-1]-i
        #     st.append(i)

        for i, t in enumerate(temperatures):
            while st and temperatures[st[-1]] < t:
                j = st.pop()
                ans[j] = i - j
            st.append(i)

        return ans

    def trap(self, height: List[int]) -> int:
        """
        42.给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，
        计算按此排列的柱子，下雨之后能接多少雨水。
        """
        # n = len(height)
        # ans = 0
        # pre, suf = 0, 0
        # l, r = 0, n - 1
        # while l <= r:
        #     pre = max(pre, height[l])
        #     suf = max(suf, height[r])
        #     if pre < suf:
        #         ans += pre - height[l]
        #         l += 1
        #     else:
        #         ans += suf - height[r]
        #         r -= 1
        # return ans

        # n = len(height)
        # ans = 0
        # pre, suf = [0] * n, [0] * n
        # pre[0] = height[0]
        # for i in range(1, n):
        #     pre[i] = max(pre[i - 1], height[i])
        # suf[n - 1] = height[n - 1]
        # for i in range(n - 2, -1, -1):
        #     suf[i] = max(suf[i + 1], height[i])
        # for i in range(n):
        #     ans += min(suf[i], pre[i]) - height[i]
        # return ans

        ans = 0
        st = []
        for i, h in enumerate(height):
            while st and h >= height[st[-1]]:
                bottom_h = height[st.pop()]
                if not st:
                    break
                l = st[-1]
                high = min(height[l], h) - bottom_h
                ans += high * (i - l - 1)
            st.append(i)
        return ans

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        496.nums1 中数字 x 的 下一个更大元素 是指 x 在 nums2 中对应位置 右侧 的 第一个 比 x 大的元素。
        给你两个 没有重复元素 的数组 nums1 和 nums2 ，下标从 0 开始计数，其中nums1 是 nums2 的子集。
        对于每个 0 <= i < nums1.length ，找出满足 nums1[i] == nums2[j] 的下标 j ，
        并且在 nums2 确定 nums2[j] 的 下一个更大元素 。
        如果不存在下一个更大元素，那么本次查询的答案是 -1 。
        返回一个长度为 nums1.length 的数组 ans 作为答案，满足 ans[i] 是如上所述的 下一个更大元素 。
        """
        idx = {x: i for i, x in enumerate(nums1)}
        ans = [-1] * len(nums1)
        st = []
        for x in nums2:
            while st and x > st[-1]:
                ans[idx[st.pop()]] = x
            if x in idx:
                st.append(x)
        return ans

    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        """
        503.给定一个循环数组 nums （ nums[nums.length - 1] 的下一个元素是 nums[0] ），
        返回 nums 中每个元素的 下一个更大元素 。
        数字 x 的 下一个更大的元素 是按数组遍历顺序，这个数字之后的第一个比它更大的数，
        这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1 。
        """
        n = len(nums)
        st = []
        ans = [-1] * n
        for i in range(2 * n):
            x = nums[i % n]
            while st and x > nums[st[-1]]:
                ans[st.pop()] = x
            if i < n:
                st.append(i)
        return ans

    def largestRectangleArea(self, heights: List[int]) -> int:
        """
        84.给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
        求在该柱状图中，能够勾勒出来的矩形的最大面积。
        """
        # n = len(heights)
        # left = [-1] * n
        # right = [n] * n
        # st = []

        # 两次遍历 + 一次遍历算答案
        # for i in range(n):
        #     while st and heights[st[-1]]>=heights[i]:
        #         st.pop()
        #     if st:
        #         left[i]=st[-1]
        #     st.append(i)
        # st.clear()
        # for j in range(n-1,-1,-1):
        #     while st and heights[st[-1]]>=heights[j]:
        #         st.pop()
        #     if st:
        #         right[j]=st[-1]
        #     st.append(j)

        # 一次遍历 + 一次遍历算答案
        # for i,h in enumerate(heights):
        #     while st and heights[st[-1]]>=h:
        #         right[st.pop()]=i
        #     if st:
        #         left[i]=st[-1]
        #     st.append(i)

        # 一次遍历算答案
        # ans=0
        # for l,r,h in zip(left,right,heights):
        #     ans=max(ans,h*(r-l-1))

        # 一次遍历直接出答案
        heights.append(-1)
        st = [-1]
        ans = 0
        for r, h in enumerate(heights):
            while len(st) > 1 and h <= heights[st[-1]]:
                i = st.pop()
                l = st[-1]
                ans = max(ans, heights[i] * (r - l - 1))
            st.append(r)

        return ans

    def maximumScore(self, nums: List[int], k: int) -> int:
        """
        1793.给你一个整数数组 nums （下标从 0 开始）和一个整数 k 。一个子数组 (i, j) 的
        分数 定义为 min(nums[i], nums[i+1], ..., nums[j]) * (j - i + 1) 。
        一个 好 子数组的两个端点下标需要满足 i <= k <= j 。
        请你返回 好 子数组的最大可能 分数 。
        """
        # 和题目84也就是上一题类似，只需要在84题代码的计算答案处加一个判断。
        # nums.append(-1)
        # st = [-1]
        # ans = 0
        # for r, h in enumerate(nums):
        #     while len(st) > 1 and h <= nums[st[-1]]:
        #         i = st.pop()
        #         l = st[-1]
        #         if l<k<r:
        #             ans = max(ans, nums[i] * (r - l - 1))
        #     st.append(r)

        # n=len(nums)
        # ans=min_h=nums[k]
        # l=r=k
        # for _ in range(n-1):
        #     if r==n-1 or (l and nums[l-1]>nums[r+1]):
        #         l-=1
        #         min_h=min(min_h,nums[l])
        #     else:
        #         r+=1
        #         min_h=min(min_h,nums[r])
        #     ans=max(ans,min_h*(r-l+1))

        n = len(nums)
        l, r, base = k - 1, k + 1, nums[k]
        ans = 0
        while True:
            while l >= 0 and nums[l] >= base:
                l -= 1
            while r < n and nums[r] >= base:
                r += 1
            ans = max(ans, base * (r - l - 1))
            base = max(-1 if l == -1 else nums[l], -1 if r == n else nums[r])
            if base == -1: break

        return ans
