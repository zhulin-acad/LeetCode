from typing import List
from bisect import bisect_left, bisect_right

"""二分查找"""


class Binary:
    # 二分查找函数返回 target 的第一个出现位置。
    def f1(self, nums: List[int], target: int) -> int:
        """闭区间二分查找"""
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
            # if nums[mid]>=target:
            #     r=mid-1
            # else:
            #     l=mid+1
        return l  # r+1

    def f2(self, nums: List[int], target: int) -> int:
        """左闭右开区间二分查找"""
        l, r = 0, len(nums)
        while l < r:
            mid = (l + r) // 2
            # if nums[mid] < target:
            #     l = mid + 1
            # else:
            #     r = mid
            if nums[mid] >= target:
                r = mid
            else:
                l = mid + 1
        return l  # r

    def f3(self, nums: List[int], target: int) -> int:
        """开区间二分查找"""
        l, r = -1, len(nums)
        while l + 1 < r:
            mid = (l + r) // 2
            if nums[mid] < target:
                l = mid
            else:
                r = mid
        return r  # l+1


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        34.输入有序数组，和target，返回target开始位置和结束位置，否则返回[-1,-1]
        """
        if target not in nums: return [-1, -1]
        # return [bisect_left(nums,target),bisect_right(nums,target)-1]

        # ans_l=ans_r=0
        # l,r=0,len(nums)-1
        # while l<=r:
        #     mid=l+(r-l)//2
        #     if nums[mid]>=target:
        #         r=mid-1
        #     else:
        #         l=mid+1
        # ans_l=l
        # l,r=0,len(nums)-1
        # while l<=r:
        #     mid=l+(r-l)//2
        #     if nums[mid]>=target+1:
        #         r=mid-1
        #     else:
        #         l=mid+1
        # ans_r=l-1
        # return [ans_l,ans_r]

        n = len(nums)
        i, j = 0, n - 1
        while i <= j:
            mid = (i + j) // 2
            if target > nums[mid]:
                i = mid + 1
            elif target < nums[mid]:
                j = mid - 1
            else:
                x, y = mid, mid
                while x - 1 >= 0 and nums[x - 1] == target:
                    x = x - 1
                while y + 1 <= n - 1 and nums[y + 1] == target:
                    y = y + 1
                return [x, y]

    def maximumCount(self, nums: List[int]) -> int:
        """
        2529.输入有序数组，返回数组中正数个数和负数个数中的最大的值。
        """
        return max(bisect_left(nums, 0), len(nums) - bisect_right(nums, 0))

    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        """
        2300.输入两个数组，返回一个数组ans，ans[i] 为 potions 中和 spells[i] 相乘的积大于
        等于success的元素个数。
        """
        ans = []
        potions.sort()
        n = len(potions)
        for x in spells:
            y = (success - 1) // x + 1
            l = bisect_left(potions, y)
            ans.append(n - l)
        return ans

    def countFairPairs(self, nums: List[int], lower: int, upper: int) -> int:
        """
        2563.返回公平数对的对数，公平数对满足 lower <= nums[i] + nums[j] <= upper
        """
        nums.sort()
        ans = 0
        for j, x in enumerate(nums):
            r = bisect_right(nums, upper - x, 0, j)
            l = bisect_left(nums, lower - x, 0, j)
            ans += r - l
        return ans

    def hIndex(self, citations: List[int]) -> int:
        """
        275.输入研究者的（非降序）论文引用次数数组，citations[i]是研究者第 i 篇论文被引用的次数。
        返回研究者的 H 指数：研究者至少有 h 篇论文被引用 h 次。
        """
        l, r = 1, len(citations)
        # if len(citations)==1:return 0 if citations[0]==0 else 1
        while l <= r:
            mid = (r + l) // 2
            if citations[-mid] >= mid:
                l = mid + 1
            else:
                r = mid - 1
        return r

    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        """
        875.这里有 n 堆香蕉，第 i 堆中有 piles[i] 根香蕉。
        警卫已经离开了，将在 h 小时后回来。珂珂可以决定她吃香蕉的速度 k （单位：根/小时）。
        每个小时，她将会选择一堆香蕉，从中吃掉 k 根。如果这堆香蕉少于 k 根，她将吃掉这堆的所有香蕉，
        然后这一小时内不会再吃更多的香蕉。  珂珂喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。
        返回她可以在 h 小时内吃掉所有香蕉的最小速度 k（k 为整数）。
        """
        # check = lambda k: sum((p - 1) // k for p in piles) <= h - len(piles)
        # return 1 + bisect_left(range(1, max(piles)), True, key=check)  # 左闭右开区间

        n = len(piles)
        left = 0  # 恒为 False
        right = max(piles)  # 恒为 True
        while left + 1 < right:  # 开区间不为空
            mid = (left + right) // 2
            if sum((p - 1) // mid for p in piles) <= h - n:
                right = mid  # 循环不变量：恒为 True
            else:
                left = mid  # 循环不变量：恒为 False
        return right  # 最小的 True

    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        """
        2187.给你一个数组 time ，其中 time[i] 表示第 i 辆公交车完成 一趟旅途 所需要花费的时间。
        每辆公交车可以 连续 完成多趟旅途，也就是说，一辆公交车当前旅途完成后，可以 立马开始 下一趟旅途。
        每辆公交车 独立 运行，也就是说可以同时有多辆公交车在运行且互不影响。
        给你一个整数 totalTrips ，表示所有公交车 总共 需要完成的旅途数目。
        请你返回完成 至少 totalTrips 趟旅途需要花费的 最少 时间。
        """
        # check = lambda x: sum(x // t for t in time) >= totalTrips
        # min_t = min(time)
        # # bisect_left 需要用左闭右开区间
        # left = min_t
        # right = min_t * totalTrips
        # return bisect_left(range(right), True, left, key=check)

        min_t = min(time)
        left = min_t - 1  # 循环不变量：sum >= totalTrips 恒为 False
        right = min_t * totalTrips  # 循环不变量：sum >= totalTrips 恒为 True
        while left + 1 < right:  # 开区间 (left, right) 不为空
            mid = (left + right) // 2
            if sum(mid // t for t in time) >= totalTrips:
                right = mid  # 缩小二分区间为 (left, mid)
            else:
                left = mid  # 缩小二分区间为 (mid, right)
        # 此时 left 等于 right-1
        # sum(left) < totalTrips 且 sum(right) >= totalTrips，所以答案是 right
        return right


if __name__ == '__main__':
    binary = Binary()
    # nums = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6]
    nums = [1, 2, 3]
    target = 0
    print(binary.f3(nums, target))
