from typing import List
from collections import defaultdict, Counter

"""
双向指针，滑动窗口。
"""


class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        """
        167.输入下标1开始的非递减有序数组，返回一组两个相加后等于target的数字的下标。
        """
        l, r = 0, len(numbers) - 1
        while l < r:
            if numbers[l] + numbers[r] > target:
                r -= 1
            elif numbers[l] + numbers[r] < target:
                l += 1
            else:
                return [l + 1, r + 1]
        return [-1, -1]

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        15.输入整数数组，返回所有和为0的不重复三元组的索引值。
        """
        nums.sort()
        ans = []
        n = len(nums)
        for i in range(n - 2):
            if nums[i] == nums[i - 1] and i > 0:
                continue
            tar = -nums[i]
            l = i + 1
            r = n - 1
            while l < r:
                s = nums[l] + nums[r]
                if s > tar:
                    r -= 1
                elif s < tar:
                    l += 1
                else:
                    ans.append([nums[i], nums[l], nums[r]])
                    r -= 1
                    while nums[r] == nums[r + 1] and r > l:
                        r -= 1
                    l += 1
                    while nums[l] == nums[l - 1] and r > l:
                        l += 1
        return ans

    def countPairs(self, nums: List[int], target: int) -> int:
        """
        2824.输入整数数组，返回数组中两数和小于target的对数。
        """
        nums.sort()
        ans = 0
        for l, x in enumerate(nums):
            y = target - x
            while l < len(nums) - 1 and nums[l + 1] < y:
                l += 1
                ans += 1
        # for right, x in enumerate(nums):
        #     y = target - x
        #     left = right - 1
        #     while left >= 0 and nums[left] >= y:
        #         left -= 1
        #     ans += left + 1
        return ans

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        """
        16.输入整数数组，返回数组中和target最接近的三数和。
        """
        nums.sort()
        n = len(nums)
        ans, min_diff = 0, float('inf')
        for i in range(n - 2):
            x = nums[i]
            if i and x == nums[i - 1]:
                continue  # 优化三

            # 优化一
            s = x + nums[i + 1] + nums[i + 2]
            if s > target:  # 后面无论怎么选，选出的三个数的和不会比 s 还小
                if s - target < min_diff:
                    ans = s  # 由于下一行直接 break，这里无需更新 min_diff
                break

            # 优化二
            s = x + nums[-2] + nums[-1]
            if s < target:  # x 加上后面任意两个数都不超过 s，所以下面的双指针就不需要跑了
                if target - s < min_diff:
                    min_diff = target - s
                    ans = s
                continue

            # 双指针
            j, k = i + 1, n - 1
            while j < k:
                s = x + nums[j] + nums[k]
                if s == target:
                    return s
                if s > target:
                    if s - target < min_diff:  # s 与 target 更近
                        min_diff = s - target
                        ans = s
                    k -= 1
                else:  # s < target
                    if target - s < min_diff:  # s 与 target 更近
                        min_diff = target - s
                        ans = s
                    j += 1
        return ans

    def triangleNumber(self, nums: List[int]) -> int:
        """
        611.输入正整数数组，返回可以组成三角形的三元组的个数。
        """
        nums.sort()
        n = len(nums)
        ans = 0
        for i in range(n - 1, 1, -1):
            l, r = 0, i - 1
            while l < r:
                if nums[l] + nums[r] <= nums[i]:
                    l += 1
                else:
                    ans += r - l
                    r -= 1

        return ans

    def maxArea(self, height: List[int]) -> int:
        """
        11.输入长为n的整数数组，数组的值为容器边的高度，返回最大的盛水面积。
        """
        l, r = 0, len(height) - 1
        ans = -1
        while l < r:
            ans = max(ans, (r - l) * min(height[r], height[l]))
            if height[r] > height[l]:
                l += 1
            else:
                r -= 1
        return ans

    def trap(self, height: List[int]) -> int:
        """
        42.输入n个非负整数的数组，宽度为1高为数组值，返回下雨后可以解多少水。
        """
        n = len(height)
        ans = 0
        pre, suf = 0, 0
        l, r = 0, n - 1
        while l <= r:
            pre = max(pre, height[l])
            suf = max(suf, height[r])
            if pre < suf:
                ans += pre - height[l]
                l += 1
            else:
                ans += suf - height[r]
                r -= 1
        return ans

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        """
        209.输入整数数组，返回子数组和大于等于target的最小长度。
        """
        n = len(nums)
        ans = n + 1
        s = left = 0
        for right, x in enumerate(nums):
            s += x
            while s - nums[left] >= target:
                s -= nums[left]
                left += 1
            if s >= target:
                ans = min(ans, right - left + 1)
        return ans if ans <= n else 0

    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        3.输入字符串，返回该字符串无重复的最长子串的长度。
        """
        ans = left = 0
        cnt = defaultdict(int)
        for right, c in enumerate(s):
            cnt[c] += 1
            while cnt[c] > 1:
                cnt[s[left]] -= 1
                left += 1
            ans = max(ans, right - left + 1)
        return ans

    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        """
        713.输入整数数组和k，返回数组中乘积严格小于k的子数组的个数。
        """
        if k <= 1: return 0
        p = 1
        l, ans = 0, 0
        for r, x in enumerate(nums):
            p *= x
            while p >= k:
                p /= nums[l]
                l += 1
            ans += r - l + 1
        return ans

    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        """
        2958.输入整数数组和k，返回好子数组的最长长度。
        好子数组：数组中每个元素的出现次数都小于等于k。
        """
        cnt = defaultdict(int)
        ans, l = -1, 0
        for r, x in enumerate(nums):
            cnt[x] += 1
            while cnt[x] > k:
                cnt[nums[l]] -= 1
                l += 1
            ans = max(ans, r - l + 1)
        return ans

    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        """
        2730.输入字符串，返回半重复子串的最大长度。
        半重复字符串：至多有一对相邻字符相等的字符串。
        """
        left = same = 0
        ans = 1
        for right in range(1, len(s)):
            same += (s[right] == s[right - 1])
            if same > 1:
                left += 1
                while s[left] != s[left - 1]:
                    left += 1
                same = 1
            ans = max(ans, right - left + 1)
        return ans

    def maximumBeauty(self, nums: List[int], k: int) -> int:
        """
        2779.输入整数数组和整数k，返回数组的最大美丽值。
        最大美丽值：数组的所有值相等的子序列的最大长度。
        操作：每次操作可以把num[i]变为[num[i]-k,num[i]+k]]中的任意值，每个数只能操作一次。
        """
        nums.sort()
        ans = left = 0
        for right, x in enumerate(nums):
            while x - nums[left] > 2 * k:
                left += 1
            ans = max(ans, right - left + 1)

        return ans

    def longestOnes(self, nums: List[int], k: int) -> int:
        """
        1004.给定一个二进制数组 nums 和一个整数 k，假设最多可以翻转 k 个 0 ，
        则返回执行操作后 数组中连续 1 的最大个数 。
        """
        _len = len(nums)
        left = 0
        for right, n in enumerate(nums):
            if n == 0:
                k -= 1
            if k < 0:
                if nums[left] == 0:
                    k += 1
                left += 1
        return _len - left

    def countSubarrays(self, nums: List[int], k: int) -> int:
        """
        2962.输入整数数组和k，返回包含nums最大元素至少 k 次的子数组个数。
        """
        mx = max(nums)
        cntmx = l = ans = 0
        for r, x in enumerate(nums):
            if x == mx: cntmx += 1
            while cntmx == k:
                if nums[l] == mx:
                    cntmx -= 1
                l += 1
            ans += l  # 一开始 l 为0，子数组越长越满足要求。
        return ans

    def countSubarrays1(self, nums: List[int], k: int) -> int:
        """
        2302.输入一个正整数数组 nums 和一个整数 k ，
        返回 nums 中分数严格小于 k 的 非空整数子数组数目。
        分数：定义为数组之和 乘以 数组的长度。
        """
        ans = l = 0
        temp = 0
        for r, x in enumerate(nums):
            temp += x
            while temp * (r - l + 1) >= k:
                temp -= nums[l]
                l += 1
            ans += r - l + 1  # 窗口总是满足条件的。
        return ans

    def minOperations(self, nums: List[int], x: int) -> int:
        """
        1658.输入一个整数数组 nums 和一个整数 x 。
        每一次操作时，你应当移除数组 nums 最左边或最右边的元素，
        然后从 x 中减去该元素的值。请注意，需要 修改 数组以供接下来的操作使用。
        如果可以将 x 恰好 减到 0 ，返回 最小操作数 ；否则，返回 -1 。
        """
        target = sum(nums) - x
        if target < 0: return -1
        tmp = l = 0
        ans = -1
        for r in range(len(nums)):
            tmp += nums[r]
            while tmp > target:
                tmp -= nums[l]
                l += 1
            if tmp == target:
                ans = max(ans, r - l + 1)
        return -1 if ans < 0 else len(nums) - ans

    def balancedString(self, s: str) -> int:
        """
        1234.输入一个字符串 s，通过「替换一个子串」的方式，使原字符串 s 变成一个「平衡字符串」。
        返回替换的最小子串长度。
        平衡字符串：四个字符都恰好出现 n/4 次。
        """
        m = len(s) / 4
        count = Counter(s)
        if len(count) == 4 and min(count.values()) == m: return 0
        ans = float('inf')
        l = 0
        for r, x in enumerate(s):
            count[x] -= 1
            while max(count.values()) <= m:
                ans = min(ans, r - l + 1)
                count[s[l]] += 1
                l += 1
        return ans

    def minWindow(self, s: str, t: str) -> str:
        """
        76.输入一个字符串 s 、一个字符串 t 。
        返回 s 中涵盖 t 所有字符的最小子串。
        如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
        """
        if not t: return ""
        ans_left, ans_right = -1, len(s)
        cnt_t = Counter(t)
        less = len(cnt_t)
        left = 0
        for right, char in enumerate(s):
            cnt_t[char] -= 1
            if cnt_t[char] == 0: less -= 1
            while less == 0:
                if right - left < ans_right - ans_left:
                    ans_left, ans_right = left, right
                out = s[left]
                if cnt_t[out] == 0:
                    less += 1
                cnt_t[out] += 1
                left += 1
        return "" if ans_left == -1 else s[ans_left: ans_right + 1]


if __name__ == '__main__':
    solution = Solution()
    nums = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    K = 3
    # print(solution.longestOnes(nums, K))
