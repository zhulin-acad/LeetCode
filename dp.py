from typing import List


def rob(nums: List[int]) -> int:
    """
    你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，
    影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，
    如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
    """
    n = len(nums)
    if n == 0: return 0
    if n == 1: return nums[0]

    dp = [0] * n

    dp[0], dp[1] = nums[0], max(nums[0], nums[1])
    for i in range(2, n):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

    return dp[-1]


def climbStairs(n: int) -> int:
    """
    假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
    每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
    """
    if n == 1: return 1
    if n == 2: return 2
    # dp=[0]*n
    # dp[0],dp[1]=1,2
    # for i in range(2,n):
    #     dp[i]=dp[i-1]+dp[i-2]
    # return dp[-1]
    dp1, dp2 = 1, 2
    for i in range(2, n):
        dp2, dp1 = dp1 + dp2, dp2
    return dp2


def minCostClimbingStairs(cost: List[int]) -> int:
    """
    给你一个整数数组 cost ，其中 cost[i] 是从楼梯第 i 个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个或者两个台阶。
    你可以选择从下标为 0 或下标为 1 的台阶开始爬楼梯。
    """
    dp = [0] * (len(cost) + 1)
    for i in range(2, len(cost) + 1):
        dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])

    return dp[-1]


def combinationSum4(nums: List[int], target: int) -> int:
    """
    给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。
    """
    if len(nums) == 0: return 0

    dp = [0] * (target + 1)
    dp[0] = 1

    for i in range(1, target + 1):
        dp[i] = sum(dp[i - x] for x in nums if i >= x)

    return dp[-1]


def countGoodStrings(low: int, high: int, zero: int, one: int) -> int:
    """
    给你整数 zero ，one ，low 和 high ，我们从空字符串开始构造一个字符串，每一步执行下面操作中的一种：
    将 '0' 在字符串末尾添加 zero  次。
    将 '1' 在字符串末尾添加 one 次。
    以上操作可以执行任意次。
    如果通过以上过程得到一个 长度 在 low 和 high 之间（包含上下边界）的字符串，那么这个字符串我们称为 好 字符串。
    请你返回满足以上要求的 不同 好字符串数目。由于答案可能很大，请将结果对 109 + 7 取余 后返回。
    """
    MOD = 1_000_000_007
    dp = [0] * (high + 1)
    dp[0] = 1
    for i in range(1, high + 1):
        if i >= zero: dp[i] = dp[i - zero]
        if i >= one: dp[i] = (dp[i] + dp[i - one]) % MOD

    return sum(dp[low:]) % MOD


def rob2(nums: List[int]) -> int:
    """
    你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，
    这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
    给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。
    """

    def rob1(nums):
        dp1, dp2 = 0, 0
        for x in nums:
            dp2, dp1 = max(dp1 + x, dp2), dp2
        return dp2

    return max(rob1(nums[2:-1]) + nums[0], rob1(nums[1:]))


def minPathSum(grid: List[List[int]]) -> int:
    """给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。"""
    # dp=[[float('inf')]*(len(grid[0])+1) for _ in range(len(grid)+1)]
    # dp[0][1]=0
    # for i in range(len(grid)):
    #     for j in range(len(grid[0])):
    #         dp[i+1][j+1]=min(dp[i+1][j],dp[i][j+1])+grid[i][j]

    # return dp[-1][-1]

    memo = {}

    def dfs(i: int, j: int):
        if i < 0 or j < 0:
            return float('inf')
        if i == 0 and j == 0:
            return grid[i][j]
        if (i, j) in memo: return memo[(i, j)]
        memo[(i, j)] = min(dfs(i, j - 1), dfs(i - 1, j)) + grid[i][j]
        return min(dfs(i, j - 1), dfs(i - 1, j)) + grid[i][j]

    return dfs(len(grid) - 1, len(grid[0]) - 1)
