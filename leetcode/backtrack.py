def letterCombinations(digits: str):
    """给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回"""
    if not digits: return []
    phmap = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz"
    }

    def dfs(path, choice):
        if len(choice) == 0:
            res.append("".join(path))
            return

        for x in phmap[choice[0]]:
            path.append(x)
            dfs(path, choice[1:])
            path.pop()

    res = []
    dfs([], digits)
    return res


def subsets(nums):
    """给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。"""

    def dfs(path, choice):
        res.append(path[:])
        if len(choice) == 0: return

        for i in range(len(choice)):
            path.append(choice[i])
            dfs(path, choice[i + 1:])
            path.pop()

    res = []
    dfs([], nums)
    return res


def partition(s: str):
    """给你一个字符串 s，请你将 s 分割成一些 子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。"""

    def dfs(path, choice):
        if len(choice) == 0:
            res.append(path[:])
            return

        for i in range(1, len(choice) + 1):
            s = choice[:i]
            if s == s[::-1]:
                path.append(s)
                dfs(path, choice[i:])
                path.pop()

    res = []
    dfs([], s)
    return res


def combine(n: int, k: int):
    """给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。你可以按任何顺序返回答案。"""

    def dfs(path, choice):
        if len(path) == k:
            res.append(path[:])
            return
        for i in range(choice, n + 1):
            path.append(i)
            dfs(path, i + 1)
            path.pop()

    res = []
    dfs([], 1)
    return res


def combinationSum3(k: int, n: int):
    """
    找出所有相加之和为 n 的 k 个数的组合，且满足下列条件：
    只使用数字1到9
    每个数字 最多使用一次
    返回 所有可能的有效组合的列表 。该列表不能包含相同的组合两次，组合可以以任何顺序返回。
    """

    def dfs(path, choice, tar):
        if tar == 0 and len(path) == k:
            res.append(path[:])
            return
        if len(path) > k: return
        for i in range(choice, 10):
            if tar < i: break
            path.append(i)
            dfs(path, i + 1, tar - i)
            path.pop()

    res = []
    dfs([], 1, n)
    return res


def generateParenthesis(n: int):
    """数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。"""

    def dfs(path, left, right):
        if len(path) == 2 * n:
            res.append("".join(path))
            return
        if left > 0:
            path.append("(")
            dfs(path, left - 1, right)
            path.pop()
        if right > left:
            path.append(")")
            dfs(path, left, right - 1)
            path.pop()

    res = []
    dfs([], n, n)
    return res


def combinationSum(candidates, target: int):
    """
    给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。
    candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。

    """

    def dfs(path, choice, tar):
        if tar == 0:
            res.append(path[:])
            return
        for i, x in enumerate(choice):
            if tar >= x:
                path.append(x)
                dfs(path, choice[i:], tar - x)
                path.pop()

    res = []
    dfs([], candidates, target)
    return res


def permute(nums):
    """给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。"""
    res = []
    if len(nums) == 0: return [[]]
    for i in range(len(nums)):
        x = [nums[i]]
        sub_s = permute(nums[:i] + nums[i + 1:])
        for s in sub_s:
            res.append(x + s)
    return res
    # def dfs(path,choice):
    #     if len(path)==n:
    #         res.append(path[:])
    #         return
    #     for i in range(len(choice)):
    #         path.append(choice[i])
    #         dfs(path,choice[:i]+choice[i+1:])
    #         path.pop()
    # res=[]
    # n=len(nums)
    # dfs([],nums)
    # return res


if __name__ == "__main__":
    print("backtrack")
