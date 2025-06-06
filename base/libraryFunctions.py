from collections import deque, Counter, defaultdict

fruit = "apple"

dic_int = defaultdict(int)  # 初始化值为int的字典
# print(dic_int['a']) # 0

cnt = Counter(fruit)  # 计数
# print(cnt) # Counter({'p': 2, 'a': 1, 'l': 1, 'e': 1})

q = deque([3, 4])  # 初始化队列或者栈
q.append(5)
# print(q) # deque([3, 4, 5])
# q.appendleft(2)
# print(q) # deque([2, 3, 4, 5])
# q_right = q.pop()
# print(q) # deque([2, 3, 4])
# q_left = q.popleft()
# print(q) # deque([3, 4])
# q.extend([5,6])
# print(q) # deque([3, 4, 5, 6])
# q.extendleft([1,2])
# print(q) # deque([2, 1, 3, 4, 5, 6])

li = list(q)
# print(li)

from bisect import bisect_left, bisect_right

nums = [0, 1, 1, 2, 3, 4, 5, 5, 6, 7]
l, r = bisect_left(nums, 5), bisect_right(nums, 5)
# print(l,r) # 6 8

from math import sqrt, gcd, lcm
from math import factorial, comb, perm

sqrt, gcd, lcm = sqrt(9), gcd(12, 9), lcm(12, 9)  # 开方，最大公因数，最小公倍数
# print(sqrt,gcd,lcm) # 3.0 3 36

factorial = factorial(5)  # 计算阶乘，用于组合优化
# print(factorial) # 120

comb = comb(5, 2)
perm = perm(5, 2)
print(comb)  # 10
print(perm)  # 20

from itertools import combinations, permutations

com = list(combinations([1, 2, 3], 2))
per = list(permutations([1, 2, 3], 2))
# print(list(com)  # 输出 [(1,2), (1,3), (2,3)]
# print(list(per)  # 输出 [(1,2), (1,3), (2,1), (2,3), (3,1), (3,2)]

import heapq

nums = [3, 1, 4, 1, 5, 9]
heapq.heapify(nums)
print(nums)  # 输出: [1, 1, 3, 4, 5, 9]

heapq.heappush(nums, 2)
print(nums)  # 输出：[1, 1, 2, 4, 5, 9, 3]

min_val = heapq.heappop(nums)
print(min_val)  # 输出1
print(nums)  # 输出: [1, 3, 2, 4, 5, 9]

new_min = heapq.heappushpop(nums, 0)  # 插入 0 并弹出最小元素
print(new_min)  # 输出: 0

top3 = heapq.nlargest(3, nums)  # 获取前 3 大元素
print(top3)  # 输出: [9, 5, 4]

# 实现最大堆
nums = [3, 1, 4, 1, 5, 9]
max_heap = [-x for x in nums]  # 取负数
heapq.heapify(max_heap)  # 转换为最小堆（实际为最大堆）

# 获取最大值
max_val = -heapq.heappop(max_heap)  # 取负还原
print(max_val)  # 输出: 9


# 实现对排序
def heap_sort(arr):
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]


arr = [3, 1, 4, 1, 5, 9]
sorted_arr = heap_sort(arr)
print(sorted_arr)  # 输出: [1, 1, 3, 4, 5, 9]

from datetime import date, timedelta
d = date(2024, 6, 6)  # 创建一个日期对象（年, 月, 日）
print(d.weekday()) # 3 表示星期四
print(d.year, d.month, d.day)      # 输出：2024 6 6
next_day = d + timedelta(days=1)
prev_day = d - timedelta(days=1)
d2 = date(2024, 7, 1)
print(d2 - d) # 输出：25
today = date.today()  # 获取今天的日期

import calendar
calendar.isleap(2024)  # True
