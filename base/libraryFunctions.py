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
