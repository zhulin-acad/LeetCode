import itertools

def is_substring(small, big):
    return small in big

def remove_substrings(strings):
    """移除是其他字符串子串的元素"""
    filtered = []
    for i in range(len(strings)):
        keep = True
        for j in range(len(strings)):
            if i != j and is_substring(strings[i], strings[j]):
                keep = False
                break
        if keep:
            filtered.append(strings[i])
    return filtered

def merge(a, b):
    """将b拼接到a上，尽可能覆盖"""
    max_overlap = 0
    for i in range(1, min(len(a), len(b)) + 1):
        if a[-i:] == b[:i]:
            max_overlap = i
    return a + b[max_overlap:]

def shortest_common_superstring_length(s1, s2, s3):
    strings = remove_substrings([s1, s2, s3])
    min_len = float('inf')
    for order in itertools.permutations(strings):
        merged = order[0]
        for i in range(1, len(order)):
            merged = merge(merged, order[i])
        min_len = min(min_len, len(merged))
    return min_len


s="1234"
print(s[-2:])