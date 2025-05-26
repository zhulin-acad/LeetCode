arr=[1,2,3,5,5,6,7,7,9,10]

def binary_search_closed(arr,target):
    l,r=0,len(arr)-1
    while l<=r:
        mid=l+(r-l)//2
        if arr[mid]==target:
            return mid
        elif arr[mid]<target:
            l=mid+1
        else:
            r=mid-1
    return -1

def binary_search_open(arr,target):
    l,r=0,len(arr)
    while l<r:
        mid=l+(r-l)//2
        if arr[mid]==target:
            return mid
        elif arr[mid]>target:
            r=mid
        else:
            l=mid+1
    return -1

# for x in arr:
#     import bisect
#     print(bisect.bisect_left(arr,x))
#     print(bisect.bisect_right(arr,x))
#     # print(binary_search_closed(arr,x))
#     # print(binary_search_open(arr,x))

nums=[[1,2],[2,1],[1,2]]
one=lambda x:x[1]
print(sorted(nums,key=lambda x:x[1]))












