#分数p/q的循环节前面部分的长度、以及循环节的长度
#给定大小为n的序列a,每次操作可选序列中的一个数x，把x变成大于x的最小指数或把x变成小于x的最大质数，问数全都一样最少要几次操作
#最远递增祖先
#长度为n的数组a，每次计算平均数，进行多次操作把严格大于avg的元素从数组中删除，若无可删除的数，即停止，问最多可以进行多少次这样的操作

#1.
a = int(input())
nums = list(map(int,input().split()))
#二分的思路
avg = 0
cnt = 0
while len(nums)>2:
    nums.sort()
    avg = 0
    for i in range(len(nums)):
        avg += nums[i]
    avg /= len(nums)
    #print(avg)
    if avg==nums[len(nums)-1]:
        break
    l = 0
    r = len(nums)
    while l<r:
        mid = (l+r)//2
        if nums[mid]>=avg:
            r = mid
        else:
            l = mid+1
    nums = nums[0:r]
    #print(nums)
    #print(len(nums))
    #break
    cnt += 1
if avg==nums[len(nums)-1]:
    cnt += 1
print(cnt)

#3
#先找到1w以内的质数
def shai(n):
    pick = [1]*(n+1)
    pick[0] = pick[1] = 0
    for i in range(n+1):
        if pick[i]!=0:
            for j in range(i*i,n+1,i):
                pick[j] = 0
    return [x for x in range(n+1) if pick[x]]
zhishu = shai(100)
zhishu.insert(0,1)
a = int(input())
nums = list(map(int,input().split()))
#把输入的全部映射在质数数组中
def erfen(nums,target):
    if len(nums)==0:
        return -1
    l = 0
    r = len(nums)
    while l<r:
        mid = (l+r)//2
        if nums[mid]==target:
            r = mid
        elif nums[mid]>target:
            r = mid
        elif nums[mid]<target:
            l = mid + 1
    return l
cixu = {}
s = set()
for num in nums:
    cixu[num] = erfen(zhishu, num)
    if num in zhishu:
        s.add(num)
less = 9999
#print(s)
if s:
    for ns in s:
        res = 0
        for key,value in cixu.items():
            res += abs(cixu[ns]-value)
        less = min(less,res)
else:
    for num in nums:
        res = 0
        for key,value in cixu.items():
            res += abs(cixu[num]-value)
        less = min(less,res)
print(less)

#4.
#1 6  1 1
#6 7  0 6
num = list(map(int,input().split()))
ss = str(num[0]/num[1])
if len(ss)<15:
    print(-1)
else:
    s_list = ss.split('.')
    tmp = s_list[1]
    for i in range(len(tmp)-1): #只看小数点后的部分
        if tmp[i]!=tmp[i+1]:
            print('1 1')