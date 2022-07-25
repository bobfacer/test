# def erfen(a,b,c):
#     pass
# while True:
#     try:
#         t = input()
#         #print(t)
#         while t:
#             s = input.split()
#             print(s)
#             res = erfen(a,b,c)
#             t -= 1
#     except:
#         break

# while True:
#     try:
#         a = input().split()
#         print(a)
#     except:
#         break

# def dfs(mp,i,j,n,m):
#     mp[i][j]=2
#     jump = [[-2,0],[2,0],[0,-2],[0,2]]
#     for item in jump:
#         if i+item[0]>m-1 or i+item[0]<0 or j+item[1] > n-1 or j+item[1]<0 or mp[i+item[0]][j+item[1]]==2 or mp[i+item[0]][j+item[1]]==0:
#             continue
#         dfs(mp,i+item[0],j+item[1],n,m)
#     return
# a = input().split()
# n = int(a[0]) #行数
# m = int(a[1]) #列数
# mp = [[0]*m for i in range(n)]
# for i in range(n):
#     b = input()
#     for j in range(m):
#         if b[j]=="*":
#             mp[i][j] = 1
# cnt = 0
# for i in range(n):
#     for j in range(m):
#         if mp[i][j]!=0 and mp[i][j]!=2:
#             dfs(mp,i,j,n,m)
#             cnt += 1
# print(cnt)

while True:
    try:
        a = input()
        m = input().split()
        for i in range(len(m)):
            m[i] = int(m[i])
        d = {}
        for i in m:
            for j in range(1,i+1):
                if i%j==0:
                    if i not in d:
                        d[i] = 1
                    else:
                        d[i] += 1
        newd = dict(sorted(d.items(),key=lambda x:x[1]))
        print(newd)
    except:
        break