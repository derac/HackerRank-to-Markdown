# [Matrix Script](https://www.hackerrank.com/challenges/matrix-script/problem) - 100.0 - python3
```python
import re
h,w = map(int,input().split())
M = [[*input()] for _ in range(h)]
d = ''.join(M[j][i] for i in range(w) for j in range(h))
print(re.sub('(?<=[a-zA-Z0-9])[^a-zA-Z0-9]+(?=[a-zA-Z0-9])',' ',d))
```
# [Validating Postal Codes](https://www.hackerrank.com/challenges/validating-postalcode/problem) - 80.0 - python3
```python
regex_integer_in_range = r"[1-9]\d{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(.)(?=.\1)"	# Do not delete 'r'.
```
# [Matrix Layer Rotation ](https://www.hackerrank.com/challenges/matrix-rotation-algo/problem) - 80.0 - python3
```python
h,w,r=(map(int,input().split()));m=eval("input().split(),"*h)
for n in range(min(h,w)//2):
 f,x,v=h-n-1,w-n,n+1;l=m[n][n:x]+[y[x-1]for y in m[v:f]]+([y[n]for y in m[v:f]]+m[f][n:x])[::-1];o=r%len(l);l,z=l[o:]+l[:o],x-v-n;m[n][n:x],m[f][n:x]=l[:x-n],l[f+z:h+2*z][::-1]
 for y in range(v,f):m[y][n],m[y][-v]=l[n-y],l[z+y]
for y in m:print(*y)
```
# [Array Manipulation](https://www.hackerrank.com/challenges/crush/problem) - 60.0 - python3
```python
(n,m),D = map(int,input().split()),{}
for _ in range(m):
    a, b, k = map(int,input().split())
    D[a-1] = D[a-1]+k if a-1 in D else k
    D[b] = D[b]-k if b in D else -k
high = acc = 0
for key in sorted(D.keys()):
   acc += D[key]
   high = max(high,acc)
print(high)
```
# [Prim's (MST) : Special Subtree](https://www.hackerrank.com/challenges/primsmstsub/problem) - 60.0 - python3
```python
from collections import defaultdict
E = defaultdict(set)
for a,b,w in (map(int,input().split()) for _ in
              range(int(input().split()[1]))):
    E[a]|={(b,w)}; E[b]|={(a,w)}
T,t = {int(input())},0
while E.keys() != T:
    e,w = min((I for I in set.union(*(E[v] for v in T))
               if I[0] not in T), key=lambda x:x[1])
    t+=w; T|={e}
print(t)
```
# [Common Child](https://www.hackerrank.com/challenges/common-child/problem) - 60.0 - pypy3
```python
X,Y = input(),input(); m = min(len(X),len(Y))+1
C = [[0 for _ in range(m)] for _ in range(m)]
for i in range(1,m):
    for j in range(1,m):
        C[j][i] = (C[j-1][i-1]+1 if X[i-1]==Y[j-1]
                   else max(C[j-1][i],C[j][i-1]))
print(C[m-1][m-1])
```
# [Breadth First Search: Shortest Reach](https://www.hackerrank.com/challenges/bfsshortreach/problem) - 55.0 - python3
```python
from collections import defaultdict
for _ in range(int(input())):
    (n,m),E = map(int,input().split()),defaultdict(set)
    for _ in range(m):
        u,v = map(int,input().split())
        E[u]|={v}; E[v]|={u}
    s,R = int(input()),[]
    for e in (i for i in range(1,n+1) if i!=s):
        C = H = {s}; t = 0
        while C and e not in C:
            C = set.union(*(E[v] for v in C)) - H
            H |= C; t += 6
        R += [t if e in C else -1]
    print(*R)
```
# [Stock Maximize](https://www.hackerrank.com/challenges/stockmax/problem) - 50.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'stockmax' function below.
#
# The function is expected to return a LONG_INTEGER.
# The function accepts INTEGER_ARRAY prices as parameter.
#

def stockmax(prices):
    total,l,high = 0,len(prices),0
    for i in range(l-1,-1,-1):
        if prices[i]>high: high=prices[i]
        else: total += high-prices[i]
    return total

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input().strip())
    for t_itr in range(t):
        n = int(input().strip())
        prices = list(map(int, input().rstrip().split()))
        result = stockmax(prices)
        fptr.write(str(result) + '\n')
    fptr.close()
```
# [Xor and Sum](https://www.hackerrank.com/challenges/xor-and-sum/problem) - 50.0 - python3
```python
#!/bin/python3

import os
import sys

#
# Complete the xorAndSum function below.
#
def xorAndSum(a, b):
    a,b=int(a,2),int(b,2)
    return sum(a^(b<<i) for i in range(0,314160))%(10**9+7)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    a = input()
    b = input()
    result = xorAndSum(a, b)
    fptr.write(str(result) + '\n')
    fptr.close()
```
# [Word Order](https://www.hackerrank.com/challenges/word-order/problem) - 50.0 - python3
```python
from collections import Counter
C = Counter(open(0).read().splitlines()[1:])
print(len(C))
print(*C.values())
```
# [No Idea!](https://www.hackerrank.com/challenges/no-idea/problem) - 50.0 - python3
```python
input()
I,A,B = (input().split() for _ in range(3))
A,B={*A},{*B}
print(sum([(i in A)-(i in B) for i in I]))
```
# [Piling Up!](https://www.hackerrank.com/challenges/piling-up/problem) - 50.0 - python3
```python
for _ in range(int(input())):
    input();l=[*map(int,input().split())]
    l.insert((len(l)+1)//2,0) if (len(l)%2!=0) else l
    for i in range(int(len(l)/2)):
        if (max(l[i],l[len(l)-1-i])<max(l[i+1],l[len(l)-2-i])):
            print("No"); break
    else: print ("Yes")
```
# [Maximize It!](https://www.hackerrank.com/challenges/maximize-it/problem) - 50.0 - python3
```python
from itertools import product
k,m=map(int,input().split())
L=[[*map(int,input().split()[1:])] for _ in range(k)]
print(max(sum(n**2 for n in C)%m for C in product(*L)))
```
# [Sherlock and Anagrams](https://www.hackerrank.com/challenges/sherlock-and-anagrams/problem) - 50.0 - python3
```python
#!/bin/python3

import os

def sherlockAndAnagrams(s):
    anagrams,l=0,len(s)
    for i in range(1,l):
        for j in range(0,l-i):
            t=sorted(s[j:i+j])
            for k in range(j+1,l-i+1):
                if t==sorted(s[k:k+i]):
                    anagrams+=1
    return anagrams

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        s = input()

        result = sherlockAndAnagrams(s)

        fptr.write(str(result) + '\n')

    fptr.close()
```
# [Connected Cells in a Grid](https://www.hackerrank.com/challenges/connected-cell-in-a-grid/problem) - 50.0 - python3
```python
def neighbors(i,j):
    return {(x,y) for y in range(j-1,j+2)
            for x in range(i-1,i+2) if (x,y)!=(i,j)}
            
n,m = int(input()),int(input())
M = [[*map(int,input().split())] for _ in range(n)]
ones = {(i,j) for j in range(n) for i in range(m) if M[j][i]==1}
G = []

while ones:
    i,j = ones.pop()
    G.append({(i,j)})
    neighboring_ones = neighbors(i,j) & ones
    while neighboring_ones:
        i,j = neighboring_ones.pop()
        G[-1].add((i,j))
        ones.remove((i,j))
        neighboring_ones |= (neighbors(i,j) & ones)

print(max(len(S) for S in G))
```
# [Almost Sorted](https://www.hackerrank.com/challenges/almost-sorted/problem) - 50.0 - python3
```python
n = int(input())
A = [*map(int,input().split())]
S = sorted(A)
if A == S: print("yes")
else:
    D = {i:A[i] for i,e in enumerate(S) if e!=A[i]}
    K, is_swap = [*D.keys()], len(D)==2
    if is_swap: A[K[0]],A[K[1]] = A[K[1]],A[K[0]]
    else: A = A[:K[0]] + A[K[0]:K[-1]+1][::-1] + A[K[-1]+1:]
    if A == S: print("yes\n"+("swap" if is_swap else "reverse"),K[0]+1,K[-1]+1)
    else: print("no")
```
# [Pairs](https://www.hackerrank.com/challenges/pairs/problem) - 50.0 - python3
```python
from collections import Counter
(_,k),A = (map(int,input().split()) for _ in range(2))
C = Counter(A); S = sorted(C.keys())
print(sum(C[n+k]*C[n] for n in S if n+k in C.keys()))
```
# [Snakes and Ladders: The Quickest Way Up](https://www.hackerrank.com/challenges/the-quickest-way-up/problem) - 50.0 - python3
```python
for _ in range(int(input())):
    LS = dict(e for D in ({k:v for k,v in I} for I in 
              ((map(int, input().split()) for _ in range(int(input())))
               for _ in range(2))) for e in D.items())
    m,N,H = 0,{1},{1}
    while N and 100 not in N:
        N = {(LS[e] if e in LS else e) for e in
             set.union(*({*range(n+1,n+7)} for n in N))} - H
        H|=N; m+=1
    print(m if 100 in N else -1)
```
# [Journey to the Moon](https://www.hackerrank.com/challenges/journey-to-the-moon/problem) - 50.0 - python3
```python
(n,p) = map(int,input().split())
P = [{*input().split()} for _ in range(p)]
for i in range(len(P)-1,0,-1):
  for j in range(i):
    if not P[i].isdisjoint(P[j]): P[j] |= P[i]; P.pop(i); break
P = [len(S) for S in P]; s,t,o = 0,0,n-sum(P)
for l in P: t+=s*l; s+=l  
print(t + s*o + o*(o-1)//2)
```
# [Even Tree](https://www.hackerrank.com/challenges/even-tree/problem) - 50.0 - python3
```python
from collections import defaultdict
(n,e),E = map(int,input().split()),defaultdict(set)
for f,t in (map(int,input().split()) for _ in range(e)):E[t]|={f}
def ΣT(n): return len(E[n])+sum(ΣT(e) for e in E[n])
print(sum(ΣT(i)%2 for i in range(2,n)))
```
# [Kruskal (MST): Really Special Subtree](https://www.hackerrank.com/challenges/kruskalmstrsub/problem) - 50.0 - python3
```python
from collections import defaultdict
(n,e),S,t = map(int,input().split()),defaultdict(set),0
E = sorted(([*map(int,input().split())] for _ in range(e)),
           key=lambda x:x[2])
for a,b,w in E:
    C=H={a}
    while C and b not in C: C=set.union(*(S[v] for v in C))-H; H|=C
    if b not in C: S[a]|={b}; S[b]|={a}; t+=w
print(t)
```
# [Largest Rectangle ](https://www.hackerrank.com/challenges/largest-rectangle/problem) - 50.0 - pypy3
```python
m,n,S = 0,int(input()),[[*map(int,input().split())]]
while S:
    B=S.pop(); l=min(B); m=max(l*len(B),m); i=B.index(l)
    S+=([B[:i]] if B[:i] else [])+([B[i+1:]] if B[i+1:] else [])
print(m)
```
# [Missing Numbers](https://www.hackerrank.com/challenges/missing-numbers/problem) - 45.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the missingNumbers function below.
def missingNumbers(arr, brr):
    for a in arr:brr.remove(a)
    return sorted([*{*brr}])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    m = int(input())

    brr = list(map(int, input().rstrip().split()))

    result = missingNumbers(arr, brr)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [Fibonacci Modified](https://www.hackerrank.com/challenges/fibonacci-modified/problem) - 45.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the fibonacciModified function below.
def fibonacciModified(t1, t2, n):
    for i in range(n-2):
        t1,t2=t2,t1+t2**2
    return t2

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t1T2n = input().split()

    t1 = int(t1T2n[0])

    t2 = int(t1T2n[1])

    n = int(t1T2n[2])

    result = fibonacciModified(t1, t2, n)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Merge Sort: Counting Inversions](https://www.hackerrank.com/challenges/ctci-merge-sort/problem) - 45.0 - python3
```python
def merge(A,swaps = 0):
    length = len(A); half = length//2
    if length > 1:
        L,R = A[:half],A[half:]
        swaps = merge(L,swaps) + merge(R,swaps) 
        x,y,z,l,r = 0,0,0,len(L),len(R)
        while x<l and y<r:
            if L[x]>R[y]: A[z]=R[y]; y+=1; swaps+=(l-x)
            else:         A[z]=L[x]; x+=1
            z+=1
        while x<l: A[z]=L[x]; x+=1; z+=1
        while y<r: A[z]=R[y]; y+=1; z+=1
    return swaps

for _ in range(int(input())):
    _,A,swaps = input(),[*map(int,input().split())],0
    print(merge(A))
```
# [New Year Chaos](https://www.hackerrank.com/challenges/new-year-chaos/problem) - 40.0 - python
```
#!/bin/python

import math
import os
import random
import re
import sys

# Complete the minimumBribes function below.
def minimumBribes(q):
    n = len(q)
    end_state = [i for i in range(1,n+1)]
    swaps = 0

    #print(q)

    while(q != end_state):
        no_swaps = True
        for i in range(n - 2, -1, -1):
            #print("looking at %s: " % i, q, q[i], q[i + 1])
            if q[i] > i + 1 and q[i + 1] <= i + 2:
                if q[i] > i + 3:
                    print("Too chaotic")
                    return
                q[i], q[i + 1] = q[i + 1], q[i]
                #print("swapping at %s:" % i, q)
                swaps += 1
                no_swaps = False
        #print("looped", no_swaps, swaps)
        if (no_swaps):
            print("Too chaotic")
            return
    
    print(swaps)
        


if __name__ == '__main__':
    t = int(raw_input())

    for t_itr in xrange(t):
        n = int(raw_input())

        q = map(int, raw_input().rstrip().split())

        minimumBribes(q)
```
# [Minimum Swaps 2](https://www.hackerrank.com/challenges/minimum-swaps-2/problem) - 40.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the minimumSwaps function below.
def minimumSwaps(arr):
    swaps, n = 0, len(arr)
    indexes = [0]*n
    # 2 3 4 1 5
    for i in range(n):
        indexes[arr[i] - 1] = i
    #fptr.write('array:   ' + str(arr) + '\nindexes: ' + str(indexes) + '\n')
    for i in range(n):
        if (arr[i] != i + 1):
            indexes[arr[i] - 1] = indexes[i]
            arr[i],arr[indexes[i]] = arr[indexes[i]],arr[i]
            
            #fptr.write('array:   ' + str(arr) + '\nindexes: ' + str(indexes) + '\n')
            swaps += 1
    return swaps

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    res = minimumSwaps(arr)
    fptr.write(str(res) + '\n')
    fptr.close()
```
# [Flipping bits](https://www.hackerrank.com/challenges/flipping-bits/problem) - 40.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the flippingBits function below.
def flippingBits(n):
    return (2**32-1) ^ n

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        n = int(input())

        result = flippingBits(n)

        fptr.write(str(result) + '\n')

    fptr.close()
```
# [Permuting Two Arrays](https://www.hackerrank.com/challenges/two-arrays/problem) - 40.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the twoArrays function below.
def twoArrays(k, A, B):
    A,B = sorted(A),sorted(B,reverse=True)
    return ('NO','YES')[all([i+j>=k for i,j in zip(A,B)])]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(input())
    for q_itr in range(q):
        nk = input().split()
        n = int(nk[0])
        k = int(nk[1])
        A = list(map(int, input().rstrip().split()))
        B = list(map(int, input().rstrip().split()))
        result = twoArrays(k, A, B)
        fptr.write(result + '\n')
    fptr.close()
```
# [Jim and the Orders](https://www.hackerrank.com/challenges/jim-and-the-orders/problem) - 40.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the jimOrders function below.
def jimOrders(O):
    return sorted(range(1,len(O)+1),key=lambda i:O[i-1][0]+O[i-1][1])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    orders = []

    for _ in range(n):
        orders.append(list(map(int, input().rstrip().split())))

    result = jimOrders(orders)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [Zig Zag Sequence](https://www.hackerrank.com/challenges/zig-zag-sequence/problem) - 40.0 - python3
```python
def findZigZagSequence(a, n):
    a.sort()
    mid = int((n + 1)/2 - 1)
    a[mid], a[n-1] = a[n-1], a[mid]

    st = mid + 1
    ed = n - 2
    while(st <= ed):
        a[st], a[ed] = a[ed], a[st]
        st = st + 1
        ed = ed - 1

    for i in range (n):
        if i == n-1:
            print(a[i])
        else:
            print(a[i], end = ' ')
    return

test_cases = int(input())
for cs in range (test_cases):
    n = int(input())
    a = list(map(int, input().split()))
    findZigZagSequence(a, n)
```
# [Sherlock and Array](https://www.hackerrank.com/challenges/sherlock-and-array/problem) - 40.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the balancedSums function below.
def balancedSums(A):
    l = len(A)
    if l == 1: return 'YES'
    left = 0
    right = sum(A[1:])
    for i in range(l-1):
        if left == right: return 'YES'
        left += A[i]
        right -= A[i+1]
    return 'NO'


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    T = int(input().strip())

    for T_itr in range(T):
        n = int(input().strip())

        arr = list(map(int, input().rstrip().split()))

        result = balancedSums(arr)

        fptr.write(result + '\n')

    fptr.close()
```
# [Merge the Tools!](https://www.hackerrank.com/challenges/merge-the-tools/problem) - 40.0 - python3
```python
def merge_the_tools(S, k):
    for i in range(len(S)//k):
        t = S[i*k:(i+1)*k]
        u = {t.index(c):c for c in (chr(i) for i in range(65,91)) if c in t}
        u = ''.join(u[i] for i in sorted(u.keys()))
        print(u)
```
# [ginortS](https://www.hackerrank.com/challenges/ginorts/problem) - 40.0 - python3
```python
print(*sorted(input(),key=lambda c:(-ord(c)//32,(int(c)%2==0,int(c))if c.isdigit()else c)),sep='')
```
# [Iterables and Iterators](https://www.hackerrank.com/challenges/iterables-and-iterators/problem) - 40.0 - python3
```python
from itertools import combinations
input(); A,k=input().split(),int(input())
C = [*combinations(A,k)]
print(sum('a' in c for c in C)/len(C))
```
# [Validating UID ](https://www.hackerrank.com/challenges/validating-uid/problem) - 40.0 - python3
```python
import re
R = ['[A-Z].*[A-Z]','[\d].*[\d].*[\d]','^[a-zA-Z\d]{10}$']
for _ in range(int(input())):
    u = input()
    print(('Invalid','Valid')[all(re.search(r,u)!=None for r in R)
                              and not re.search(r'(.).*\1',u)])
```
# [Validating Credit Card Numbers](https://www.hackerrank.com/challenges/validating-credit-card-number/problem) - 40.0 - python3
```python
import re
for _ in range(int(input())):
    c = input()
    dash_format = (True,all(len(s)==4 for s in c.split('-')))['-' in c]
    if '-' in c: c = c.replace('-','')
    print(('Invalid','Valid')[dash_format and
                              re.search('^\d{16}$',c)!=None and
                              re.search('^[456]',c)!=None and
                              re.search(r'(.)\1\1\1',c)==None])
```
# [The Minion Game](https://www.hackerrank.com/challenges/the-minion-game/problem) - 40.0 - python3
```python
def minion_game(S):
    V,k,s,l =['A','E','I','O','U'],0,0,len(S)
    for i in range(l):
        if S[i] in V: k+=l-i
        else: s+=l-i
    else: print("Draw" if s==k else
                ("Stuart"+" "+ "%d"%s if s>k
                else "Kevin"+" "+'%d'%k))
```
# [The Full Counting Sort](https://www.hackerrank.com/challenges/countingsort4/problem) - 40.0 - python3
```python
n = int(input())//2
A = [[] for _ in range(100)]
for _ in range(n):
    i = int(input().split()[0])
    A[i].append('-')
for _ in range(n):
    i,s = input().split()
    i = int(i)
    A[i].append(s)
print(' '.join([e for e in A for e in e]))
```
# [Fraudulent Activity Notifications](https://www.hackerrank.com/challenges/fraudulent-activity-notifications/problem) - 40.0 - python3
```python
from collections import Counter
from bisect import insort

def median(Count,Order,d):
    target = d//2 - (not d%2)
    for i in range(len(Order)):
        target -= Count[Order[i]]
        if target < 0:
            if d%2: return Order[i]
            return (Order[i] + Order[i+(target>-2)])/2

def activityNotifications(E, days):
    acc, C = 0, Counter(E[:days])
    O = sorted(C.keys())
    for i in range(d,len(E)):
        if E[i] >= median(C,O,days)*2: acc += 1
        if C[E[i-days]] == 1: O.remove(E[i-days])
        if C[E[i]] == 0: insort(O,E[i])
        C[E[i-days]] -= 1; C[E[i]] += 1
    print(acc)

d = int(input().split()[1])
activityNotifications([*map(int, input().strip().split())], d)
```
# [Larry's Array](https://www.hackerrank.com/challenges/larrys-array/problem) - 40.0 - python3
```python
for _ in range(int(input())):
    n,A=int(input()),[*map(int,input().split())]
    c=sum(A[i]>A[j] for i in range(n)
                    for j in range(i+1,n))
    print(("YES","NO")[c%2])
```
# [Absolute Permutation](https://www.hackerrank.com/challenges/absolute-permutation/problem) - 40.0 - python3
```python
for _ in range(int(input())):
    n,k = map(int,input().split())
    if k==0: print(*range(1,n+1))
    elif (n/k)%2: print(-1)
    else: print(*(i+1+(k,-k)[(i//k)%2] for i in range(n)))
```
# [The Bomberman Game](https://www.hackerrank.com/challenges/bomber-man/problem) - 40.0 - python3
```python
r,c,s = map(int,input().split())
if not s%2:print(*(['O'*c]*r),sep='\n'); quit()
D,H = ((1,0),(-1,0),(0,1),(0,-1),(0,0)),0
B = [[c=="O" for c in input()] for _ in range(r)]
for t in range((s-1)//2):
    B = [[not any(B[j+y][i+x] for x,y in D
                  if 0<=i+x<c and 0<=j+y<r)
          for i in range(c)] for j in range(r)]
    if B == H: break
    if t%2!=s//2%2: H = B
for R in B: print(*(('.','O')[n] for n in R),sep='')
```
# [Ema's Supercomputer](https://www.hackerrank.com/challenges/two-pluses/problem) - 40.0 - python3
```python
from itertools import combinations
(r,c),D = map(int,input().split()),((0,1),(0,-1),(1,0),(-1,0))
G,S = [[c=='G' for c in [*input()]] for _ in range(r)],[]
for i,j in ((i,j) for j in range(r) for i in range(c)):
    if G[j][i]:
        N,S,f = {(i,j)},S+[{(i,j)}],1
        while all(0<=i+x*f<c and 0<=j+y*f<r and 
                  G[j+y*f][i+x*f] for x,y in D):
            N |= {(i+x*f,j+y*f) for x,y in D}
            S,f = S+[N.copy()], f+1
print(max(len(A)*len(B) for A,B in 
          combinations(S,2) if A.isdisjoint(B)))
```
# [Prime Dates](https://www.hackerrank.com/challenges/prime-date/problem) - 40.0 - python3
```python
import re
month = []

def updateLeapYear(year):
    if year % 400 == 0:
        month[2] = 29
    elif year % 100 == 0:
        month[2] = 28
    elif year % 4 == 0:
        month[2] = 29
    else:
        month[2] = 28

def storeMonth():
    month[1] = 31
    month[2] = 28
    month[3] = 31
    month[4] = 30
    month[5] = 31
    month[6] = 30
    month[7] = 31
    month[8] = 31
    month[9] = 30
    month[10] = 31
    month[11] = 30
    month[12] = 31

def findPrimeDates(d1, m1, y1, d2, m2, y2):
    storeMonth()
    result = 0

    while(True):
        x = d1
        x = x * 100 + m1
        x = x * 10000 + y1
        if x % 4 == 0 or x % 7 == 0:
            result = result + 1
        if d1 == d2 and m1 == m2 and y1 == y2:
            break
        updateLeapYear(y1)
        d1 = d1 + 1
        if d1 > month[m1]:
            m1 = m1 + 1
            d1 = 1
            if m1 > 12:
                y1 =  y1 + 1
                m1 = 1
    return result;

for i in range(1, 15):
    month.append(31)

line = input()
date = re.split('-| ', line)
d1 = int(date[0])
m1 = int(date[1])
y1 = int(date[2])
d2 = int(date[3])
m2 = int(date[4])
y2 = int(date[5])

result = findPrimeDates(d1, m1, y1, d2, m2, y2)
print(result)
```
# [Frequency Queries](https://www.hackerrank.com/challenges/frequency-queries/problem) - 40.0 - python3
```python
C,A,max_f = {},[],0
for q,v in (input().split() for _ in range(int(input()))):
    if q == '1':
        C[v] = C.get(v,0)+1
        max_f = max(max_f,C[v])
    elif q == '3': A.append(int(int(v)<=max_f and int(v) in C.values()))
    elif v in C and C[v]:
        C[v] -= 1
        max_f = max(C.values())
print(*A,sep='\n')
```
# [Special String Again](https://www.hackerrank.com/challenges/special-palindrome-again/problem) - 40.0 - python3
```python
from re import findall; _,s = input(),input()
Over1ch = set(M[0]+M[2]+M[0] for M in
              findall(r'(?=((.?)\2*)(.?)\1)',s) if M[0])
print(sum(len(findall(r'(?=%s)'%m,s)) for m in Over1ch)+len(s))
```
# [Find the Median](https://www.hackerrank.com/challenges/find-the-median/problem) - 35.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the findMedian function below.
def findMedian(arr):
    return sorted(arr)[len(arr)//2]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    result = findMedian(arr)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Greedy Florist](https://www.hackerrank.com/challenges/greedy-florist/problem) - 35.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the getMinimumCost function below.
def getMinimumCost(n, k, c):
    price = 0
    c.sort()
    for p in range(math.ceil(n/k)):
        price += sum([f*(p+1) for f in c[max(0,n-(p+1)*k):n-p*k]])
    return price

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = int(nk[0])
    k = int(nk[1])
    c = list(map(int, input().rstrip().split()))
    minimumCost = getMinimumCost(n, k, c)
    fptr.write(str(minimumCost) + '\n')
    fptr.close()
```
# [Mark and Toys](https://www.hackerrank.com/challenges/mark-and-toys/problem) - 35.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the maximumToys function below.
def maximumToys(prices, k):
    prices.sort()
    for toy in range(len(prices)):
        k -= prices[toy]
        if k < 0: return toy
    return len(prices)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = int(nk[0])
    k = int(nk[1])
    prices = list(map(int, input().rstrip().split()))
    result = maximumToys(prices, k)
    fptr.write(str(result) + '\n')
    fptr.close()
```
# [Closest Numbers](https://www.hackerrank.com/challenges/closest-numbers/problem) - 35.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the closestNumbers function below.
def closestNumbers(A):
    m = 32**2-1
    A.sort()
    result = []
    for i in range(len(A)-1):
        t = abs(A[i]-A[i+1])
        if t == m: result += [A[i],A[i+1]]
        if t < m:
            result = [A[i],A[i+1]]
            m = t
    return result
            


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    result = closestNumbers(arr)
    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')
    fptr.close()
```
# [Max Min](https://www.hackerrank.com/challenges/angry-children/problem) - 35.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the maxMin function below.
def maxMin(k, A):
    A.sort()
    return min(A[i+k-1]-A[i] for i in range(len(A)-k+1))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    k = int(input())

    arr = []

    for _ in range(n):
        arr_item = int(input())
        arr.append(arr_item)

    result = maxMin(k, arr)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Bigger is Greater](https://www.hackerrank.com/challenges/bigger-is-greater/problem) - 35.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the biggerIsGreater function below.
def biggerIsGreater(w):
    for i in range(2,len(w)+1):
        for j in range(1,i):
            if w[-i] < w[-j]:
                w = [*w]
                w[-i],w[-j]=w[-j],w[-i]
                w = w[:-i+1] + sorted(w[-i+1:])
                return ''.join(w)
    return 'no answer'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    T = int(input())

    for T_itr in range(T):
        w = input()

        result = biggerIsGreater(w)

        fptr.write(result + '\n')

    fptr.close()
```
# [KnightL on a Chessboard](https://www.hackerrank.com/challenges/knightl-on-chessboard/problem) - 35.0 - python3
```python
def next_moves(a,b,i,j,n):
    M = ((a+i,b+j),(a+i,b-j),(a-i,b+j),(a-i,b-j),
         (a+j,b+i),(a+j,b-i),(a-j,b+i),(a-j,b-i))
    return {(a,b) for a,b in M if 0<=a<n and 0<=b<n}

n = int(input()); match = (n-1,n-1)
B = [[[] for _ in range(1,n)] for _ in range(1,n)]
for i,j in ((i,j) for i in range(1,n) for j in range(i,n)):
    moves = history = {(0,0)}; turn = 0
    while match not in moves:
        moves = set.union(*(next_moves(a,b,i,j,n)
                            for a,b in moves)) - history
        if not moves: turn = -1; break
        history |= moves; turn += 1
    B[i-1][j-1] = B[j-1][i-1] = turn;
for r in B: print(*r)
```
# [Sorting: Comparator](https://www.hackerrank.com/challenges/ctci-comparator-sorting/problem) - 35.0 - python3
```python
from functools import cmp_to_key
class Player:
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def __repr__(self):
        print(name, score)
        
    def comparator(p1,p2):
        if p1.score>p2.score: return -1
        if p1.score==p2.score:
            if p1.name<p2.name: return -1
            if p1.name==p2.name: return 0
            return 1
        return 1
```
# [Count Triplets](https://www.hackerrank.com/challenges/count-triplets-1/problem) - 35.0 - python3
```python
from collections import defaultdict
r,C,D,t = int(input().split()[1]),defaultdict(int),defaultdict(int),0
for n in map(int,input().split()[::-1]):
    if n*r in C: t += D[n*r]; D[n] += C[n*r]
    C[n] += 1
print(t)
```
# [Encryption](https://www.hackerrank.com/challenges/encryption/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the encryption function below.
def encryption(s):
    s = s.replace(" ", "")
    l = len(s)
    c = math.ceil(math.sqrt(l))
    return " ".join(("".join((
                    sub[i] if i < len(sub) else ""
                    for sub in (s[i : i + c] for i in range(0, l, c))
                    )) for i in range(c)))



if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = encryption(s)

    fptr.write(result + '\n')

    fptr.close()
```
# [Hex Color Code](https://www.hackerrank.com/challenges/hex-color-code/problem) - 30.0 - python3
```python
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
for _ in range(n):
    s = input()
    if len(s) and s[0].isspace():
        m = re.findall('#[0-9A-Fa-f]{3,6}', s)
        [print(h) for h in m]
```
# [Making Anagrams](https://www.hackerrank.com/challenges/making-anagrams/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the makingAnagrams function below.
def makingAnagrams(s1, s2):
    return sum([abs(s1.count(c)-s2.count(c))for c in[*map(chr,range(97,123))]])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s1 = input()

    s2 = input()

    result = makingAnagrams(s1, s2)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Intro to Tutorial Challenges](https://www.hackerrank.com/challenges/tutorial-intro/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the introTutorial function below.
def introTutorial(V, arr):
    return arr.index(V)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    V = int(input())

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    result = introTutorial(V, arr)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Insertion Sort - Part 1](https://www.hackerrank.com/challenges/insertionsort1/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the insertionSort1 function below.
def insertionSort1(n, A):
    n -= 1
    e = A[n]
    while n > 0 and A[n-1] > e:
        A[n] = A[n-1]
        print(*A)
        n -= 1
    A[n] = e
    print(*A)


if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)
```
# [Correctness and the Loop Invariant](https://www.hackerrank.com/challenges/correctness-invariant/problem) - 30.0 - python3
```python
def insertion_sort(A):
    i=1
    while i<len(A):
        j=i
        while j > 0 and A[j-1] > A[j]:
            A[j-1],A[j]=A[j],A[j-1]
            j-=1
        i+=1
    return A

m = int(input().strip())
ar = [int(i) for i in input().strip().split()]
insertion_sort(ar)
print(" ".join(map(str,ar)))
```
# [Insertion Sort - Part 2](https://www.hackerrank.com/challenges/insertionsort2/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the insertionSort2 function below.
def insertionSort2(n, A):
    for i in range(1,n):
        j = i
        while j>0 and A[j-1]>A[j]:
            A[j-1],A[j]=A[j],A[j-1]
            j-=1
        print(*A)

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
```
# [3D Surface Area](https://www.hackerrank.com/challenges/3d-surface-area/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the surfaceArea function below.
def surfaceArea(A,h,w):
    outside_vert       = sum(A[y][0]+A[y][-1]for y in range(h))
    outside_horizontal = sum(A[0][x]+A[-1][x]for x in range(w))
    inside_vert      = sum(abs(A[y][x]-A[y+1][x])for y in range(h-1) for x in range(w))
    inside_horizonal = sum(abs(A[y][x]-A[y][x+1])for x in range(w-1) for y in range(h))
    return h*w*2 + outside_vert + outside_horizontal + inside_vert + inside_horizonal

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    HW = input().split()
    H = int(HW[0])
    W = int(HW[1])
    A = []
    for _ in range(H):
        A.append(list(map(int, input().rstrip().split())))
    result = surfaceArea(A,H,W)
    fptr.write(str(result) + '\n')
    fptr.close()
```
# [Manasa and Stones](https://www.hackerrank.com/challenges/manasa-and-stones/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

from itertools import combinations_with_replacement as c_w_r

# Complete the stones function below.
def stones(n,a,b):
    return sorted([*{*[sum(c) for c in c_w_r([a,b],n-1)]}])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    T = int(input())
    for T_itr in range(T):
        n = int(input())
        a = int(input())
        b = int(input())
        result = stones(n,a,b)
        fptr.write(' '.join(map(str, result)))
        fptr.write('\n')
    fptr.close()
```
# [Priyanka and Toys](https://www.hackerrank.com/challenges/priyanka-and-toys/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the toys function below.
def toys(weights):
    weights.sort()
    max_w = weights[0] + 4
    containers = 1
    for w in weights:
        if w > max_w:
            max_w = w + 4
            containers += 1
    return containers




if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    w = list(map(int, input().rstrip().split()))

    result = toys(w)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Ice Cream Parlor](https://www.hackerrank.com/challenges/icecream-parlor/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the icecreamParlor function below.
def icecreamParlor(m, A):
    for i in range(len(A)):
        if m - A[i] in A[i+1:]: return (i+1,A.index(m-A[i],i+1)+1)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        m = int(input())

        n = int(input())

        arr = list(map(int, input().rstrip().split()))

        result = icecreamParlor(m, arr)

        fptr.write(' '.join(map(str, result)))
        fptr.write('\n')

    fptr.close()
```
# [Counting Sort 2](https://www.hackerrank.com/challenges/countingsort2/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the countingSort function below.
def countingSort(arr):return sorted(arr)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    result = countingSort(arr)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [Counting Sort 1](https://www.hackerrank.com/challenges/countingsort1/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the countingSort function below.
def countingSort(A):
    count = [0]*100
    for e in A:
        count[e] += 1
    return count

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    result = countingSort(arr)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [Running Time of Algorithms](https://www.hackerrank.com/challenges/runningtime/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the runningTime function below.
def runningTime(A):
    n,ops = len(A),0
    for i in range(1,n):
        j = i
        while j>0 and A[j-1]>A[j]:
            A[j-1],A[j]=A[j],A[j-1]
            ops+=1
            j-=1
    return ops
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    result = runningTime(arr)
    fptr.write(str(result) + '\n')
    fptr.close()
```
# [Maximizing XOR](https://www.hackerrank.com/challenges/maximizing-xor/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

from itertools import combinations_with_replacement

# Complete the maximizingXor function below.
def maximizingXor(l, r):
    return max([a^b for a,b in combinations_with_replacement([*range(l,r+1)],2)])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    l = int(input())

    r = int(input())

    result = maximizingXor(l, r)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Cavity Map](https://www.hackerrank.com/challenges/cavity-map/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the cavityMap function below.
def cavityMap(G):
    l,x_y_list = len(G),[]
    for i in range(l): grid[i] = [*map(int,G[i])]

    for x,y in [(x,y) for x in range(1,l-1) for y in range(1,l-1)]:
        if G[y][x] > max(G[y+1][x],G[y-1][x],G[y][x+1],G[y][x-1]):
            x_y_list.append((x,y))

    for x,y in x_y_list: G[y][x] = 'X'
    for i in range(l):
        grid[i] = ''.join(map(str,G[i]))
    return grid
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    grid = []
    for _ in range(n):
        grid_item = input()
        grid.append(grid_item)
    result = cavityMap(grid)
    fptr.write('\n'.join(result))
    fptr.write('\n')
    fptr.close()
```
# [Sherlock and The Beast](https://www.hackerrank.com/challenges/sherlock-and-the-beast/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the decentNumber function below.
def decentNumber(n):
    f = n//3*3
    t = n%3
    while t%5 != 0 and f > 0:
        f -= 3
        t += 3
    if t%5==0 and f%3==0:
        print('5'*f + '3'*t)
    else: print(str(-1))
    

if __name__ == '__main__':
    t = int(input().strip())
    for t_itr in range(t):
        n = int(input().strip())
        decentNumber(n)
```
# [Modified Kaprekar Numbers](https://www.hackerrank.com/challenges/kaprekar-numbers/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the kaprekarNumbers function below.
def kaprekarNumbers(p, q):
    result = []
    for n in range(p,q+1):
        t,d = str(n**2),len(str(n))
        t1 = int(t[:-d]) if t[:-d] else 0
        t2 = int(t[-d:])
        if t1+t2 == n: result += [n]
    print(' '.join(map(str,result)) if result else 'INVALID RANGE')

if __name__ == '__main__':
    p = int(input())
    q = int(input())
    kaprekarNumbers(p, q)
```
# [Sansa and XOR](https://www.hackerrank.com/challenges/sansa-and-xor/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

from functools import reduce

# Complete the sansaXor function below.
def sansaXor(A):
    return (0,reduce(lambda a,b:a^b,A[::2]))[len(A)%2]


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        n = int(input())
        arr = list(map(int, input().rstrip().split()))
        result = sansaXor(arr)
        fptr.write(str(result) + '\n')
    fptr.close()
```
# [Game of Thrones - I](https://www.hackerrank.com/challenges/game-of-thrones/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the gameOfThrones function below.
def gameOfThrones(s):
    F = dict((chr(n),0) for n in range(97,123))
    for c in s: F[c] = not F[c]
    return ('NO','YES')[sum(F.values())<2]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = gameOfThrones(s)

    fptr.write(result + '\n')

    fptr.close()
```
# [Day 0: Hello, World.](https://www.hackerrank.com/challenges/30-hello-world/problem) - 30.0 - python3
```python
# Read a full line of input from stdin and save it to our dynamically typed variable, input_string.
input_string = input()

# Print a string literal saying "Hello, World." to stdout.
print('Hello, World.')

# TODO: Write a line of code here that prints the contents of input_string to stdout.
print(input_string)
```
# [Largest Permutation](https://www.hackerrank.com/challenges/largest-permutation/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the largestPermutation function below.
def largestPermutation(k, A):
    l = len(A)
    if k >= int(l * math.log(l)): return sorted(A, reverse=True)
    i, D = 0, {e:i for i,e in enumerate(A)}
    while i != k:
        n = l-i
        if A[i] != n:
            index = D[n]
            D[A[i]] = index
            A[i],A[index] = A[index],A[i]
        else: k = min(k+1,l)
        i += 1
    return A

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = int(nk[0])

    k = int(nk[1])

    arr = list(map(int, input().rstrip().split()))

    result = largestPermutation(k, arr)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [Day 1: Data Types](https://www.hackerrank.com/challenges/30-data-types/problem) - 30.0 - python3
```python
for a,b in zip((i,d,s),(int(input()),float(input()),input())):print(a+b)
```
# [Strange Counter](https://www.hackerrank.com/challenges/strange-code/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the strangeCounter function below.
def strangeCounter(t):
    return 3*(2**math.ceil(math.log(1+t/3)/math.log(2))-1)+1-t

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    result = strangeCounter(t)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Happy Ladybugs](https://www.hackerrank.com/challenges/happy-ladybugs/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the happyLadybugs function below.
def happyLadybugs(b):
    if any(b.count(c)==1 for c in map(chr,range(65,91))):
        return 'NO'
    return ('NO','YES')[b.count('_')>0
                        or all(b[i-1]==b[i] or b[i+1]==b[i]
                               for i in range(1,len(b)-1))]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    g = int(input())

    for g_itr in range(g):
        n = int(input())

        b = input()

        result = happyLadybugs(b)

        fptr.write(result + '\n')

    fptr.close()
```
# [Day 2: Operators](https://www.hackerrank.com/challenges/30-operators/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the solve function below.
def solve(meal_cost, tip_percent, tax_percent):
    print(str(round((tip_percent + tax_percent + 100) / 100 * meal_cost)))

if __name__ == '__main__':
    meal_cost = float(input())

    tip_percent = int(input())

    tax_percent = int(input())

    solve(meal_cost, tip_percent, tax_percent)
```
# [Day 3: Intro to Conditional Statements](https://www.hackerrank.com/challenges/30-conditional-statements/problem) - 30.0 - python3
```python
N = int(input())
print(('Not Weird','Weird')[N%2 or N in range(6,21)])
```
# [Day 4: Class vs. Instance](https://www.hackerrank.com/challenges/30-class-vs-instance/problem) - 30.0 - python3
```python
class Person:
    def __init__(self,initialAge):
        if age < 0:
            print("Age is not valid, setting age to 0.")
            initialAge = 0
        self.age = initialAge

    def amIOld(self):
        if self.age < 13:
            print("You are young.")
        elif self.age < 18:
            print("You are a teenager.")
        else:
            print("You are old.")

    def yearPasses(self):
        self.age += 1
```
# [Flipping the Matrix](https://www.hackerrank.com/challenges/flipping-the-matrix/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the flippingMatrix function below.
def flippingMatrix(M):
    s = len(M)//2
    return sum(max(M[y][x],M[y][-x-1],M[-y-1][x],M[-y-1][-x-1])
               for x,y in [(x,y) for x in range(s) for y in range(s)])
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        n = int(input())

        matrix = []

        for _ in range(2*n):
            matrix.append(list(map(int, input().rstrip().split())))

        result = flippingMatrix(matrix)

        fptr.write(str(result) + '\n')

    fptr.close()
```
# [Beautiful Pairs](https://www.hackerrank.com/challenges/beautiful-pairs/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

from collections import Counter

def beautifulPairs(A, B):
    d = sum((Counter(A)-Counter(B)).values())
    return len(A)-d+(1,-1)[d==0]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    A = list(map(int, input().rstrip().split()))

    B = list(map(int, input().rstrip().split()))

    result = beautifulPairs(A, B)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Day 5: Loops](https://www.hackerrank.com/challenges/30-loops/problem) - 30.0 - python3
```python
n = int(input())
for i in range(1,11):print(n,'x',i,'=',n*i)
```
# [Day 6: Let's Review](https://www.hackerrank.com/challenges/30-review-loop/problem) - 30.0 - python3
```python
for s in open(0).read().split('\n')[1:]:print(s[::2],s[1::2])
```
# [Day 7: Arrays](https://www.hackerrank.com/challenges/30-arrays/problem) - 30.0 - python3
```python
input();print(*input().split()[::-1])
```
# [Time Delta](https://www.hackerrank.com/challenges/python-time-delta/problem) - 30.0 - python3
```python
from datetime import datetime as dt
for i in range(int(input())):
    t1,t2=eval("dt.strptime(input(),'%a %d %b %Y %H:%M:%S %z'),"*2)
    print(int(abs((t1-t2).total_seconds())))
```
# [Company Logo](https://www.hackerrank.com/challenges/most-commons/problem) - 30.0 - python3
```python
from collections import Counter as C
for c in C(sorted(input())).most_common(3):print(*c)
```
# [Reduce Function](https://www.hackerrank.com/challenges/reduce-function/problem) - 30.0 - python3
```python
def product(fracs):
    t = reduce(lambda a,b:a*b,fracs)
    return t.numerator, t.denominator
```
# [HTML Parser - Part 1](https://www.hackerrank.com/challenges/html-parser-part-1/problem) - 30.0 - python3
```python
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        for a in attrs:
            print("-> %s > %s"%a)
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        for a in attrs:
            print("-> %s > %s"%a)

# instantiate the parser and fed it some HTML
parser = MyHTMLParser()
parser.feed(''.join(open(0).read().splitlines()[1:]))
```
# [HTML Parser - Part 2](https://www.hackerrank.com/challenges/html-parser-part-2/problem) - 30.0 - python3
```python
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if '\n' in data:
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')
        print(data)

    def handle_data(self, data):
        if not data.isspace():
            print('>>> Data')
            print(data)
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()
```
# [Detect HTML Tags, Attributes and Attribute Values](https://www.hackerrank.com/challenges/detect-html-tags-attributes-and-attribute-values/problem) - 30.0 - python3
```python
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for a in attrs:
            print("-> %s > %s"%a)
    def handle_startendtag(self, tag, attrs):
        print(tag)
        for a in attrs:
            print("-> %s > %s"%a)

parser = MyHTMLParser()
parser.feed(''.join(open(0).read().splitlines()[1:]))
```
# [Standardize Mobile Number Using Decorators](https://www.hackerrank.com/challenges/standardize-mobile-number-using-decorators/problem) - 30.0 - python3
```python
def wrapper(f):
    def format_phone_numbers(L):
        f('+91 %s %s'%(s[-10:-5],s[-5:]) for s in L)
    return format_phone_numbers
```
# [Decorators 2 - Name Directory](https://www.hackerrank.com/challenges/decorators-2-name-directory/problem) - 30.0 - python3
```python
def person_lister(f):
    def inner(people):
        return map(f,sorted(people,key=lambda p:int(p[2])))
    return inner
```
# [Athlete Sort](https://www.hackerrank.com/challenges/python-sort-sort/problem) - 30.0 - python3
```python
n=int(input().split()[0])
A=[[*map(int,input().split())] for _ in range(n)]
k=int(input())
for a in sorted(A,key=lambda x:x[k]):print(*a)
```
# [Day 8: Dictionaries and Maps](https://www.hackerrank.com/challenges/30-dictionaries-and-maps/problem) - 30.0 - python3
```python
I=open(0).read().splitlines();n=int(I[0])+1
D=dict(i.split() for i in I[1:n])
for s in I[n:]:print(s+'='+D[s] if s in D else 'Not found')
```
# [Default Arguments](https://www.hackerrank.com/challenges/default-arguments/problem) - 30.0 - python3
```python
def print_from_stream(n, stream=EvenStream()):
    stream.__init__()
    for _ in range(n):
        print(stream.get_next())
```
# [Day 9: Recursion 3  ](https://www.hackerrank.com/challenges/30-recursion/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the factorial function below.
def factorial(n):
    if n < 2: return 1
    else: return n * factorial(n - 1)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = factorial(n)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Day 10: Binary Numbers](https://www.hackerrank.com/challenges/30-binary-numbers/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys
from itertools import groupby


if __name__ == '__main__':
    b = bin(int(input()))[2:]
    print(max(len([*g]) for k,g in groupby(b) if k=='1'))
```
# [Day 11: 2D Arrays](https://www.hackerrank.com/challenges/30-2d-arrays/problem) - 30.0 - python3
```python
A = [[*map(int,input().split())] for _ in range(6)]
print(max(sum(A[j][i:i+3]+[A[j+1][i+1]]+A[j+2][i:i+3])
      for i,j in ((i,j) for i in range(4) for j in range(4))))
```
# [Day 12: Inheritance](https://www.hackerrank.com/challenges/30-inheritance/problem) - 30.0 - python3
```python
class Student(Person):

    #   Class Constructor
    #   
    #   Parameters:
    #   firstName - A string denoting the Person's first name.
    #   lastName - A string denoting the Person's last name.
    #   id - An integer denoting the Person's ID number.
    #   scores - An array of integers denoting the Person's test scores.
    #
    # Write your constructor here
    def __init__(self, firstName, lastName, idNumber, scores):
        self.firstName = firstName
        self.lastName = lastName
        self.idNumber = idNumber
        self.scores = scores

    #   Function Name: calculate
    #   Return: A character denoting the grade.
    #
    # Write your function here
    def calculate(self):
        avg = sum(self.scores)//len(self.scores)
        if avg >= 90: return 'O'
        if avg >= 80: return 'E'
        if avg >= 70: return 'A'
        if avg >= 55: return 'P'
        if avg >= 40: return 'D'
        else: return 'T'
```
# [Organizing Containers of Balls](https://www.hackerrank.com/challenges/organizing-containers-of-balls/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the organizingContainers function below.
def organizingContainers(C):
    r = sorted(sum(i) for i in C)
    c = sorted(sum(i) for i in zip(*C))
    return(('Possible','Impossible')[r!=c])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        n = int(input())

        container = []

        for _ in range(n):
            container.append(list(map(int, input().rstrip().split())))

        result = organizingContainers(container)

        fptr.write(result + '\n')

    fptr.close()
```
# [Day 13: Abstract Classes](https://www.hackerrank.com/challenges/30-abstract-classes/problem) - 30.0 - python3
```python
#Write MyBook class
class MyBook(Book):
    def __init__(self,title,author,price):
        super().__init__(title, author)
        self.price = price
    def display(self):
        print("Title:", self.title)
        print("Author:", self.author)
        print("Price:", self.price)
```
# [Day 14: Scope](https://www.hackerrank.com/challenges/30-scope/problem) - 30.0 - python3
```python
def computeDifference(self):
        s = sorted(self.__elements)
        self.maximumDifference = abs(s[0]-s[-1])
	# Add your code here
```
# [Day 15: Linked List](https://www.hackerrank.com/challenges/30-linked-list/problem) - 30.0 - python3
```python
def insert(self,head,data):
        if head==None: return Node(data)
        current = head
        while current:
            previous = current
            current = current.next
        previous.next = Node(data)
        return head
```
# [Sorting: Bubble Sort](https://www.hackerrank.com/challenges/ctci-bubble-sort/problem) - 30.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the countSwaps function below.
def countSwaps(A):
    l, swaps = len(A), 0
    for i in range(l):
        for j in range(l-1):
            if A[j] > A[j+1]:
                A[j],A[j+1] = A[j+1],A[j]
                swaps += 1
    print('Array is sorted in %s swaps.'%swaps)
    print('First Element:',A[0])
    print('Last Element:',A[-1])

if __name__ == '__main__':
    n = int(input())

    a = list(map(int, input().rstrip().split()))

    countSwaps(a)
```
# [Day 16: Exceptions - String to Integer](https://www.hackerrank.com/challenges/30-exceptions-string-to-integer/problem) - 30.0 - python3
```python
try:
    print(int(input()))
except:
    print("Bad String")
```
# [Day 17: More Exceptions](https://www.hackerrank.com/challenges/30-more-exceptions/problem) - 30.0 - python3
```python
class Calculator():
    def power(self,n,p):
        if n<0 or p<0:
            raise Exception('n and p should be non-negative')
        return n**p
```
# [Queen's Attack II](https://www.hackerrank.com/challenges/queens-attack-2/problem) - 30.0 - python3
```python
(n,k),moves = map(int,input().split()),0
Qr,Qc = tuple(map(int,input().split()))
Obstacles = {tuple(map(int,input().split())) for _ in range(k)}
for Dr,Dc in ((0,1),(1,1),(1,0),(1,-1),
              (0,-1),(-1,-1),(-1,0),(-1,1)):
    Cr,Cc = Qr+Dr,Qc+Dc
    while (Cr,Cc) not in Obstacles and 0<Cr<=n and 0<Cc<=n:
        Cr,Cc,moves = Cr+Dr,Cc+Dc,moves+1
print(moves)
```
# [The Grid Search](https://www.hackerrank.com/challenges/the-grid-search/problem) - 30.0 - python3
```python
import re
for _ in range(int(input())):
    R,C = map(int,input().split())
    G = [input() for _ in range(R)]
    r,c = map(int,input().split())
    P = [input() for _ in range(r)]
    match = False
    for i in range(R-r+1):
        for M in re.finditer('(?='+P[0]+')',G[i]):
            for j in range(1,r):
                if P[j] != G[i+j][M.start():M.start()+c]: break
            else: match = True; break
        if match: break
    print("YES" if match else "NO")
```
# [Day 18: Queues and Stacks](https://www.hackerrank.com/challenges/30-queues-stacks/problem) - 30.0 - python3
```python
from collections import deque

class Solution:
    def __init__(self):
        self.S,self.Q = [],deque()

    def pushCharacter(self,c):
        self.S.append(c)
    
    def enqueueCharacter(self,c):
        self.Q.append(c)
    
    def popCharacter(self):
        return self.S.pop()
    
    def dequeueCharacter(self):
        return self.Q.popleft()
```
# [Day 19: Interfaces](https://www.hackerrank.com/challenges/30-interfaces/problem) - 30.0 - python3
```python
from math import sqrt, ceil

class Calculator(AdvancedArithmetic):
    def divisorSum(self, n):
        if n==1: return 1
        sq = sqrt(n); csq = ceil(sq)
        return sum((i+n//i) for i in range(1,csq) if n%i==0) + (csq if sq.is_integer() else 0)
```
# [Day 20: Sorting](https://www.hackerrank.com/challenges/30-sorting/problem) - 30.0 - python3
```python
n = int(input().strip())
A = [*map(int, input().strip().split())]
total = 0
for i in range(n):
    swaps = 0
    for j in range(n-1):
        if A[j] > A[j+1]:
            A[j],A[j+1] = A[j+1],A[j]
            swaps += 1
    total += swaps
    if not swaps: break

print("Array is sorted in %s swaps." % total)
print("First Element:",A[0])
print("Last Element:",A[-1])
```
# [Day 21: Generics](https://www.hackerrank.com/challenges/30-generics/problem) - 30.0 - csharp
```
public static void PrintArray<T>(T[] A) {
    foreach (T v in A) {
        Console.WriteLine(v);
    }
}
```
# [Day 22: Binary Search Trees](https://www.hackerrank.com/challenges/30-binary-search-trees/problem) - 30.0 - python3
```python
def getHeight(self,root):
        if root==None: return -1
        return 1 + max(self.getHeight(root.left),self.getHeight(root.right))
```
# [Day 23: BST Level-Order Traversal](https://www.hackerrank.com/challenges/30-binary-trees/problem) - 30.0 - python3
```python
def levelOrder(self,root):
        from collections import deque
        Q = deque([root])
        while Q:
            tree = Q.popleft()
            print(tree.data,end=' ')
            if tree.left: Q.append(tree.left)
            if tree.right: Q.append(tree.right)
```
# [Day 24: More Linked Lists](https://www.hackerrank.com/challenges/30-linked-list-deletion/problem) - 30.0 - python3
```python
def removeDuplicates(self,head):
        current,items = head,set()
        while current:
            if current.data in items: previous.next = current.next
            else: items.add(current.data); previous = current
            current = current.next
        return head
```
# [Day 25: Running Time and Complexity](https://www.hackerrank.com/challenges/30-running-time-and-complexity/problem) - 30.0 - python3
```python
from math import sqrt
for _ in range(int(input())):
    n = int(input())
    if n<2: p = False
    elif n<4: p = True
    elif n%2==0: p = False
    else:
        for i in range(3,int(sqrt(n))+1,2):
            if n%i==0: p = False; break
        else: p = True
    print('Prime' if p else 'Not prime')
```
# [Day 26: Nested Logic](https://www.hackerrank.com/challenges/30-nested-logic/problem) - 30.0 - python3
```python
(d2,m2,y2),(d1,m1,y1) = (map(int,input().split()) for _ in range(2))
if y2 == y1:
    if m2 == m1:
        if d2 <= d1: f = 0
        else: f = 15 * (d2 - d1)
    elif m2 < m1: f = 0
    else: f = 500 * (m2 - m1)
elif y2 < y1: f = 0
else: f = 10000
print(f)
```
# [Day 27: Testing](https://www.hackerrank.com/challenges/30-testing/problem) - 30.0 - python3
```python
class TestDataEmptyArray(object):
    
    @staticmethod
    def get_array():
        return []

class TestDataUniqueValues(object):

    @staticmethod
    def get_array():
        return [1,2]

    @staticmethod
    def get_expected_result():
        return 0

class TestDataExactlyTwoDifferentMinimums(object):

    @staticmethod
    def get_array():
        return [1,1]

    @staticmethod
    def get_expected_result():
        return 0
```
# [Day 29: Bitwise AND](https://www.hackerrank.com/challenges/30-bitwise-and/problem) - 30.0 - python3
```python
from itertools import combinations
for _ in range(int(input())):
    n,k = map(int,input().split())
    print(k-1 if (k-1) | k <= n else k-2)
```
# [Day 28: RegEx, Patterns, and Intro to Databases](https://www.hackerrank.com/challenges/30-regex-patterns/problem) - 30.0 - python3
```python
import re
print(*sorted(n for n,e in (input().split() for _ in range(int(input()))) if re.search('@gmail.com',e)),sep='\n')
```
# [Binary Search Tree : Lowest Common Ancestor](https://www.hackerrank.com/challenges/binary-search-tree-lowest-common-ancestor/problem) - 30.0 - python3
```python
# Enter your code here. Read input from STDIN. Print output to STDOUT
'''
class Node:
      def __init__(self,info): 
          self.info = info  
          self.left = None  
          self.right = None 
           

       // this is a node of the tree , which contains info as data, left , right
'''

def lca(root, v1, v2):
    if min(v1,v2)<=root.info<=max(v1,v2): return root
    return lca((root.right,root.left)[root.info>v1],v1,v2)
```
# [Queues: A Tale of Two Stacks](https://www.hackerrank.com/challenges/ctci-queue-using-two-stacks/problem) - 30.0 - pypy3
```python
from collections import deque; L = deque()
for _ in range(int(input())):
    Q = input().split()
    if Q[0]=='1': L.append(Q[1])
    elif Q[0]=='2': L.popleft()
    else: print(L[0])
```
# [Find Digits](https://www.hackerrank.com/challenges/find-digits/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the findDigits function below.
def findDigits(n):
    return sum(1 for d in str(n) if int(d) and n%int(d)==0)


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        n = int(input())

        result = findDigits(n)

        fptr.write(str(result) + '\n')

    fptr.close()
```
# [Cut the sticks](https://www.hackerrank.com/challenges/cut-the-sticks/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the cutTheSticks function below.
def cutTheSticks(a):
    result = []
    while(a):
        result += [len(a)]
        s = min(a)
        a = [e-s for e in a if e != s]
    return result

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    result = cutTheSticks(arr)

    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [The Time in Words](https://www.hackerrank.com/challenges/the-time-in-words/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the timeInWords function below.

numbers = ['one','two','three','four','five','six','seven','eight',
            'nine','ten','eleven','twelve','thirteen','fourteen',
            'fifteen','sixteen','seventeen','eighteen','nineteen',
            'twenty']
numbers += ['twenty ' + numbers[i] for i in range(9)]

def timeInWords(h, m):
    s = '' if m==59 or m==1 else 's'
    if m == 0: return numbers[h-1] + " o' clock"
    elif m > 30:
        minutes = numbers[60-m-1] + " minute" + s if m != 45 else "quarter"
        return minutes + " to " + numbers[0 if h==12 else h]
    else:
        if m % 15 != 0: minutes = (numbers[m-1] + " minute" + s)
        else: minutes =  "quarter" if m == 15 else "half"
        return minutes + " past " + numbers[h-1]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    h = int(input())

    m = int(input())

    result = timeInWords(h, m)

    fptr.write(result + '\n')

    fptr.close()
```
# [Anagram](https://www.hackerrank.com/challenges/anagram/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the anagram function below.
def anagram(s):
    if len(s)%2!=0:return -1
    l=len(s)//2
    s1,s2=s[:l],s[l:]
    return sum([abs(s1.count(c)-s2.count(c))for c in[*map(chr,range(97,123))]])//2

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(input())
    for q_itr in range(q):
        s = input()
        result = anagram(s)
        fptr.write(str(result) + '\n')
    fptr.close()
```
# [Lisa's Workbook](https://www.hackerrank.com/challenges/lisa-workbook/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the workbook function below.
def workbook(n, k, A):
    special = 0
    page = 1
    for c in range(n):
        n_problems = A[c]
        for c_pages in range(math.ceil(n_problems/k)):
            if page in range(c_pages*k+1,min((c_pages+1)*k+1,n_problems+1)):
                special += 1
            page += 1
    return special


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = int(nk[0])
    k = int(nk[1])
    arr = list(map(int, input().rstrip().split()))
    result = workbook(n, k, arr)
    fptr.write(str(result) + '\n')
    fptr.close()
```
# [Chocolate Feast ](https://www.hackerrank.com/challenges/chocolate-feast/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the chocolateFeast function below.
def chocolateFeast(n, c, m):
    bars = wrappers = n//c
    while wrappers >= m:
        new_bars = wrappers//m
        bars += new_bars
        wrappers = wrappers%m + new_bars
    return bars

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        ncm = input().split()

        n = int(ncm[0])

        c = int(ncm[1])

        m = int(ncm[2])

        result = chocolateFeast(n, c, m)

        fptr.write(str(result) + '\n')

    fptr.close()
```
# [String Construction ](https://www.hackerrank.com/challenges/string-construction/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the stringConstruction function below.
def stringConstruction(s):
    return len({*s})

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        s = input()

        result = stringConstruction(s)

        fptr.write(str(result) + '\n')

    fptr.close()
```
# [Funny String](https://www.hackerrank.com/challenges/funny-string/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the funnyString function below.
def funnyString(s):
    s = [abs(ord(s[i])-ord(s[i+1])) for i in range(len(s)-1)]
    return ('Not Funny','Funny')[s==s[::-1]]


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        s = input()

        result = funnyString(s)

        fptr.write(result + '\n')

    fptr.close()
```
# [XOR Strings](https://www.hackerrank.com/challenges/strings-xor/problem) - 25.0 - python3
```python
def strings_xor(s, t):
    res = ""
    for i in range(len(s)):
        if s[i] == t[i]:
            res += '0';
        else:
            res += '1';

    return res

s = input()
t = input()
print(strings_xor(s, t))
```
# [Taum and B'day](https://www.hackerrank.com/challenges/taum-and-bday/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'taumBday' function below.
#
# The function is expected to return a LONG_INTEGER.
# The function accepts following parameters:
#  1. INTEGER b
#  2. INTEGER w
#  3. INTEGER bc
#  4. INTEGER wc
#  5. INTEGER z
#

def taumBday(b, w, bc, wc, z):
    wc = min(wc,bc+z)
    bc = min(bc,wc+z)
    return b * bc + w * wc

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input().strip())

    for t_itr in range(t):
        first_multiple_input = input().rstrip().split()

        b = int(first_multiple_input[0])

        w = int(first_multiple_input[1])

        second_multiple_input = input().rstrip().split()

        bc = int(second_multiple_input[0])

        wc = int(second_multiple_input[1])

        z = int(second_multiple_input[2])

        result = taumBday(b, w, bc, wc, z)

        fptr.write(str(result) + '\n')

    fptr.close()
```
# [Fair Rations](https://www.hackerrank.com/challenges/fair-rations/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the fairRations function below.
def fairRations(B):
    l, loaves = len(B), 0
    for i in range(l):
        if i == l - 1: return (loaves,'NO')[B[i]%2]
        if B[i]%2:
            B[i] += 1
            B[i+1] += 1
            loaves += 2

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    N = int(input())

    B = list(map(int, input().rstrip().split()))

    result = fairRations(B)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Smart Number](https://www.hackerrank.com/challenges/smart-number/problem) - 25.0 - python3
```python
import math

def is_smart_number(num):
    val = int(math.sqrt(num))
    if num / val == val:
        return True
    return False

for _ in range(int(input())):
    num = int(input())
    ans = is_smart_number(num)
    if ans:
        print("YES")
    else:
        print("NO")
```
# [Two Strings](https://www.hackerrank.com/challenges/two-strings/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the twoStrings function below.
def twoStrings(s1, s2):return ('NO','YES')[[*{*s1}&{*s2}]!=[]]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        s1 = input()

        s2 = input()

        result = twoStrings(s1, s2)

        fptr.write(result + '\n')

    fptr.close()
```
# [ACM ICPC Team](https://www.hackerrank.com/challenges/acm-icpc-team/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

from itertools import combinations

# Complete the acmTeam function below.
def acmTeam(T):
    maximum = teams = 0
    for p1,p2 in combinations(T,2):
        test = bin(int(p1,2) | int(p2,2)).count('1')
        if test == maximum: teams += 1
        elif test > maximum: maximum, teams = test, 1
    return [maximum, teams]


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    topic = []
    for _ in range(n):
        topic_item = input()
        topic.append(topic_item)
    result = acmTeam(topic)
    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')
    fptr.close()
```
# [Sparse Arrays](https://www.hackerrank.com/challenges/sparse-arrays/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the matchingStrings function below.
def matchingStrings(S, Q): return (S.count(q) for q in Q)
        


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    strings_count = int(input())

    strings = []

    for _ in range(strings_count):
        strings_item = input()
        strings.append(strings_item)

    queries_count = int(input())

    queries = []

    for _ in range(queries_count):
        queries_item = input()
        queries.append(queries_item)

    res = matchingStrings(strings, queries)

    fptr.write('\n'.join(map(str, res)))
    fptr.write('\n')

    fptr.close()
```
# [Sum vs XOR](https://www.hackerrank.com/challenges/sum-vs-xor/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

def sumXor(n):
    return (2**bin(n)[2:].count('0'),'1')[n<1]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = sumXor(n)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Palindrome Index](https://www.hackerrank.com/challenges/palindrome-index/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the palindromeIndex function below.
def palindromeIndex(s):
    for i in range(len(s)//2):
        if s[i] != s[-1-i]:
            return (i,len(s)-1-i)[s[i:i+2]==s[-2-i:-4-i:-1]]
    return -1

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        s = input()

        result = palindromeIndex(s)

        fptr.write(str(result) + '\n')

    fptr.close()
```
# [Flatland Space Stations](https://www.hackerrank.com/challenges/flatland-space-stations/problem) - 25.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the flatlandSpaceStations function below.
def flatlandSpaceStations(n, C):
    C.sort()
    return max(C[0],n-1-C[-1],*[(C[i+1]-C[i])//2 for i in range(len(C)-1)])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    c = list(map(int, input().rstrip().split()))

    result = flatlandSpaceStations(n, c)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Strings: Making Anagrams](https://www.hackerrank.com/challenges/ctci-making-anagrams/problem) - 25.0 - python3
```python
a,b = input(),input()
print(sum(abs(a.count(c)-b.count(c)) for c in (chr(n) for n in range(97,123))))
```
# [Hash Tables: Ransom Note](https://www.hackerrank.com/challenges/ctci-ransom-note/problem) - 25.0 - python3
```python
from collections import Counter
_,M,N = (Counter(input().split()) for _ in range(3))
print('No' if any(N[w] > M[w] for w in N.keys()) else 'Yes')
```
# [QHEAP1](https://www.hackerrank.com/challenges/qheap1/problem) - 25.0 - python3
```python
class heap:
    def __init__(self):
        self.array=[None]
        self.lenght=0  # points to the index of our last element 
    
    def bubble_down(self,i):
        j=i//2+0
        if j<1:
            return
        if self.array[j]>self.array[i]:
            self.array[i],self.array[j]=self.array[j],self.array[i]
            self.bubble_down(j)
        else:
            return
    def insert(self,val):
        self.array.append(val)
        self.lenght+=1
        self.bubble_down(self.lenght)

    def bubble_up(self,i):
        j=i+0
        if 2*i>self.lenght:
            return
        if self.array[i]>self.array[2*i]:
            j=2*i+0
        if 2*i+1<=self.lenght and self.array[j]>self.array[2*i+1]:
            j=2*i+1
        if i==j:
            return
        self.array[j],self.array[i]=self.array[i],self.array[j]
        self.bubble_up(j)

    def delete(self,x):
        i=self.array.index(x)
        self.array[-1],self.array[i]=self.array[i],self.array[-1]        
        self.array.pop()
        self.lenght-=1
        self.bubble_up(i)

my_heap=heap()
for _ in range(int(input())):
    a=input().split()
    if a[0]=='1':
        my_heap.insert(int(a[1]))
    elif a[0]=='2':
        my_heap.delete(int(a[1]))
    else:
        print(my_heap.array[1])
```
# [Jesse and Cookies](https://www.hackerrank.com/challenges/jesse-and-cookies/problem) - 25.0 - python3
```python
import heapq as hq
k,H,i = int(input().split()[1]),[*map(int,input().split())],0
hq.heapify(H)
while H[0] < k:
    if len(H) == 1: print(-1); quit()
    a,b = hq.heappop(H),hq.heappop(H)
    hq.heappush(H,a+b*2); i += 1
print(i)
```
# [Equal Stacks](https://www.hackerrank.com/challenges/equal-stacks/problem) - 25.0 - python3
```python
input(); S = [[*map(int,input().split())][::-1] for _ in range(3)]
H = [sum(s) for s in S]
while len(set(H)) != 1:
    i = H.index(max(H))
    H[i] -= S[i].pop()
print(sum(S[0]))
```
# [Balanced Brackets](https://www.hackerrank.com/challenges/balanced-brackets/problem) - 25.0 - python3
```python
M = {'{':'}','[':']','(':')'}
for _ in range(int(input())):
    B = []
    for c in input():
        if c in M.keys(): B.append(c)
        elif not B or M[B.pop()]!=c: print("NO"); break
    else: print("YES" if not B else "NO")
```
# [Lonely Integer](https://www.hackerrank.com/challenges/lonely-integer/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the lonelyinteger function below.
def lonelyinteger(a):
    for e in a:
        if a.count(e) == 1:
            return e

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    a = list(map(int, input().rstrip().split()))

    result = lonelyinteger(a)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Two Characters](https://www.hackerrank.com/challenges/two-characters/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys
from collections import Counter

# Complete the alternate function below.
def alternate(s):
    mc, length = Counter(s).most_common(), 0
    for i in range(len(mc) - 1):
        j = i+1
        while j < len(mc) and mc[i][1]-mc[j][1] < 2:
            a = re.sub('[^'+mc[i][0]+mc[j][0]+']+','',s)
            if len(a) < length: return length
            if re.match('^'+mc[j][0]+
                        '?('+mc[i][0]+
                        mc[j][0]+')*'+
                        mc[i][0]+'?$',a):
                length = len(a)
            j+=1
    return length



if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    l = int(input().strip())

    s = input()

    result = alternate(s)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [HackerRank in a String!](https://www.hackerrank.com/challenges/hackerrank-in-a-string/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the hackerrankInString function below.
def hackerrankInString(s):
    if re.match('.*h.*a.*c.*k.*e.*r.*r.*a.*n.*k.*',s):
        return 'YES'
    else: return 'NO'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        s = input()

        result = hackerrankInString(s)

        fptr.write(result + '\n')

    fptr.close()
```
# [Pangrams](https://www.hackerrank.com/challenges/pangrams/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys
import string

# Complete the pangrams function below.
def pangrams(s):
    for c in string.ascii_lowercase:
        if c not in s.lower():
            return 'not pangram'
    return 'pangram'


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = pangrams(s)

    fptr.write(result + '\n')

    fptr.close()
```
# [Weighted Uniform Strings](https://www.hackerrank.com/challenges/weighted-uniform-string/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys
from string import ascii_lowercase

# Complete the weightedUniformStrings function below.
def weightedUniformStrings(s, queries):
    results, c_max = [], {}
    for c in ascii_lowercase:
        if c in s:
            c_max[c] = len(max(re.findall(c+'+',s),key=len))
        else: c_max[c] = 0

    for q in queries:
        found = False
        for i in range(1,27):
            if q % i == 0 and  i * c_max[chr(96+i)] >= q:
                found = True
                break
        if found: results.append('Yes')
        else: results.append('No')
    return results

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    queries_count = int(input())

    queries = []

    for _ in range(queries_count):
        queries_item = int(input())
        queries.append(queries_item)

    result = weightedUniformStrings(s, queries)

    fptr.write('\n'.join(result))
    fptr.write('\n')

    fptr.close()
```
# [Picking Numbers](https://www.hackerrank.com/challenges/picking-numbers/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'pickingNumbers' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY a as parameter.
#

def pickingNumbers(a):
    m = 0
    for i in range(min(a),max(a)+1):
        l = a.count(i) + a.count(i+1)
        if l > m: m = l
    return m


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    a = list(map(int, input().rstrip().split()))

    result = pickingNumbers(a)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Designer PDF Viewer](https://www.hackerrank.com/challenges/designer-pdf-viewer/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the designerPdfViewer function below.
def designerPdfViewer(h, word):
    return len(word)*max([h[ord(c)-97] for c in word])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    h = list(map(int, input().rstrip().split()))

    word = input()

    result = designerPdfViewer(h, word)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Utopian Tree](https://www.hackerrank.com/challenges/utopian-tree/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the utopianTree function below.
def utopianTree(n):
    h = 1
    for i in range(n):
        if i%2==0: h*=2
        else: h+=1
    return h

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        n = int(input())

        result = utopianTree(n)

        fptr.write(str(result) + '\n')

    fptr.close()
```
# [Angry Professor](https://www.hackerrank.com/challenges/angry-professor/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the angryProfessor function below.
'''
k: the threshold number of students
a: an array of integers representing arrival times
'''
def angryProfessor(k, a):
    if sum(1 for t in a if t <= 0) < k:
        return 'YES'
    return 'NO'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        nk = input().split()

        n = int(nk[0])

        k = int(nk[1])

        a = list(map(int, input().rstrip().split()))

        result = angryProfessor(k, a)

        fptr.write(result + '\n')

    fptr.close()
```
# [Circular Array Rotation](https://www.hackerrank.com/challenges/circular-array-rotation/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the circularArrayRotation function below.
'''
a: an array of integers to rotate
k: an integer, the rotation count
queries: an array of integers, the indices to report
'''
def circularArrayRotation(a, k, queries):
    return [a[q-k%len(a)] for q in queries]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nkq = input().split()

    n = int(nkq[0])

    k = int(nkq[1])

    q = int(nkq[2])

    a = list(map(int, input().rstrip().split()))

    queries = []

    for _ in range(q):
        queries_item = int(input())
        queries.append(queries_item)

    result = circularArrayRotation(a, k, queries)

    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [Sequence Equation](https://www.hackerrank.com/challenges/permutation-equation/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the permutationEquation function below.
def permutationEquation(p):
    return [p.index(p.index(i)+1)+1 for i in range(1,max(p)+1)]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    p = list(map(int, input().rstrip().split()))

    result = permutationEquation(p)

    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [Forming a Magic Square](https://www.hackerrank.com/challenges/magic-square-forming/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys
from itertools import product

# Complete the formingMagicSquare function below.
rots = []
for m in ([[8,1,6],[3,5,7],[4,9,2]],
          [[8,3,4],[1,5,9],[6,7,2]]):
    rots += [m,[r[::-1] for r in m],m[::-1],
             [r[::-1] for r in m[::-1]]]

def formingMagicSquare(s):
    diffs = [0]*8
    for i in range(8):
        for x, y in product(range(3), range(3)):
            diffs[i] += abs(rots[i][y][x] - s[y][x])
    print(diffs)
    return min(diffs)
    
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = []

    for _ in range(3):
        s.append(list(map(int, input().rstrip().split())))

    result = formingMagicSquare(s)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Extra Long Factorials](https://www.hackerrank.com/challenges/extra-long-factorials/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys
from functools import reduce 

# Complete the extraLongFactorials function below.
def extraLongFactorials(n):
    print(reduce(lambda x,y: x*y, range(1,n+1)))

if __name__ == '__main__':
    n = int(input())

    extraLongFactorials(n)
```
# [Repeated String](https://www.hackerrank.com/challenges/repeated-string/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the repeatedString function below.
def repeatedString(s, n):
    l = len(s)
    factor = n//l
    return s.count('a')*factor + s[:n-l*factor].count('a')

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    n = int(input())

    result = repeatedString(s, n)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Print Function](https://www.hackerrank.com/challenges/python-print/problem) - 20.0 - python3
```python
if __name__ == '__main__':
    print(''.join([str(i) for i in range(1,int(input())+1)]))
```
# [Climbing the Leaderboard](https://www.hackerrank.com/challenges/climbing-the-leaderboard/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys
from bisect import bisect

# Complete the climbingLeaderboard function below.
def climbingLeaderboard(scores, alice):
    scores=sorted([*{*scores}])
    return [len(scores)-bisect(scores,A)+1for A in alice]


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    scores_count = int(input())
    scores = list(map(int, input().rstrip().split()))
    alice_count = int(input())
    alice = list(map(int, input().rstrip().split()))
    result = climbingLeaderboard(scores, alice)
    
    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')
    fptr.close()
```
# [Equalize the Array](https://www.hackerrank.com/challenges/equality-in-a-array/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the equalizeArray function below.
def equalizeArray(A):
    return len(A)-max([A.count(n)for n in A])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    result = equalizeArray(arr)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Alternating Characters ](https://www.hackerrank.com/challenges/alternating-characters/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the alternatingCharacters function below.
def alternatingCharacters(s):
    a,d='',0
    for c in s:
        if a!=c: a=c
        else: d+=1
    return d

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        s = input()

        result = alternatingCharacters(s)

        fptr.write(str(result) + '\n')

    fptr.close()
```
# [Beautiful Binary String](https://www.hackerrank.com/challenges/beautiful-binary-string/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the beautifulBinaryString function below.
def beautifulBinaryString(b):
    c = 0
    while '010' in b:
        i = b.index('010')+2
        b = [*b]
        b[i]='1'
        b = ''.join(b)
        c += 1
    return c

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    b = input()

    result = beautifulBinaryString(b)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Service Lane](https://www.hackerrank.com/challenges/service-lane/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the serviceLane function below.
def serviceLane(n, cases, widths):
    return [min([widths[i]for i in range(c[0],c[1]+1)])for c in cases]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nt = input().split()
    n = int(nt[0])
    t = int(nt[1])
    widths = list(map(int, input().rstrip().split()))
    cases = []
    for _ in range(t):
        cases.append(list(map(int, input().rstrip().split())))
    result = serviceLane(n, cases, widths)
    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')
    fptr.close()
```
# [Minimum Distances](https://www.hackerrank.com/challenges/minimum-distances/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the minimumDistances function below.
def minimumDistances(A):
    min = -1
    for i in range(len(A)):
        if A[i] in A[i+1:]:
            t=A.index(A[i],i+1)-i
            if min==-1 or t < min:
                min = t
    return min


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    a = list(map(int, input().rstrip().split()))
    result = minimumDistances(a)
    fptr.write(str(result) + '\n')
    fptr.close()
```
# [Beautiful Triplets](https://www.hackerrank.com/challenges/beautiful-triplets/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the beautifulTriplets function below.
def beautifulTriplets(d, A):
    beautiful = 0
    for j in range(len(A)):
        if A[j]+d in A and A[j]-d in A:
            beautiful += math.ceil((A.count(A[j]+d)+A.count(A[j]-d))/2)
    return beautiful

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nd = input().split()

    n = int(nd[0])

    d = int(nd[1])

    arr = list(map(int, input().rstrip().split()))

    result = beautifulTriplets(d, arr)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Jumping on the Clouds](https://www.hackerrank.com/challenges/jumping-on-the-clouds/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the jumpingOnClouds function below.
def jumpingOnClouds(c):
    i = j = 0
    while i!=len(c)-1:
        i += (1,2)[i+2<len(c) and c[i+2]==0]
        j += 1
    return j

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    c = list(map(int, input().rstrip().split()))

    result = jumpingOnClouds(c)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Maximum Perimeter Triangle](https://www.hackerrank.com/challenges/maximum-perimeter-triangle/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

from itertools import combinations

# Complete the maximumPerimeterTriangle function below.
def maximumPerimeterTriangle(sticks):
    sticks.sort(reverse=True)
    for a,b,c in combinations(sticks,3):
        if a<b+c: return (c,b,a)
    return [-1]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    sticks = list(map(int, input().rstrip().split()))

    result = maximumPerimeterTriangle(sticks)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [Luck Balance](https://www.hackerrank.com/challenges/luck-balance/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the luckBalance function below.
def luckBalance(k, contests):
    split = [[],[]]
    for c in contests:
        if c[1]: split[1].append(c[0])
        else:    split[0].append(c[0])
    split[1].sort(reverse=True)
    return sum(split[0]) + sum(split[1][:k]) - sum(split[1][k:])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = int(nk[0])

    k = int(nk[1])

    contests = []

    for _ in range(n):
        contests.append(list(map(int, input().rstrip().split())))

    result = luckBalance(k, contests)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [The Love-Letter Mystery](https://www.hackerrank.com/challenges/the-love-letter-mystery/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the theLoveLetterMystery function below.
def theLoveLetterMystery(s):
    l = len(s)
    return sum([abs(ord(s[i])-ord(s[l-1-i])) for i in range(len(s)//2)])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        s = input()

        result = theLoveLetterMystery(s)

        fptr.write(str(result) + '\n')

    fptr.close()
```
# [Gemstones](https://www.hackerrank.com/challenges/gem-stones/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the gemstones function below.
def gemstones(A):
    gems = {*A[0]}
    for i in range(1,len(A)):
        gems &= {*A[i]}
    return len(gems)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = []

    for _ in range(n):
        arr_item = input()
        arr.append(arr_item)

    result = gemstones(arr)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Poker Nim](https://www.hackerrank.com/challenges/poker-nim-1/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the pokerNim function below.
from functools import reduce
def pokerNim(k, c):
    return ('First','Second')[reduce((lambda a,b:a^b),c)==0]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        nk = input().split()
        n = int(nk[0])
        k = int(nk[1])
        c = list(map(int, input().rstrip().split()))
        result = pokerNim(k, c)
        fptr.write(result + '\n')
    fptr.close()
```
# [Grid Challenge](https://www.hackerrank.com/challenges/grid-challenge/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the gridChallenge function below.
def gridChallenge(G):
    for i in range(len(G)):G[i] = sorted(G[i])
    for i in range(len(G[0])):
        t = [g[i] for g in G]
        if t != sorted(t): return 'NO'
    return 'YES'


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        n = int(input())
        grid = []
        for _ in range(n):
            grid_item = input()
            grid.append(grid_item)
        result = gridChallenge(grid)
        fptr.write(result + '\n')
    fptr.close()
```
# [Separate the Numbers](https://www.hackerrank.com/challenges/separate-the-numbers/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the separateNumbers function below.
def separateNumbers(s):
    l = len(s)
    if s[0] != '0':
        for i in range(1,l//2+1):
            test = s[:i]
            first = int(test)
            while len(test) <= l:
                first += 1
                test += str(first)
                if test == s: return 'YES ' + s[:i]
    return 'NO'

if __name__ == '__main__':
    q = int(input())
    for q_itr in range(q):
        s = input()
        print(separateNumbers(s))
```
# [Halloween Sale](https://www.hackerrank.com/challenges/halloween-sale/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the howManyGames function below.
def howManyGames(p, d, m, s):
    n = 0
    while True:
        s -= max(m,p-d*n)
        if s < 0: return n
        n += 1

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    pdms = input().split()

    p = int(pdms[0])

    d = int(pdms[1])

    m = int(pdms[2])

    s = int(pdms[3])

    answer = howManyGames(p, d, m, s)

    fptr.write(str(answer) + '\n')

    fptr.close()
```
# [Sherlock and Squares](https://www.hackerrank.com/challenges/sherlock-and-squares/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the squares function below.
def squares(a, b):
    return len([n**2 for n in range(math.ceil(math.sqrt(a)),
                                    math.floor(math.sqrt(b))+1)])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        ab = input().split()

        a = int(ab[0])

        b = int(ab[1])

        result = squares(a, b)

        fptr.write(str(result) + '\n')

    fptr.close()
```
# [Left Rotation](https://www.hackerrank.com/challenges/array-left-rotation/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nd = input().split()

    n = int(nd[0])

    d = int(nd[1])

    a = list(map(int, input().rstrip().split()))

    print(*a[d:]+a[:d])
```
# [Big Sorting](https://www.hackerrank.com/challenges/big-sorting/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the bigSorting function below.
def bigSorting(unsorted):
    return sorted(unsorted,key=lambda e:(len(e),e))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    unsorted = []

    for _ in range(n):
        unsorted_item = input()
        unsorted.append(unsorted_item)

    result = bigSorting(unsorted)

    fptr.write('\n'.join(result))
    fptr.write('\n')

    fptr.close()
```
# [Nimble Game](https://www.hackerrank.com/challenges/nimble-game-1/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

from functools import reduce

# Complete the nimbleGame function below.
def nimbleGame(S):
    S = [(0,i)[s[i]%2] for i in range(1,len(S))]
    return ('Second','First')[reduce(lambda a,b:a^b,S,0)!=0]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        n = int(input())

        s = list(map(int, input().rstrip().split()))

        result = nimbleGame(s)

        fptr.write(str(result) + '\n')

    fptr.close()
```
# [Misère Nim](https://www.hackerrank.com/challenges/misere-nim-1/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

from functools import reduce

# Complete the misereNim function below.
def misereNim(S):
    if all(n==1 for n in S): return ('First','Second')[len(S)%2]
    return ('First','Second')[reduce(lambda a,b:a^b,S)==0]

    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        n = int(input())

        s = list(map(int, input().rstrip().split()))

        result = misereNim(s)

        fptr.write(result + '\n')

    fptr.close()
```
# [Append and Delete](https://www.hackerrank.com/challenges/append-and-delete/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the appendAndDelete function below.
def appendAndDelete(s, t, k):
    i, ls, lt = 0, len(s), len(t)
    if ls+lt<=k: return 'Yes'
    m = min(ls, lt)
    while i < m and s[i] == t[i]: i += 1
    return ('No','Yes')[ls+lt-i*2<=k and (k-ls-lt+i*2)%2==0]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    t = input()

    k = int(input())

    result = appendAndDelete(s, t, k)

    fptr.write(result + '\n')

    fptr.close()
```
# [Compress the String! ](https://www.hackerrank.com/challenges/compress-the-string/problem) - 20.0 - python3
```python
from itertools import groupby
print(*[(len([*e[1]]),int(e[0])) for e in groupby(input())])
```
# [Alphabet Rangoli](https://www.hackerrank.com/challenges/alphabet-rangoli/problem) - 20.0 - python3
```python
def print_rangoli(s):
    C,w = [chr(96+s-i) for i in range(s)],1+(s-1)*4
    P = ['-'.join(C[:i+1]+C[:i][::-1]).center(w,'-') for i in range(s)]
    print('\n'.join(P+P[::-1][1:]))
```
# [Capitalize!](https://www.hackerrank.com/challenges/capitalize/problem) - 20.0 - python3
```python
import string 

# Complete the solve function below.
def solve(s):
    for x in s.split():
        s = s.replace(x, x.capitalize())
    return s
```
# [Collections.namedtuple()](https://www.hackerrank.com/challenges/py-collections-namedtuple/problem) - 20.0 - python3
```python
n,i = int(input()),input().split().index('MARKS')
print(sum(int(input().split()[i]) for _ in range(n))/n)
```
# [DefaultDict Tutorial](https://www.hackerrank.com/challenges/defaultdict-tutorial/problem) - 20.0 - python3
```python
from collections import defaultdict
D,n,m = defaultdict(list),*map(int,input().split())
for i in range(n): D[input()].append(i+1) 
for e in (input() for _ in range(m)): 
    print(" ".join(map(str,D[e])) if e in D else -1)
```
# [Collections.OrderedDict()](https://www.hackerrank.com/challenges/py-collections-ordereddict/problem) - 20.0 - python3
```python
from collections import OrderedDict
D = OrderedDict()
for _ in range(int(input())):
    s = input()
    n,p = s[:s.rindex(' ')],s[s.rindex(' ')+1:]
    if n not in D: D[n] = 0
    D[n] += int(p)
for n,p in D.items():
    print(n,p)
```
# [Incorrect Regex](https://www.hackerrank.com/challenges/incorrect-regex/problem) - 20.0 - python3
```python
import re
for _ in range(int(input())):
    try: re.compile(input()); print(True)
    except: print(False)
```
# [Re.split()](https://www.hackerrank.com/challenges/re-split/problem) - 20.0 - python3
```python
regex_pattern = r"[.|,]"
```
# [Map and Lambda Function](https://www.hackerrank.com/challenges/map-and-lambda-expression/problem) - 20.0 - python3
```python
cube = lambda x: x**3

def fibonacci(n):
    L = [0,1]
    for _ in range(n-2):
        L += [L[-1]+L[-2]]
    return L[:n]
```
# [Detect Floating Point Number](https://www.hackerrank.com/challenges/introduction-to-regex/problem) - 20.0 - python3
```python
import re
for _ in range(int(input())):
    print(bool(re.match('[-|+]?\d*\.\d*$',input())))
```
# [Any or All](https://www.hackerrank.com/challenges/any-or-all/problem) - 20.0 - python3
```python
input();N=input().split()
print(any(n==n[::-1] for n in N) and all(int(n)>0 for n in N))
```
# [Python Evaluation](https://www.hackerrank.com/challenges/python-eval/problem) - 20.0 - python3
```python
eval(input())
```
# [Input()](https://www.hackerrank.com/challenges/input/problem) - 20.0 - python3
```python
x,k = map(int,input().split())
print(k==eval(input()))
```
# [Collections.deque()](https://www.hackerrank.com/challenges/py-collections-deque/problem) - 20.0 - python3
```python
from collections import deque
D=deque()
for _ in range(int(input())):
    eval('D.{}({})'.format(*input().split()+['']))
print(*D)
```
# [Min and Max](https://www.hackerrank.com/challenges/np-min-and-max/problem) - 20.0 - python3
```python
import numpy
print(max(numpy.min(eval('[*map(int,input().split())],'*int(input()[0])),1)))
```
# [Sum and Prod](https://www.hackerrank.com/challenges/np-sum-and-prod/problem) - 20.0 - python3
```python
import numpy as N
print(N.prod(N.sum(eval('[*map(int,input().split())],'*int(input()[0])),0)))
```
# [Polynomials](https://www.hackerrank.com/challenges/np-polynomials/problem) - 20.0 - python3
```python
import numpy
print(numpy.polyval([*map(float,input().split())],int(input())))
```
# [Inner and Outer](https://www.hackerrank.com/challenges/np-inner-and-outer/problem) - 20.0 - python3
```python
import numpy as N
A,B=eval('N.array(input().split(),int),'*2)
print(N.inner(A,B))
print(N.outer(A,B))
```
# [Dot and Cross](https://www.hackerrank.com/challenges/np-dot-and-cross/problem) - 20.0 - python3
```python
import numpy as N
print(N.dot(*eval("N.array(eval('input().split(),'*%s),int),"%input()*2)))
```
# [Mean, Var, and Std](https://www.hackerrank.com/challenges/np-mean-var-and-std/problem) - 20.0 - python3
```python
import numpy as N; N.set_printoptions(legacy='1.13')
A=N.array(eval('input().split(),'*int(input().split()[0])),int)
print(N.mean(A,1),N.var(A,0),N.std(A),sep='\n')
```
# [Class 2 - Find the Torsional Angle](https://www.hackerrank.com/challenges/class-2-find-the-torsional-angle/problem) - 20.0 - python3
```python
class Points(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __iter__(self):
        return iter([self.x,self.y,self.z])

    def __sub__(self, no):
        return Points(*(a-b for a,b in zip(self,no)))

    def dot(self, no):
        return sum(a*b for a,b in zip(self,no))

    def cross(self, no):
        return Points(self.y*no.z-self.z*no.y,
                      self.z*no.x-self.x*no.z,
                      self.x*no.y-self.y*no.x)

    def absolute(self):
        return sum(n**2 for n in self)**.5
```
# [Validating phone numbers](https://www.hackerrank.com/challenges/validating-the-phone-number/problem) - 20.0 - python3
```python
import re
print(*eval("('NO','YES')[re.match('[789]\d{9}$',input())!=None],"*int(input())),sep='\n')
```
# [Validating and Parsing Email Addresses](https://www.hackerrank.com/challenges/validating-named-email-addresses/problem) - 20.0 - python3
```python
import re
for _ in range(int(input())):
    A=input()
    if re.match('<[a-zA-Z](\w|[.-])+@[a-zA-Z]+\.[a-zA-Z]{1,3}>$',A.split()[1]):
        print(A)
```
# [Group(), Groups() & Groupdict()](https://www.hackerrank.com/challenges/re-group-groups/problem) - 20.0 - python3
```python
import re
m=re.search(r'([a-zA-Z0-9])\1',input())
print(m.group(1) if m else -1)
```
# [Re.findall() & Re.finditer()](https://www.hackerrank.com/challenges/re-findall-re-finditer/problem) - 20.0 - python3
```python
import re
c='[qwrtypsdfghjklzxcvbnm]'
m=re.findall('(?<='+c+')([aeiou]{2,})'+c,input(),re.I)
print('\n'.join(m or['-1']))
```
# [Re.start() & Re.end()](https://www.hackerrank.com/challenges/re-start-re-end/problem) - 20.0 - python3
```python
import re;S,k=input(),input()
M=[(m.start(1),m.end(1)-1)for m in re.finditer("(?=(%s))"%k,S)]
print(*M if M else [(-1,-1)],sep='\n')
```
# [Arrays](https://www.hackerrank.com/challenges/np-arrays/problem) - 20.0 - python3
```python
def arrays(A):return numpy.array(A,float)[::-1]
```
# [Shape and Reshape](https://www.hackerrank.com/challenges/np-shape-reshape/problem) - 20.0 - python3
```python
import numpy as np
print(np.reshape(np.array(input().split(),int),(3,3)))
```
# [Transpose and Flatten](https://www.hackerrank.com/challenges/np-transpose-and-flatten/problem) - 20.0 - python3
```python
import numpy as N
A=N.array(eval('input().split(),'*int(input().split()[0])),int)
print(N.transpose(A),A.flatten(),sep='\n')
```
# [Zeros and Ones](https://www.hackerrank.com/challenges/np-zeros-and-ones/problem) - 20.0 - python3
```python
import numpy as N
I=[*map(int, input().split())]
print(N.zeros(I,int))
print(N.ones(I,int))
```
# [Eye and Identity](https://www.hackerrank.com/challenges/np-eye-and-identity/problem) - 20.0 - python3
```python
import numpy as N
N.set_printoptions(legacy='1.13')
print(N.eye(*map(int,input().split())))
```
# [Array Mathematics](https://www.hackerrank.com/challenges/np-array-mathematics/problem) - 20.0 - python3
```python
import numpy as N
A,B=eval("N.array(eval('input().split(),'*%s),int),"%input()[0]*2)
print(A+B,A-B,A*B,A//B,A%B,A**B,sep='\n')
```
# [Floor, Ceil and Rint](https://www.hackerrank.com/challenges/floor-ceil-and-rint/problem) - 20.0 - python3
```python
import numpy as np
np.set_printoptions(sign=' ')
A=[*map(float,input().split())]
print(np.floor(A),np.ceil(A),np.rint(A),sep='\n')
```
# [Linear Algebra](https://www.hackerrank.com/challenges/np-linear-algebra/problem) - 20.0 - python3
```python
import numpy
A=[[*map(float,input().split())] for _ in range(int(input()))]
print(round(numpy.linalg.det(A),2))
```
# [Triangle Quest 2](https://www.hackerrank.com/challenges/triangle-quest-2/problem) - 20.0 - python3
```python
for i in range(1,int(input())+1):
    print((10**i-1)**2//81)
```
# [XML 1 - Find the Score](https://www.hackerrank.com/challenges/xml-1-find-the-score/problem) - 20.0 - python3
```python
def get_attr_number(N):
    return sum(len(e.items()) for e in N.iter())
```
# [XML2 - Find the Maximum Depth](https://www.hackerrank.com/challenges/xml2-find-the-maximum-depth/problem) - 20.0 - python3
```python
maxdepth = -1
def depth(E, l):
    global maxdepth
    if l==maxdepth: maxdepth+=1
    for e in E: depth(e,l+1)
```
# [Concatenate](https://www.hackerrank.com/challenges/np-concatenate/problem) - 20.0 - python3
```python
import numpy as np
n,m,_=map(int,input().split())
print(np.array([[*map(int,input().split())] for _ in range(n+m)]))
```
# [Triangle Quest](https://www.hackerrank.com/challenges/python-quest-1/problem) - 20.0 - python3
```python
for i in range(1,int(input())): #More than 2 lines will result in 0 score. Do not leave a blank line also
    print(i*(10**i-1)//9)
```
# [Regex Substitution](https://www.hackerrank.com/challenges/re-sub-regex-substitution/problem) - 20.0 - python3
```python
import re
for _ in range(int(input())):
    print(re.sub('(?<= )\&\&(?= )','and',re.sub('(?<= )\|\|(?= )','or',input())))
```
# [Validating Email Addresses With a Filter ](https://www.hackerrank.com/challenges/validate-list-of-email-address-with-filter/problem) - 20.0 - python3
```python
import re

def fun(s):
    return re.match('[\w\-]+@[a-zA-Z\d]+\.[a-zA-Z]{1,3}$',s)
```
# [Classes: Dealing with Complex Numbers](https://www.hackerrank.com/challenges/class-1-dealing-with-complex-numbers/problem) - 20.0 - python3
```python
class Complex(object):
    def __init__(self, r, i):
        self.r = r
        self.i = i

    def __iter__(self):
        return iter((self.r, self.i))
    
    def __add__(self, no):
        return Complex(*(a+b for a,b in zip(self,no)))
        
    def __sub__(self, no):
        return Complex(*(a-b for a,b in zip(self,no)))
        
    def __mul__(self, no):
        return Complex(self.r*no.r-self.i*no.i,
                       self.r*no.i+self.i*no.r)

    def __truediv__(self, no):
        d = no.r**2+no.i**2
        return Complex((self.r*no.r+self.i*no.i)/d,
                       (self.i*no.r-self.r*no.i)/d)

    def mod(self):
        return Complex(math.sqrt(self.r**2+self.i**2),0)

    def __str__(self):
        if self.i == 0:
            result = "%.2f+0.00i" % (self.r)
        elif self.r == 0:
            if self.i >= 0:
                result = "0.00+%.2fi" % (self.i)
            else:
                result = "0.00-%.2fi" % (abs(self.i))
        elif self.i > 0:
            result = "%.2f+%.2fi" % (self.r, self.i)
        else:
            result = "%.2f-%.2fi" % (self.r, abs(self.i))
        return result
```
# [Validating Roman Numerals](https://www.hackerrank.com/challenges/validate-a-roman-number/problem) - 20.0 - python3
```python
regex_pattern = r"M{,3}(D?C{,3}|CD|CM)(L?X{,3}|XL|XC)(V?I{,3}|IV|IX)$"
```
# [Maximum Element](https://www.hackerrank.com/challenges/maximum-element/problem) - 20.0 - python3
```python
S,m = [],0
for _ in range(int(input())):
    Q=input().split()
    if Q[0]=='1':
        S.append(int(Q[1]))
        m=max(S[-1],m)
    elif Q[0]=='2':
        l=S.pop()
        if l==m:m=max(S) if S else 0
    else:print(m)
```
# [Non-Divisible Subset](https://www.hackerrank.com/challenges/non-divisible-subset/problem) - 20.0 - python3
```python
from collections import Counter
(n,k),S = eval('[*map(int,input().split())],'*2)
F = Counter([n%k for n in S])
t = sum(max(F[i],F[k-i]) for i in range(1,k//2+k%2))
print(t + (k%2==0)*bool(F[k//2]) + any(n%k==0 for n in S))
```
# [The Power Sum](https://www.hackerrank.com/challenges/the-power-sum/problem) - 20.0 - python3
```python
def power_sum(x,n,b=1):
    p = pow(b,n)
    if p < x: return power_sum(x,n,b+1) + power_sum(x-p,n,b+1)
    if p == x: return 1
    return 0

print(power_sum(int(input()),int(input())))
```
# [Tree: Level Order Traversal](https://www.hackerrank.com/challenges/tree-level-order-traversal/problem) - 20.0 - python3
```python
"""
Node is defined as
self.left (the left child of the node)
self.right (the right child of the node)
self.info (the value of the node)
"""
from collections import deque

def levelOrder(root):
    print(root.info,end=' ')
    Level = deque()
    if root.left: Level += [root.left]
    if root.right: Level += [root.right]
    while Level:
        last = Level.popleft()
        print(last.info,end=' ')
        if last.left: Level += [last.left]
        if last.right: Level += [last.right]
```
# [Binary Search Tree : Insertion](https://www.hackerrank.com/challenges/binary-search-tree-insertion/problem) - 20.0 - python3
```python
#Node is defined as
#self.left (the left child of the node)
#self.right (the right child of the node)
#self.info (the vue of the node)

    def insertion(self, c, v):
        if not c:  c = Node(v)
        elif c.info > v: c.left = self.insertion(c.left, v)
        else: c.right = self.insertion(c.right, v)
        return c
        

    def insert(self, v):
        if self.root: self.insertion(self.root, v)
        else: self.root = Node(v)
```
# [Max Array Sum ](https://www.hackerrank.com/challenges/max-array-sum/problem) - 20.0 - pypy3
```python
_,A=input(),[*map(int,input().split())]
m,l=max(A[:2]),A[0]
for e in A[2:]: m,l=max(e,e+l,m),m
print(m)
```
# [Arrays: Left Rotation](https://www.hackerrank.com/challenges/ctci-array-left-rotation/problem) - 20.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the rotLeft function below.
def rotLeft(a, d):
    d %= len(a)
    return a[d:] + a[:d]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nd = input().split()

    n = int(nd[0])

    d = int(nd[1])

    a = list(map(int, input().rstrip().split()))

    result = rotLeft(a, d)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [Time Conversion](https://www.hackerrank.com/challenges/time-conversion/problem) - 15.0 - python3
```python
#!/bin/python3

import os
import sys

#
# Complete the timeConversion function below.
#
def timeConversion(s):
    if s[:2] == '12': s = '00' + s[2:]
    if s[-2:] == 'AM': return s[:-2]
    else: return str(int(s[:2]) + 12) + s[2:-2]

if __name__ == '__main__':
    f = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = timeConversion(s)

    f.write(result + '\n')

    f.close()
```
# [Strong Password](https://www.hackerrank.com/challenges/strong-password/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the minimumNumber function below.
def minimumNumber(n, password):
    additions = 0
    patterns = ['[A-Z]+',
                '[a-z]+',
                '[0-9]+',
                '[!@#$%^&*()+-]+']
    for p in patterns:
        if (not re.search(p,password)):
            additions += 1
    return 6-n if 6-n > additions else additions

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    password = input()

    answer = minimumNumber(n, password)

    fptr.write(str(answer) + '\n')

    fptr.close()
```
# [Caesar Cipher](https://www.hackerrank.com/challenges/caesar-cipher-1/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the caesarCipher function below.
def caesarCipher(s, k):
    enc = ''
    for c in s:
        if re.match('[a-z]',c):
            enc += chr((ord(c)-97+k)%26+97)
        elif re.match('[A-Z]',c):
            enc += chr((ord(c)-65+k)%26+65)
        else: enc += c
    return enc

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    s = input()

    k = int(input())

    result = caesarCipher(s, k)

    fptr.write(result + '\n')

    fptr.close()
```
# [Mars Exploration](https://www.hackerrank.com/challenges/mars-exploration/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the marsExploration function below.
def marsExploration(s):
    c, a = int(len(s)/3)*'SOS', 0
    for i in range(len(s)):
        if c[i] != s[i]: a += 1
    return a

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = marsExploration(s)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [CamelCase](https://www.hackerrank.com/challenges/camelcase/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the camelcase function below.
def camelcase(s): return len(re.findall('[A-Z]',s))+1

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = camelcase(s)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Counting Valleys](https://www.hackerrank.com/challenges/counting-valleys/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys
from itertools import groupby

'''
A mountain is a sequence of consecutive steps above sea level, starting with a step up from sea level and ending with a step down to sea level.
A valley is a sequence of consecutive steps below sea level, starting with a step down from sea level and ending with a step up to sea level.
'''
# Complete the countingValleys function below.
def countingValleys(n, s):
    h, v_c, in_v = 0, 0, False
    for d,l in groupby(s):
        l = len(list(l))
        h = h+l if d=='U' else h-l
        if h >= 0: in_v = False
        elif h < 0 and not in_v: v_c, in_v = v_c+1, True
    return v_c

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    s = input()

    result = countingValleys(n, s)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Beautiful Days at the Movies](https://www.hackerrank.com/challenges/beautiful-days-at-the-movies/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the beautifulDays function below.
def beautifulDays(i, j, k):
    return sum(1 for d in range(i,j+1) if abs(d-int(str(d)[::-1]))%k==0)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ijk = input().split()

    i = int(ijk[0])

    j = int(ijk[1])

    k = int(ijk[2])

    result = beautifulDays(i, j, k)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Viral Advertising](https://www.hackerrank.com/challenges/strange-advertising/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the viralAdvertising function below.
def viralAdvertising(n):
    liked, cumulative = 2, 2
    for i in range(n-1):
        liked = liked*3//2
        cumulative += liked
    return cumulative

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Cats and a Mouse](https://www.hackerrank.com/challenges/cats-and-a-mouse/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the catAndMouse function below.
'''
x: an integer, Cat A's position
y: an integer, Cat B's position
z: an integer, Mouse C's position
'''
def catAndMouse(x, y, z):
    a, b = abs(z-x), abs(z-y)
    if a < b: return 'Cat A'
    elif b < a: return 'Cat B'
    return 'Mouse C'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        xyz = input().split()

        x = int(xyz[0])

        y = int(xyz[1])

        z = int(xyz[2])

        result = catAndMouse(x, y, z)

        fptr.write(result + '\n')

    fptr.close()
```
# [Electronics Shop](https://www.hackerrank.com/challenges/electronics-shop/problem) - 15.0 - python3
```python
#!/bin/python3

import os
import sys

#
# Complete the getMoneySpent function below.
#
def getMoneySpent(keyboards, drives, b):
    spent = -1
    for k,d in [(k,d) for k in keyboards for d in drives]:
        if spent < k+d <= b:
            spent = k+d
            if spent == b: break
    return spent

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    bnm = input().split()

    b = int(bnm[0])

    n = int(bnm[1])

    m = int(bnm[2])

    keyboards = list(map(int, input().rstrip().split()))

    drives = list(map(int, input().rstrip().split()))

    #
    # The maximum amount of money she can spend on a keyboard and USB drive, or -1 if she can't purchase both items
    #

    moneySpent = getMoneySpent(keyboards, drives, b)

    fptr.write(str(moneySpent) + '\n')

    fptr.close()
```
# [The Hurdle Race](https://www.hackerrank.com/challenges/the-hurdle-race/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the hurdleRace function below.
def hurdleRace(k, height):
    d = max(height) - k
    return d if d > 0 else 0

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = int(nk[0])

    k = int(nk[1])

    height = list(map(int, input().rstrip().split()))

    result = hurdleRace(k, height)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Jumping on the Clouds: Revisited](https://www.hackerrank.com/challenges/jumping-on-the-clouds-revisited/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the jumpingOnClouds function below.
def jumpingOnClouds(c, k):
    e, n, i = 100, len(c), 0
    while True:
        i = (i+k)%n
        e = e - c[i]*2 - 1
        if i == 0: break
    return e

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = int(nk[0])

    k = int(nk[1])

    c = list(map(int, input().rstrip().split()))

    result = jumpingOnClouds(c, k)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Day of the Programmer](https://www.hackerrank.com/challenges/day-of-the-programmer/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the dayOfProgrammer function below.
def dayOfProgrammer(year):
    if year == 1918: dm = '26.09.'
    elif year%4==0 and not (year > 1918 and year%100==0 and not year%400==0): dm = '12.09.'
    else: dm = '13.09.'
    return dm + str(year)
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    year = int(input().strip())
    result = dayOfProgrammer(year)
    fptr.write(result + '\n')
    fptr.close()
```
# [Save the Prisoner!](https://www.hackerrank.com/challenges/save-the-prisoner/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the saveThePrisoner function below.
def saveThePrisoner(n, m, s):
    return (s+m-2)%n+1

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        nms = input().split()

        n = int(nms[0])

        m = int(nms[1])

        s = int(nms[2])

        result = saveThePrisoner(n, m, s)

        fptr.write(str(result) + '\n')

    fptr.close()
```
# [Marc's Cakewalk](https://www.hackerrank.com/challenges/marcs-cakewalk/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the marcsCakewalk function below.
def marcsCakewalk(C):
    C.sort(reverse=True)
    return sum(2**i * C[i] for i in range(len(C)))


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    calorie = list(map(int, input().rstrip().split()))
    result = marcsCakewalk(calorie)
    fptr.write(str(result) + '\n')
    fptr.close()
```
# [A Chessboard Game](https://www.hackerrank.com/challenges/a-chessboard-game-1/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

def chessboardGame(x,y):
    x %= 4
    y %= 4
    return ('Second','First')[y==0 or y==3 or x==0 or x==3]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        xy = input().split()
        x = int(xy[0])
        y = int(xy[1])
        result = chessboardGame(x, y)
        fptr.write(result + '\n')
    fptr.close()
```
# [Tower Breakers ](https://www.hackerrank.com/challenges/tower-breakers-1/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the towerBreakers function below.
def towerBreakers(n, m): return (1,2)[m==1 or n%2-1]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        nm = input().split()
        n = int(nm[0])
        m = int(nm[1])
        result = towerBreakers(n, m)
        fptr.write(str(result) + '\n')
    fptr.close()
```
# [Game of Stones](https://www.hackerrank.com/challenges/game-of-stones-1/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the gameOfStones function below.
def gameOfStones(n): return ('First','Second')[n%7 in (0,1)]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        n = int(input())

        result = gameOfStones(n)

        fptr.write(result + '\n')

    fptr.close()
```
# [Minimum Absolute Difference in an Array](https://www.hackerrank.com/challenges/minimum-absolute-difference-in-an-array/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the minimumAbsoluteDifference function below.
def minimumAbsoluteDifference(A):
    A.sort()
    return min(abs(A[i]-A[i+1]) for i in range(len(A)-1))


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    result = minimumAbsoluteDifference(arr)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Library Fine](https://www.hackerrank.com/challenges/library-fine/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the libraryFine function below.
def libraryFine(d1, m1, y1, d2, m2, y2):
    if y2<y1:return 10000
    elif y1==y2 and m2<m1:return 500 * (m1-m2)
    elif y1==y2 and m1==m2 and d2<d1:return 15 * (d1-d2)
    else:return 0

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    d1M1Y1 = input().split()

    d1 = int(d1M1Y1[0])

    m1 = int(d1M1Y1[1])

    y1 = int(d1M1Y1[2])

    d2M2Y2 = input().split()

    d2 = int(d2M2Y2[0])

    m2 = int(d2M2Y2[1])

    y2 = int(d2M2Y2[2])

    result = libraryFine(d1, m1, y1, d2, m2, y2)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Introduction to Nim Game](https://www.hackerrank.com/challenges/nim-game-1/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

from functools import reduce

# Complete the nimGame function below.
def nimGame(pile):
    return ('Second','First')[reduce(lambda a,b:a^b,pile)>0]


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    g = int(input())

    for g_itr in range(g):
        n = int(input())

        pile = list(map(int, input().rstrip().split()))

        result = nimGame(pile)

        fptr.write(result + '\n')

    fptr.close()
```
# [2D Array - DS](https://www.hackerrank.com/challenges/2d-array/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the hourglassSum function below.
def hourglassSum(A):
    return max(sum(A[i][j:j+3]+[A[i+1][j+1]]+A[i+2][j:j+3]) for i,j in ((i,j) for i in range(4) for j in range(4)))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    arr = []

    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))

    result = hourglassSum(arr)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Dynamic Array](https://www.hackerrank.com/challenges/dynamic-array/problem) - 15.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'dynamicArray' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. INTEGER n
#  2. 2D_INTEGER_ARRAY queries
#

def dynamicArray(n, queries):
    seqList = [[] for _ in range(n)]
    lastAnswer = 0
    result = []
    for q in queries:
        if q[0] == 1:
            seqList[(q[1] ^ lastAnswer) % n].append(q[2])
        else:
            seq = seqList[(q[1] ^ lastAnswer) % n]
            lastAnswer = seq[q[2] % len(seq)]
            result.append(lastAnswer)
    return result

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = int(first_multiple_input[0])

    q = int(first_multiple_input[1])

    queries = []

    for _ in range(q):
        queries.append(list(map(int, input().rstrip().split())))

    result = dynamicArray(n, queries)

    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [Day 7: Regular Expressions I](https://www.hackerrank.com/challenges/js10-regexp-1/problem) - 15.0 - javascript
```javascript
function regexVar() {
    return /^([aeiou]).*\1$/;
}
```
# [Day 3: Try, Catch, and Finally](https://www.hackerrank.com/challenges/js10-try-catch-and-finally/problem) - 15.0 - javascript
```javascript
/*
 * Complete the reverseString function
 * Use console.log() to print to stdout.
 */
function reverseString(s) {
    try { s = s.split('').reverse().join('') }
    catch (e) { console.log(e.message) }
    console.log(s)
}
```
# [Day 3: Arrays](https://www.hackerrank.com/challenges/js10-arrays/problem) - 15.0 - javascript
```javascript
/**
*   Return the second largest number in the array.
*   @param {Number[]} nums - An array of numbers.
*   @return {Number} The second largest number in the array.
**/
function getSecondLargest(nums) {
    return [...new Set(nums)].sort((a,b)=>b>a)[1]
}
```
# [Day 3: Throw](https://www.hackerrank.com/challenges/js10-throw/problem) - 15.0 - javascript
```javascript
/*
 * Complete the isPositive function.
 * If 'a' is positive, return "YES".
 * If 'a' is 0, throw an Error with the message "Zero Error"
 * If 'a' is negative, throw an Error with the message "Negative Error"
 */
function isPositive(a) {
    if(a>0) {return "YES";}
    throw new Error(a<0?"Negative Error":"Zero Error");
}
```
# [Day 4: Create a Rectangle Object](https://www.hackerrank.com/challenges/js10-objects/problem) - 15.0 - javascript
```javascript
/*
 * Complete the Rectangle function
 */
function Rectangle(a, b) {
    return {length:a,width:b,perimeter: 2*(a+b),area: a*b}
}
```
# [Day 4: Count Objects](https://www.hackerrank.com/challenges/js10-count-objects/problem) - 15.0 - javascript
```javascript
/*
 * Return a count of the total number of objects 'o' satisfying o.x == o.y.
 * 
 * Parameter(s):
 * objects: an array of objects with integer properties 'x' and 'y'
 */
const getCount = O=>O.reduce((t,n)=>t+(n.x==n.y),0)
```
# [Day 4: Classes](https://www.hackerrank.com/challenges/js10-class/problem) - 15.0 - javascript
```javascript
class Polygon {
    constructor(S) { this.S = S }
    perimeter() { return this.S.reduce((t,s)=>t+s,0) }
}
```
# [Day 5: Arrow Functions](https://www.hackerrank.com/challenges/js10-arrows/problem) - 15.0 - javascript
```javascript
/*
 * Modify and return the array so that all even elements are doubled and all odd elements are tripled.
 * 
 * Parameter(s):
 * nums: An array of numbers.
 */
const modifyArray = N=>N.map(n=>n*(2+n%2))
```
# [Day 5: Template Literals](https://www.hackerrank.com/challenges/js10-template-literals/problem) - 15.0 - javascript
```javascript
/*
 * Determine the original side lengths and return an array:
 * - The first element is the length of the shorter side
 * - The second element is the length of the longer side
 * 
 * Parameter(s):
 * literals: The tagged template literal's array of strings.
 * expressions: The tagged template literal's array of expression values (i.e., [area, perimeter]).
 */
function sides(literals, a, p) {
    let s = Math.sqrt(p**2 - 16*a)
    return [(p+s)/4,(p-s)/4].sort((a,b)=>a>b)
}
```
# [Day 5: Inheritance](https://www.hackerrank.com/challenges/js10-inheritance/problem) - 15.0 - javascript
```javascript
Rectangle.prototype.area = function() { return this.w * this.h; }
class Square extends Rectangle { constructor(s) { super(s,s); } }
```
# [Weather Observation Station 1](https://www.hackerrank.com/challenges/weather-observation-station-1/problem) - 15.0 - oracle
```sql
SELECT CITY, STATE FROM STATION;
```
# [Compare the Triplets](https://www.hackerrank.com/challenges/compare-the-triplets/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the compareTriplets function below.
def compareTriplets(a, b):
    points = [0,0]

    for i in range(3):
        if a[i] < b[i]: points[1] += 1
        if a[i] > b[i]: points[0] += 1

    return points

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    a = list(map(int, input().rstrip().split()))

    b = list(map(int, input().rstrip().split()))

    result = compareTriplets(a, b)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [Diagonal Difference](https://www.hackerrank.com/challenges/diagonal-difference/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'diagonalDifference' function below.
#
# The function is expected to return an INTEGER.
# The function accepts 2D_INTEGER_ARRAY arr as parameter.
#

def diagonalDifference(arr):
    result, l = 0, len(arr[0])

    for i in range(l):
        result += arr[i][i] - arr[i][l-i-1]

    return abs(result)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    result = diagonalDifference(arr)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Plus Minus](https://www.hackerrank.com/challenges/plus-minus/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the plusMinus function below.
def plusMinus(arr):
    l = len(arr)
    pos,neg,zero = 0,0,0
    for n in arr:
        if (n > 0):
            pos += 1
        elif (n < 0):
            neg += 1
        else:
            zero += 1
    print("%.6f" % (pos/l))
    print("%.6f" % (neg/l))
    print("%.6f" % (zero/l))

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    plusMinus(arr)
```
# [Staircase](https://www.hackerrank.com/challenges/staircase/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the staircase function below.
def staircase(n):
    for i in range(1,n+1):
        print((n-i)*' ' + (i)*'#')

if __name__ == '__main__':
    n = int(input())

    staircase(n)
```
# [Mini-Max Sum](https://www.hackerrank.com/challenges/mini-max-sum/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the miniMaxSum function below.
def miniMaxSum(arr):
    arr.sort()
    print('%s %s' % (sum(arr[:len(arr)-1]), sum(arr[1:])))

if __name__ == '__main__':
    arr = list(map(int, input().rstrip().split()))

    miniMaxSum(arr)
```
# [Birthday Cake Candles](https://www.hackerrank.com/challenges/birthday-cake-candles/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the birthdayCakeCandles function below.
def birthdayCakeCandles(arr):
    return arr.count(max(arr))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(ar)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Grading Students](https://www.hackerrank.com/challenges/grading/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'gradingStudents' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts INTEGER_ARRAY grades as parameter.
#

def gradingStudents(grades):
    for i in range(len(grades)):
        if grades[i] > 37 and grades[i] % 5 > 2:
            grades[i] += (5 - grades[i] % 5)
    return grades
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    grades_count = int(input().strip())

    grades = []

    for _ in range(grades_count):
        grades_item = int(input().strip())
        grades.append(grades_item)

    result = gradingStudents(grades)

    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [Divisible Sum Pairs](https://www.hackerrank.com/challenges/divisible-sum-pairs/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the divisibleSumPairs function below.
def divisibleSumPairs(n, k, a):
    pairs = 0
    for j in range(1,len(a)):
        for i in range(j):
            if (a[i] + a[j]) % k == 0:
                pairs += 1
    return pairs


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = int(nk[0])

    k = int(nk[1])

    ar = list(map(int, input().rstrip().split()))

    result = divisibleSumPairs(n, k, ar)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Super Reduced String](https://www.hackerrank.com/challenges/reduced-string/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the superReducedString function below.
def superReducedString(s):
    i = 0
    while True:
        if s[i] == s[i+1]:
            s = s[:i] + s[i+2:]
            if i > 0: i -= 1
        if i >= len(s) - 1: break
        if s[i] != s[i+1]: i += 1
        if i >= len(s) - 1: break
    if s == '': return 'Empty String'
    else: return s


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = superReducedString(s)

    fptr.write(result + '\n')

    fptr.close()
```
# [Apple and Orange](https://www.hackerrank.com/challenges/apple-and-orange/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the countApplesAndOranges function below.
'''
s: integer, starting point of Sam's house location.
t: integer, ending location of Sam's house location.
a: integer, location of the Apple tree.
b: integer, location of the Orange tree.
apples: integer array, distances at which each apple falls from the tree.
oranges: integer array, distances at which each orange falls from the tree.
'''
def countApplesAndOranges(s, t, a, b, apples, oranges):
    a_count, b_count = 0, 0
    for fruit in apples:
        if s <= a + fruit <= t:
            a_count += 1
    for fruit in oranges:
        if s <= b + fruit <= t:
            b_count += 1
    print('%s\n%s' % (a_count, b_count))



if __name__ == '__main__':
    st = input().split()

    s = int(st[0])

    t = int(st[1])

    ab = input().split()

    a = int(ab[0])

    b = int(ab[1])

    mn = input().split()

    m = int(mn[0])

    n = int(mn[1])

    apples = list(map(int, input().rstrip().split()))

    oranges = list(map(int, input().rstrip().split()))

    countApplesAndOranges(s, t, a, b, apples, oranges)
```
# [Number Line Jumps](https://www.hackerrank.com/challenges/kangaroo/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the kangaroo function below.
'''
x1, v1: integers, starting position and jump distance for kangaroo 1
x2, v2: integers, starting position and jump distance for kangaroo 2
'''
def kangaroo(x1, v1, x2, v2):
    if v1 != v2:
        t = (-x1 + x2)/(v1 - v2)
        if t > 0 and t.is_integer(): return "YES"
    return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()
```
# [Between Two Sets](https://www.hackerrank.com/challenges/between-two-sets/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'getTotalX' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY a
#  2. INTEGER_ARRAY b
#

'''
1. The elements of A are all factors of the integer being considered
2. The integer being considered is a factor of all elements of B
'''
def getTotalX(a, b):
    m = max(a)
    candidates = (c for c in range(m,101,m) if all(c%divisor==0 for divisor in a))
    return len([c for c in candidates if all(dividend%c==0 for dividend in b)])
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = int(first_multiple_input[0])

    m = int(first_multiple_input[1])

    arr = list(map(int, input().rstrip().split()))

    brr = list(map(int, input().rstrip().split()))

    total = getTotalX(arr, brr)

    fptr.write(str(total) + '\n')

    fptr.close()
```
# [Breaking the Records](https://www.hackerrank.com/challenges/breaking-best-and-worst-records/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the breakingRecords function below.
def breakingRecords(s):
    h, l, n_h, n_l = [s[0]]*2+[0]*2
    for i in range(1,len(s)):
        if s[i] > h: h,n_h = s[i],n_h+1
        if s[i] < l: l,n_l = s[i],n_l+1
    return [n_h,n_l]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    scores = list(map(int, input().rstrip().split()))

    result = breakingRecords(scores)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [Sub-array Division](https://www.hackerrank.com/challenges/the-birthday-bar/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the birthday function below.
def birthday(s, d, m):
    ways = 0
    for i in range(len(s)-m+1):
        if sum(s[i:i+m]) == d: ways += 1
    return ways

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    s = list(map(int, input().rstrip().split()))

    dm = input().rstrip().split()

    d = int(dm[0])

    m = int(dm[1])

    result = birthday(s, d, m)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Migratory Birds](https://www.hackerrank.com/challenges/migratory-birds/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the migratoryBirds function below.
def migratoryBirds(arr):
    freq = [arr.count(t+1) for t in range(5)]
    return min(t+1 for t in range(5) if freq[t] == max(freq))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    arr_count = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    result = migratoryBirds(arr)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Bill Division](https://www.hackerrank.com/challenges/bon-appetit/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the bonAppetit function below.
'''
bill: an array of integers representing the cost of each item ordered
k: an integer representing the zero-based index of the item Anna doesn't eat
b: the amount of money that Anna contributed to the bill
'''
def bonAppetit(bill, item, charged):
    actual = sum(bill[:item]+bill[item+1:])/2
    print(int(charged - actual)) if charged > actual else print('Bon Appetit')

if __name__ == '__main__':
    nk = input().rstrip().split()

    n = int(nk[0])

    k = int(nk[1])

    bill = list(map(int, input().rstrip().split()))

    b = int(input().strip())

    bonAppetit(bill, k, b)
```
# [Drawing Book ](https://www.hackerrank.com/challenges/drawing-book/problem) - 10.0 - python3
```python
#!/bin/python3

import os
import sys
from math import floor, ceil

#
# Complete the pageCount function below.
#
def pageCount(n, p):
    return min([p//2,n//2-p//2])
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    p = int(input())

    result = pageCount(n, p)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Python If-Else](https://www.hackerrank.com/challenges/py-if-else/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
    if n%2 == 0:
        if n < 6: print('Not Weird')
        elif n < 21: print('Weird')
        else: print ('Not Weird')
    else: print('Weird')
```
# [Arithmetic Operators](https://www.hackerrank.com/challenges/python-arithmetic-operators/problem) - 10.0 - python3
```python
'''The first line contains the sum of the two numbers.
The second line contains the difference of the two numbers (first - second).
The third line contains the product of the two numbers.
'''

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print("%s\n%s\n%s" % (a+b,a-b,a*b))
```
# [Python: Division](https://www.hackerrank.com/challenges/python-division/problem) - 10.0 - python3
```python
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print("%s\n%s" % (a//b,a/b))
```
# [Loops](https://www.hackerrank.com/challenges/python-loops/problem) - 10.0 - python3
```python
if __name__ == '__main__':
    n = int(input())
    for i in range(n): print(i**2)
```
# [Write a function](https://www.hackerrank.com/challenges/write-a-function/problem) - 10.0 - python3
```python
def is_leap(year):
    return year%4==0 and (not year%100==0 or year%400==0)
```
# [List Comprehensions](https://www.hackerrank.com/challenges/list-comprehensions/problem) - 10.0 - python3
```python
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    print([[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k != n])
```
# [Find the Runner-Up Score!  ](https://www.hackerrank.com/challenges/find-second-maximum-number-in-a-list/problem) - 10.0 - python3
```python
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    print(sorted([*{*arr}])[-2])
```
# [Finding the percentage](https://www.hackerrank.com/challenges/finding-the-percentage/problem) - 10.0 - python3
```python
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    print("{:.2f}".format(sum(student_marks[query_name])/len(student_marks[query_name])))
```
# [Quicksort 1 - Partition](https://www.hackerrank.com/challenges/quicksort1/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the quickSort function below.
def quickSort(A):
    p=A[0]
    l,c,r=[],[],[]
    for e in A:
        if e<p: l+=[e]
        elif e>p: r+=[e]
        else: c+=[e]
    return l+c+r

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    result = quickSort(arr)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```
# [Arrays - DS](https://www.hackerrank.com/challenges/arrays-ds/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the reverseArray function below.
def reverseArray(a): return a[::-1]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    arr_count = int(input())

    arr = list(map(int, input().rstrip().split()))

    res = reverseArray(arr)

    fptr.write(' '.join(map(str, res)))
    fptr.write('\n')

    fptr.close()
```
# [Nested Lists](https://www.hackerrank.com/challenges/nested-list/problem) - 10.0 - python3
```python
S=[[input(),float(input())] for _ in range(int(input()))]
print('\n'.join(sorted(s[0] for s in S if s[1] ==
                       sorted({*(s[1] for s in S)})[1])))
```
# [Tuples ](https://www.hackerrank.com/challenges/python-tuples/problem) - 10.0 - python3
```python
input()
print(hash(tuple(map(int,input().split()))))
```
# [Lists](https://www.hackerrank.com/challenges/python-lists/problem) - 10.0 - python3
```python
L = []
for _ in range(int(input())):
    C = input().split()
    if C[0]=='print': print(L)
    else: eval('L.'+C[0]+'('+','.join(C[1:])+')')
```
# [itertools.product()](https://www.hackerrank.com/challenges/itertools-product/problem) - 10.0 - python3
```python
from itertools import product
print(*product(*[map(int,input().split()) for _ in range(2)]))
```
# [itertools.permutations()](https://www.hackerrank.com/challenges/itertools-permutations/problem) - 10.0 - python3
```python
from itertools import permutations
(S, k) = input().split()
print('\n'.join(sorted(''.join(l)
                for l in permutations(S,int(k)))))
```
# [itertools.combinations()](https://www.hackerrank.com/challenges/itertools-combinations/problem) - 10.0 - python3
```python
from itertools import combinations
S,k=input().split();S,k=sorted(S),int(k)
for r in range(1,k+1):
    for C in combinations(S,r): print(''.join(C))
```
# [itertools.combinations_with_replacement()](https://www.hackerrank.com/challenges/itertools-combinations-with-replacement/problem) - 10.0 - python3
```python
from itertools import combinations_with_replacement
S,k=input().split();S,k=sorted(S),int(k)
for C in combinations_with_replacement(S,k): print(''.join(C))
```
# [sWAP cASE](https://www.hackerrank.com/challenges/swap-case/problem) - 10.0 - python3
```python
def swap_case(s):
    return s.swapcase()
```
# [String Split and Join](https://www.hackerrank.com/challenges/python-string-split-and-join/problem) - 10.0 - python3
```python
def split_and_join(line):
    return '-'.join(line.strip().split(' '))
```
# [What's Your Name?](https://www.hackerrank.com/challenges/whats-your-name/problem) - 10.0 - python3
```python
def print_full_name(a, b):
    print("Hello %s %s! You just delved into python." % (a,b))
```
# [Mutations](https://www.hackerrank.com/challenges/python-mutations/problem) - 10.0 - python3
```python
def mutate_string(S, i, c):
    return S[:i] + c + S[i+1:]
```
# [Find a string](https://www.hackerrank.com/challenges/find-a-string/problem) - 10.0 - python3
```python
import re
def count_substring(s, ss):
    return len([*re.finditer(r'(?='+ss+')',s)])
```
# [String Validators](https://www.hackerrank.com/challenges/string-validators/problem) - 10.0 - python3
```python
import re
s,R = input(),('\d','[a-z]','[A-Z]')
d,l,u = (re.search(r,s)!=None for r in R)
print('\n'.join(map(str,(l|u|d,l|u,d,l,u))))
```
# [Text Alignment](https://www.hackerrank.com/challenges/text-alignment/problem) - 10.0 - python3
```python
#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))
```
# [Text Wrap](https://www.hackerrank.com/challenges/text-wrap/problem) - 10.0 - python3
```python
def wrap(s, W):
    return '\n'.join(textwrap.wrap(s,W))
```
# [Designer Door Mat](https://www.hackerrank.com/challenges/designer-door-mat/problem) - 10.0 - python3
```python
h,w = map(int,input().split())
P = [('.|.'*(1+2*n)).center(w,'-') for n in range(h//2)]
print('\n'.join(P+['WELCOME'.center(w,'-')]+P[::-1]))
```
# [String Formatting](https://www.hackerrank.com/challenges/python-string-formatting/problem) - 10.0 - python3
```python
def print_formatted(n):
    w = len(bin(n))-2
    for i in range(1,n+1):
        D = (i,oct(i)[2:],hex(i)[2:].upper(),bin(i)[2:])
        print(' '.join(str(e).rjust(w) for e in D))
```
# [collections.Counter()](https://www.hackerrank.com/challenges/collections-counter/problem) - 10.0 - python3
```python
l, L, t = int(input()),input().split(), 0
for _ in range(int(input())):
    s, d = input().split()
    if s in L:
        L.remove(s)
        t += int(d)
print(t)
```
# [Introduction to Sets](https://www.hackerrank.com/challenges/py-introduction-to-sets/problem) - 10.0 - python3
```python
def average(A):
    A = {*A}
    return sum(A)/len(A)
```
# [Exceptions](https://www.hackerrank.com/challenges/exceptions/problem) - 10.0 - python3
```python
for _ in range(int(input())):
    try:
        a,b=map(int,input().split())
        print(a//b)
    except Exception as e:
        print("Error Code:",e)
```
# [Calendar Module](https://www.hackerrank.com/challenges/calendar-module/problem) - 10.0 - python3
```python
import calendar
m,d,y = map(int,input().split())
print(calendar.day_name[calendar.weekday(y,m,d)].upper())
```
# [Polar Coordinates](https://www.hackerrank.com/challenges/polar-coordinates/problem) - 10.0 - python3
```python
from cmath import phase
c = complex(input())
print(abs(c))
print(phase(c))
```
# [Symmetric Difference](https://www.hackerrank.com/challenges/symmetric-difference/problem) - 10.0 - python3
```python
_,m,_,n=({*input().split()}for _ in range(4))
print('\n'.join(sorted(n^m,key=int)))
```
# [Set .add() ](https://www.hackerrank.com/challenges/py-set-add/problem) - 10.0 - python3
```python
print(len({*open(0).read().split('\n')})-1)
```
# [Zipped!](https://www.hackerrank.com/challenges/zipped/problem) - 10.0 - python3
```python
x=int(input().split()[1])
for s in zip(*(input().split() for _ in range(x))):
    print(sum(map(float,s))/x)
```
# [Check Strict Superset](https://www.hackerrank.com/challenges/py-check-strict-superset/problem) - 10.0 - python3
```python
A = {*input().split()}
print(all({*input().split()}.issubset(A) for _ in range(int(input()))))
```
# [Set .union() Operation](https://www.hackerrank.com/challenges/py-set-union/problem) - 10.0 - python3
```python
_,a,_,b = ({*input().split()} for _ in range(4))
print(len(a|b))
```
# [Set .intersection() Operation](https://www.hackerrank.com/challenges/py-set-intersection-operation/problem) - 10.0 - python3
```python
_,a,_,b=({*input().split()}for _ in range(4));print(len(a&b))
```
# [Set .difference() Operation](https://www.hackerrank.com/challenges/py-set-difference-operation/problem) - 10.0 - python3
```python
_,a,_,b=({*input().split()}for _ in range(4));print(len(a-b))
```
# [Set .discard(), .remove() & .pop()](https://www.hackerrank.com/challenges/py-set-discard-remove-pop/problem) - 10.0 - python3
```python
input();S={*map(int,input().split())}
for _ in range(int(input())):
    eval('S.{}({})'.format(*input().split()+['']))
print(sum(S))
```
# [Mod Divmod](https://www.hackerrank.com/challenges/python-mod-divmod/problem) - 10.0 - python3
```python
S=divmod(*eval('int(input()),'*2))
print(*S,S,sep='\n')
```
# [Power - Mod Power](https://www.hackerrank.com/challenges/python-power-mod-power/problem) - 10.0 - python3
```python
a,b,m=eval('int(input()),'*3)
print(a**b,a**b%m,sep='\n')
```
# [Integers Come In All Sizes](https://www.hackerrank.com/challenges/python-integers-come-in-all-sizes/problem) - 10.0 - python3
```python
a,b,c,d=eval('int(input()),'*4)
print(a**b+c**d)
```
# [Set .symmetric_difference() Operation](https://www.hackerrank.com/challenges/py-set-symmetric-difference-operation/problem) - 10.0 - python3
```python
_,a,_,b=eval('{*input().split()},'*4)
print(len(a^b))
```
# [Check Subset](https://www.hackerrank.com/challenges/py-check-subset/problem) - 10.0 - python3
```python
for _ in range(int(input())):
    _,A,_,B=eval('{*input().split()},'*4)
    print(A.issubset(B))
```
# [Set Mutations](https://www.hackerrank.com/challenges/py-set-mutations/problem) - 10.0 - python3
```python
_,A,n=eval('{*map(int,input().split())},'*3)
for _ in range(*n):
    eval("A.%s({%s}),"%(input().split()[0],input().replace(" ",",")))
print(sum(A))
```
# [Find Angle MBC](https://www.hackerrank.com/challenges/find-angle/problem) - 10.0 - python3
```python
import math
print(str(round(math.degrees(math.atan(int(input())/int(input())))))+'°')
```
# [Words Score](https://www.hackerrank.com/challenges/words-score/problem) - 10.0 - python3
```python
def is_vowel(letter):
    return letter in ['a', 'e', 'i', 'o', 'u', 'y']

def score_words(words):
    score = 0
    for word in words:
        num_vowels = 0
        for letter in word:
            if is_vowel(letter):
                num_vowels += 1
        if num_vowels % 2 == 0:
            score += 2
        else:
            score += 1
    return score
```
# [The Captain's Room ](https://www.hackerrank.com/challenges/py-the-captains-room/problem) - 10.0 - python3
```python
from collections import Counter
k=int(input());R=Counter(input().split())
print(R.most_common()[-1][0])
```
# [Tree: Preorder Traversal](https://www.hackerrank.com/challenges/tree-preorder-traversal/problem) - 10.0 - python3
```python
"""
Node is defined as
self.left (the left child of the node)
self.right (the right child of the node)
self.info (the value of the node)
"""
def preOrder(node):
    if node:
        print(node.info,end=' ')
        preOrder(node.left)
        preOrder(node.right)
```
# [Tree: Postorder Traversal](https://www.hackerrank.com/challenges/tree-postorder-traversal/problem) - 10.0 - python3
```python
"""
Node is defined as
self.left (the left child of the node)
self.right (the right child of the node)
self.info (the value of the node)
"""
def postOrder(node):
    if node:
        postOrder(node.left)
        postOrder(node.right)
        print(node.info,end=' ')
```
# [Tree: Inorder Traversal](https://www.hackerrank.com/challenges/tree-inorder-traversal/problem) - 10.0 - python3
```python
"""
Node is defined as
self.left (the left child of the node)
self.right (the right child of the node)
self.info (the value of the node)
"""
def inOrder(node):
    if node:
        inOrder(node.left)
        print(node.info,end=" ")
        inOrder(node.right)
```
# [Tree: Height of a Binary Tree](https://www.hackerrank.com/challenges/tree-height-of-a-binary-tree/problem) - 10.0 - python3
```python
def height(node):
    if node: return 1 + max(height(node.left),
                            height(node.right))
    return -1
```
# [Day 0: Hello, World!](https://www.hackerrank.com/challenges/js10-hello-world/problem) - 10.0 - javascript
```javascript
/**
*   A line of code that prints "Hello, World!" on a new line is provided in the editor. 
*   Write a second line of code that prints the contents of 'parameterVariable' on a new line.
*
*	Parameter:
*   parameterVariable - A string of text.
**/
function greeting(parameterVariable) {
    // This line prints 'Hello, World!' to the console:
    console.log('Hello, World!');

    // Write a line of code that prints parameterVariable to stdout using console.log:
    console.log(parameterVariable);
}
```
# [Day 0: Data Types](https://www.hackerrank.com/challenges/js10-data-types/problem) - 10.0 - javascript
```javascript
/**
*   The variables 'firstInteger', 'firstDecimal', and 'firstString' are declared for you -- do not modify them.
*   Print three lines:
*   1. The sum of 'firstInteger' and the Number representation of 'secondInteger'.
*   2. The sum of 'firstDecimal' and the Number representation of 'secondDecimal'.
*   3. The concatenation of 'firstString' and 'secondString' ('firstString' must be first).
*
*	Parameter(s):
*   secondInteger - The string representation of an integer.
*   secondDecimal - The string representation of a floating-point number.
*   secondString - A string consisting of one or more space-separated words.
**/
function performOperation(secondInteger, secondDecimal, secondString) {
    // Declare a variable named 'firstInteger' and initialize with integer value 4.
    const firstInteger = 4;
    
    // Declare a variable named 'firstDecimal' and initialize with floating-point value 4.0.
    const firstDecimal = 4.0;
    
    // Declare a variable named 'firstString' and initialize with the string "HackerRank".
    const firstString = 'HackerRank ';
    
    // Write code that uses console.log to print the sum of the 'firstInteger' and 'secondInteger' (converted to a Number        type) on a new line.
    console.log(firstInteger + Number(secondInteger))
    
    // Write code that uses console.log to print the sum of 'firstDecimal' and 'secondDecimal' (converted to a Number            type) on a new line.
    console.log(firstDecimal + Number(secondDecimal))
    
    // Write code that uses console.log to print the concatenation of 'firstString' and 'secondString' on a new line. The        variable 'firstString' must be printed first.
    console.log(firstString + secondString)
}
```
# [Day 2: Loops](https://www.hackerrank.com/challenges/js10-loops/problem) - 10.0 - javascript
```javascript
/*
 * Complete the vowelsAndConsonants function.
 * Print your output using 'console.log()'.
 */
function vowelsAndConsonants(s) {
    for (let c of s) { if ('aeiou'.includes(c)) {console.log(c)}}
    for (let c of s) { if (!'aeiou'.includes(c)) {console.log(c)}}
}
```
# [Day 1: Arithmetic Operators](https://www.hackerrank.com/challenges/js10-arithmetic-operators/problem) - 10.0 - javascript
```javascript
/**
*   Calculate the area of a rectangle.
*
*   length: The length of the rectangle.
*   width: The width of the rectangle.
*   
*	Return a number denoting the rectangle's area.
**/
function getArea(length, width) {
    let area;
    // Write your code here
    area = length * width;
    return area;
}

/**
*   Calculate the perimeter of a rectangle.
*	
*	length: The length of the rectangle.
*   width: The width of the rectangle.
*   
*	Return a number denoting the perimeter of a rectangle.
**/
function getPerimeter(length, width) {
    let perimeter;
    // Write your code here
    perimeter = length*2 + width*2;
    return perimeter;
}
```
# [Day 1: Functions](https://www.hackerrank.com/challenges/js10-function/problem) - 10.0 - javascript
```javascript
/*
 * Create the function factorial here
 */
function factorial(n) { return n>1?n*factorial(n-1):1}
```
# [Day 1: Let and Const](https://www.hackerrank.com/challenges/js10-let-and-const/problem) - 10.0 - javascript
```javascript
function main() {
    // Write your code here. Read input using 'readLine()' and print output using 'console.log()'.
    const PI = Math.PI;
    let r = Number(readLine())
    // Print the area of the circle:
    console.log(PI*r**2)
    // Print the perimeter of the circle:
    console.log(2*r*PI)
```
# [Day 2: Conditional Statements: If-Else](https://www.hackerrank.com/challenges/js10-if-else/problem) - 10.0 - javascript
```javascript
function getGrade(score) {
    let grade;
    // Write your code here
    if (score <= 5) { grade = 'F' }
    else if (score <= 10) { grade = 'E'}
    else if (score <= 15) { grade = 'D'}
    else if (score <= 20) { grade = 'C'}
    else if (score <= 25) { grade = 'B'}
    else if (score <= 30) { grade = 'A'}
    return grade;
}
```
# [Day 2: Conditional Statements: Switch](https://www.hackerrank.com/challenges/js10-switch/problem) - 10.0 - javascript
```javascript
function getLetter(s) {
    let l;
    // Write your code here
    switch (true) {
        case 'aeiou'.includes(s[0]): l='A'; break;
        case 'bcdfg'.includes(s[0]): l='B'; break;
        case 'hjklm'.includes(s[0]): l='C'; break;
        default: l='D'
    }
    return l;
}
```
# [Sales by Match](https://www.hackerrank.com/challenges/sock-merchant/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys
from collections import Counter

# Complete the sockMerchant function below.
def sockMerchant(n, ar):
    matching_pairs = 0
    for _, value in Counter(ar).items():
        matching_pairs += int(value / 2)
    return matching_pairs


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = sockMerchant(n, ar)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Simple Array Sum](https://www.hackerrank.com/challenges/simple-array-sum/problem) - 10.0 - python3
```python
#!/bin/python3

import os
import sys

#
# Complete the simpleArraySum function below.
#
def simpleArraySum(ar):
    #
    # Write your code here.
    #
    return sum(ar)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = simpleArraySum(ar)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [A Very Big Sum](https://www.hackerrank.com/challenges/a-very-big-sum/problem) - 10.0 - python3
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the aVeryBigSum function below.
def aVeryBigSum(ar):
    return sum(ar)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = aVeryBigSum(ar)

    fptr.write(str(result) + '\n')

    fptr.close()
```
# [Revising the Select Query I](https://www.hackerrank.com/challenges/revising-the-select-query/problem) - 10.0 - oracle
```sql
SELECT * FROM CITY WHERE COUNTRYCODE = 'USA' and POPULATION > 100000;
```
# [Revising the Select Query II](https://www.hackerrank.com/challenges/revising-the-select-query-2/problem) - 10.0 - oracle
```sql
SELECT NAME FROM CITY WHERE COUNTRYCODE = 'USA' AND POPULATION > 120000;
```
# [Select All](https://www.hackerrank.com/challenges/select-all-sql/problem) - 10.0 - oracle
```sql
SELECT * FROM CITY;
```
# [Select By ID](https://www.hackerrank.com/challenges/select-by-id/problem) - 10.0 - oracle
```sql
SELECT * FROM CITY WHERE ID = 1661;
```
# [Japanese Cities' Attributes](https://www.hackerrank.com/challenges/japanese-cities-attributes/problem) - 10.0 - oracle
```sql
SELECT * FROM CITY WHERE COUNTRYCODE = 'JPN';
```
# [Japanese Cities' Names](https://www.hackerrank.com/challenges/japanese-cities-name/problem) - 10.0 - oracle
```sql
SELECT NAME FROM CITY WHERE COUNTRYCODE = 'JPN';
```
# [Weather Observation Station 3](https://www.hackerrank.com/challenges/weather-observation-station-3/problem) - 10.0 - oracle
```sql
SELECT DISTINCT CITY FROM STATION WHERE mod(ID,2) = 0;
```
# [Weather Observation Station 4](https://www.hackerrank.com/challenges/weather-observation-station-4/problem) - 10.0 - oracle
```sql
SELECT COUNT(CITY)-COUNT(DISTINCT CITY) FROM STATION
```
# [Insert a Node at the Tail of a Linked List](https://www.hackerrank.com/challenges/insert-a-node-at-the-tail-of-a-linked-list/problem) - 5.0 - python3
```python
# Complete the insertNodeAtTail function below.

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
#
#

def insertNodeAtTail(head, data):
    if head == None:
        return SinglyLinkedListNode(data)
    current = head
    while current != None:
        next_to_last = current
        current = current.next
    next_to_last.next = SinglyLinkedListNode(data)
    return head
```
# [Insert a node at the head of a linked list](https://www.hackerrank.com/challenges/insert-a-node-at-the-head-of-a-linked-list/problem) - 5.0 - python3
```python
# Complete the insertNodeAtHead function below.

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
#
#
def insertNodeAtHead(llist, data):
    if llist == None:
        return SinglyLinkedListNode(data)
    res = SinglyLinkedListNode(data)
    res.next = llist
    return res
```
# [Insert a node at a specific position in a linked list](https://www.hackerrank.com/challenges/insert-a-node-at-a-specific-position-in-a-linked-list/problem) - 5.0 - python3
```python
# Complete the insertNodeAtPosition function below.

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
#
#
def insertNodeAtPosition(head, data, position):
    if head == None:
        return SinglyLinkedListNode(data)
    current = head
    while current != None and position:
        previous = current
        current = current.next
        position -= 1
    res = SinglyLinkedListNode(data)
    previous.next = res
    res.next = current
    return head
```
# [Delete a Node](https://www.hackerrank.com/challenges/delete-a-node-from-a-linked-list/problem) - 5.0 - python3
```python
# Complete the deleteNode function below.

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
#
#
def deleteNode(head, position):
    if position == 0:
        return head.next
    current = head
    while current != None and position:
        previous = current
        current = current.next
        position -= 1
    previous.next = current.next
    return head
```
# [Print in Reverse](https://www.hackerrank.com/challenges/print-the-elements-of-a-linked-list-in-reverse/problem) - 5.0 - python3
```python
# Complete the reversePrint function below.

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
#
#
def reversePrint(head):
    L = []
    while head != None:
        L += [head.data]
        head = head.next
    print('\n'.join(map(str,L[::-1])))
```
# [Reverse a linked list](https://www.hackerrank.com/challenges/reverse-a-linked-list/problem) - 5.0 - python3
```python
# Complete the reverse function below.

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
#
#
def reverse(head):
    if head == None or head.next == None: return head
    remaining = reverse(head.next)
    head.next.next = head
    head.next = None
    return remaining
```
# [Compare two linked lists](https://www.hackerrank.com/challenges/compare-two-linked-lists/problem) - 5.0 - python3
```python
# Complete the compare_lists function below.

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
#
#
def compare_lists(ll1, ll2):
    if ll1 == None and ll2 == None: return 1
    elif ll1 == None or ll2 == None: return 0
    elif ll1.data == ll2.data: return compare_lists(ll1.next, ll2.next)
    return 0
```
# [Print the Elements of a Linked List](https://www.hackerrank.com/challenges/print-the-elements-of-a-linked-list/problem) - 5.0 - python3
```python
# Complete the printLinkedList function below.

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
#
#
def printLinkedList(head):
    while head != None:
        print(head.data)
        head = head.next
```
# [Merge two sorted linked lists](https://www.hackerrank.com/challenges/merge-two-sorted-linked-lists/problem) - 5.0 - python3
```python
# Complete the mergeLists function below.

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
#
#
sys.setrecursionlimit(40000)

def mergeLists(head1, head2):
    if head1 == None: return head2
    elif head2 == None: return head1
    if head1.data > head2.data:
        temp = head2
        head2 = head2.next
        temp.next = head1
        head1 = temp
    head1.next = mergeLists(head1.next,head2)
    return head1
```
# [Get Node Value](https://www.hackerrank.com/challenges/get-the-value-of-the-node-at-a-specific-position-from-the-tail/problem) - 5.0 - python3
```python
# Complete the getNode function below.

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
#
#
def getNode(head, positionFromTail):
    current = head
    data = []
    while current != None:
        data.append(current.data)
        current = current.next
    return data[-positionFromTail-1]
```
# [Delete duplicate-value nodes from a sorted linked list](https://www.hackerrank.com/challenges/delete-duplicate-value-nodes-from-a-sorted-linked-list/problem) - 5.0 - python3
```python
# Complete the removeDuplicates function below.

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
#
#
def removeDuplicates(head):
    if head == None: return None
    current = last = head
    while current != None:
        if current.data == last.data:
            last.next = current.next
        else:
            last = current
        current = current.next
    return head
```
# [Find Merge Point of Two Lists](https://www.hackerrank.com/challenges/find-the-merge-point-of-two-joined-linked-lists/problem) - 5.0 - python3
```python
# Complete the findMergeNode function below.

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
#
#
def findMergeNode(head1, head2):
    curr1, curr2 = head1, head2
    while curr1 != None:
        while curr2 != None:
            if curr1 == curr2: return curr1.data
            curr2 = curr2.next
        curr2 = head2
        curr1 = curr1.next
```
# [Say "Hello, World!" With Python](https://www.hackerrank.com/challenges/py-hello-world/problem) - 5.0 - python3
```python
print("Hello, World!")
```
# [Reverse a doubly linked list](https://www.hackerrank.com/challenges/reverse-a-doubly-linked-list/problem) - 5.0 - python3
```python
# Complete the reverse function below.

#
# For your reference:
#
# DoublyLinkedListNode:
#     int data
#     DoublyLinkedListNode next
#     DoublyLinkedListNode prev
#
#
def reverse(head):
    p,last = head,None
    while p!=None:
        p.prev,p.next = p.next,p.prev
        last,p = p,p.prev
    return last
```
# [Inserting a Node Into a Sorted Doubly Linked List](https://www.hackerrank.com/challenges/insert-a-node-into-a-sorted-doubly-linked-list/problem) - 5.0 - python3
```python
# Complete the sortedInsert function below.

#
# For your reference:
#
# DoublyLinkedListNode:
#     int data
#     DoublyLinkedListNode next
#     DoublyLinkedListNode prev
#
#
def sortedInsert(head, data):
    p,last = head,None
    node = DoublyLinkedListNode(data)
    while p and p.data < data:
        p,last = p.next,p
    if last: last.next = node
    if p: p.prev = node
    node.prev,node.next = last,p
    return head if last else node
```
# [Cycle Detection](https://www.hackerrank.com/challenges/detect-whether-a-linked-list-contains-a-cycle/problem) - 5.0 - python3
```python
# Complete the has_cycle function below.

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
#
#
def has_cycle(head):
    if not head: return head
    s,f=head,head
    while f and f.next:
        s,f=s.next,f.next.next
        if s==f: return 1
    return 0
```
# [Matching Specific String](https://www.hackerrank.com/challenges/matching-specific-string/problem) - 5.0 - python
```
Regex_Pattern = r'hackerrank'	# Do not delete 'r'.
```
# [Matching Anything But a Newline](https://www.hackerrank.com/challenges/matching-anything-but-new-line/problem) - 5.0 - python
```
regex_pattern = r"^...\....\....\....$"	# Do not delete 'r'.
```
# [Matching Digits & Non-Digit Characters](https://www.hackerrank.com/challenges/matching-digits-non-digit-character/problem) - 5.0 - python
```
Regex_Pattern = r"\d\d\D\d\d\D\d\d\d\d"	# Do not delete 'r'.
```
# [Matching Whitespace & Non-Whitespace Character](https://www.hackerrank.com/challenges/matching-whitespace-non-whitespace-character/problem) - 5.0 - python
```
Regex_Pattern = r"\S\S\s\S\S\s\S\S"	# Do not delete 'r'.
```
# [Matching Word & Non-Word Character](https://www.hackerrank.com/challenges/matching-word-non-word/problem) - 5.0 - python
```
Regex_Pattern = r"\w\w\w\W\w\w\w\w\w\w\w\w\w\w\W\w\w\w"	# Do not delete 'r'.
```
# [Matching Start & End](https://www.hackerrank.com/challenges/matching-start-end/problem) - 5.0 - python
```
Regex_Pattern = r"^\d\w\w\w\w\.$"	# Do not delete 'r'.
```
# [Comparing Numbers](https://www.hackerrank.com/challenges/bash-tutorials---comparing-numbers/problem) - 3.0 - bash
```bash
read x
read y
if [[ $x -lt $y ]]; then
    echo "X is less than Y"
fi
if [[ $x -eq $y ]]; then
    echo "X is equal to Y"
fi
if [[ $x -gt $y ]]; then
    echo "X is greater than Y"
fi
```
# [Getting started with conditionals](https://www.hackerrank.com/challenges/bash-tutorials---getting-started-with-conditionals/problem) - 3.0 - bash
```bash
read c
if [[ $c == 'y' ]] || [[ $c == 'Y' ]]; then
    echo 'YES'
fi
if [[ $c == 'n' ]] || [[ $c == 'N' ]]; then
    echo 'NO'
fi
```
# [More on Conditionals](https://www.hackerrank.com/challenges/bash-tutorials---more-on-conditionals/problem) - 3.0 - bash
```bash
read x
read y
read z
if [[ x -eq y ]]&&[[ y -eq z ]]; then
    echo "EQUILATERAL"
elif [[ x -eq y ]]||[[ y -eq z ]]||[[ x -eq z ]]; then
    echo "ISOSCELES"
else
    echo "SCALENE"
fi
```
# [Middle of a Text File](https://www.hackerrank.com/challenges/text-processing-in-linux---the-middle-of-a-text-file/problem) - 3.0 - bash
```bash
sed -n '12,22p'
```
# [Looping and Skipping](https://www.hackerrank.com/challenges/bash-tutorials---looping-and-skipping/problem) - 2.0 - bash
```bash
for i in {1..99}
do
    if [[ $(($i % 2)) -eq 1 ]]; then
        echo "$i"
    fi
done
```
# [Looping with Numbers](https://www.hackerrank.com/challenges/bash-tutorials---looping-with-numbers/problem) - 2.0 - bash
```bash
for i in {1..50}
do
    echo $i
done
```
# [The World of Numbers](https://www.hackerrank.com/challenges/bash-tutorials---the-world-of-numbers/problem) - 2.0 - bash
```bash
read x
read y
echo $((x+y))
echo $((x-y))
echo $((x*y))
echo $((x/y))
```
# [Head of a Text File #1](https://www.hackerrank.com/challenges/text-processing-head-1/problem) - 2.0 - bash
```bash
head -n20
```
# [Head of a Text File #2](https://www.hackerrank.com/challenges/text-processing-head-2/problem) - 2.0 - bash
```bash
head -c20
```
# [Tail of a Text File #1](https://www.hackerrank.com/challenges/text-processing-tail-1/problem) - 2.0 - bash
```bash
tail -n20
```
# [Tail of a Text File #2](https://www.hackerrank.com/challenges/text-processing-tail-2/problem) - 2.0 - bash
```bash
tail -c20
```
# [Paste - 3](https://www.hackerrank.com/challenges/paste-3/problem) - 2.0 - bash
```bash
paste -s
```
# [Paste - 4](https://www.hackerrank.com/challenges/paste-4/problem) - 2.0 - bash
```bash
paste -sd $'\t\t\n'
```
# [Paste - 1](https://www.hackerrank.com/challenges/paste-1/problem) - 2.0 - bash
```bash
paste -sd';'
```
# [Paste - 2](https://www.hackerrank.com/challenges/paste-2/problem) - 2.0 - bash
```bash
paste -sd$';;\n'
```
# [Solve Me First](https://www.hackerrank.com/challenges/solve-me-first/problem) - 1.0 - python3
```python
def solveMeFirst(a,b):
    # Hint: Type return a+b below
    return a+b


num1 = int(input())
num2 = int(input())
res = solveMeFirst(num1,num2)
print(res)
```
# [Let's Echo](https://www.hackerrank.com/challenges/bash-tutorials-lets-echo/problem) - 1.0 - bash
```bash
echo 'HELLO'
```
# [A Personalized Echo](https://www.hackerrank.com/challenges/bash-tutorials---a-personalized-echo/problem) - 1.0 - bash
```bash
read name
echo "Welcome $name"
```
# [Cut #1](https://www.hackerrank.com/challenges/text-processing-cut-1/problem) - 1.0 - bash
```bash
cut -c3
```
# [Cut #2](https://www.hackerrank.com/challenges/text-processing-cut-2/problem) - 1.0 - bash
```bash
cut -c2,7
```
# [Cut #3](https://www.hackerrank.com/challenges/text-processing-cut-3/problem) - 1.0 - bash
```bash
cut -c2-7
```
# [Cut #4](https://www.hackerrank.com/challenges/text-processing-cut-4/problem) - 1.0 - bash
```bash
cut -c-4
```
# [Cut #5](https://www.hackerrank.com/challenges/text-processing-cut-5/problem) - 1.0 - bash
```bash
cut -f-3
```
# [Cut #6](https://www.hackerrank.com/challenges/text-processing-cut-6/problem) - 1.0 - bash
```bash
cut -c13-
```
# [Cut #7](https://www.hackerrank.com/challenges/text-processing-cut-7/problem) - 1.0 - bash
```bash
cut -d' ' -f4
```
# [Cut #8](https://www.hackerrank.com/challenges/text-processing-cut-8/problem) - 1.0 - bash
```bash
cut -d' ' -f-3
```
# [Cut #9](https://www.hackerrank.com/challenges/text-processing-cut-9/problem) - 1.0 - bash
```bash
cut -f2-
```
# ['Tr' Command #1](https://www.hackerrank.com/challenges/text-processing-tr-1/problem) - 1.0 - bash
```bash
tr '()' '[]'
```
# ['Tr' Command #2](https://www.hackerrank.com/challenges/text-processing-tr-2/problem) - 1.0 - bash
```bash
tr -d '[:lower:]'
```
# ['Tr' Command #3](https://www.hackerrank.com/challenges/text-processing-tr-3/problem) - 1.0 - bash
```bash
tr -s ' '
```
# [Sort Command #1](https://www.hackerrank.com/challenges/text-processing-sort-1/problem) - 1.0 - bash
```bash
sort
```
# [Sort Command #2](https://www.hackerrank.com/challenges/text-processing-sort-2/problem) - 1.0 - bash
```bash
sort -r
```
# [Sort Command #3](https://www.hackerrank.com/challenges/text-processing-sort-3/problem) - 1.0 - bash
```bash
sort -g
```
# [Sort Command #4](https://www.hackerrank.com/challenges/text-processing-sort-4/problem) - 1.0 - bash
```bash
sort -gr
```
# [Sort Command #5](https://www.hackerrank.com/challenges/text-processing-sort-5/problem) - 1.0 - bash
```bash
sort -t$'\t' -k2 -rn
```
# ['Sort' command #6](https://www.hackerrank.com/challenges/text-processing-sort-6/problem) - 1.0 - bash
```bash
sort -t$'\t' -k2 -n
```
# ['Sort' command #7](https://www.hackerrank.com/challenges/text-processing-sort-7/problem) - 1.0 - bash
```bash
sort -t'|' -k2 -nr
```
# ['Uniq' Command #1](https://www.hackerrank.com/challenges/text-processing-in-linux-the-uniq-command-1/problem) - 1.0 - bash
```bash
uniq
```
# ['Uniq' Command #2](https://www.hackerrank.com/challenges/text-processing-in-linux-the-uniq-command-2/problem) - 1.0 - bash
```bash
uniq -c | cut -c7-
```
# ['Uniq' command #3](https://www.hackerrank.com/challenges/text-processing-in-linux-the-uniq-command-3/problem) - 1.0 - bash
```bash
uniq -ci | cut -c7-
```
# ['Uniq' command #4](https://www.hackerrank.com/challenges/text-processing-in-linux-the-uniq-command-4/problem) - 1.0 - bash
```bash
uniq -u
```
