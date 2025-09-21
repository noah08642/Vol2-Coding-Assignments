# Coding Interview Guide Sheet



## Quick decision tree (one-pass)
1. What's the input size `n`? If `n <= 1e3` \(O(n^2)\) might be OK; if `n ≈ 1e5` aim for `O(n log n)` or `O(n)`.
2. Are data elements **ordered/sorted** or can be sorted? → Sorting or two-pointers.
3. Are you asked about **paths/connectedness**? → Graphs (BFS/DFS/Dijkstra).
4. Does the problem ask for **max/min over sliding subarrays**? → Sliding window / monotonic queue.
5. Do you need **optimal value under constraints**? → Greedy or DP (knapsack / interval scheduling / scheduling with deadlines).
6. Are answers combinatorial (perm/subsets)? → Backtracking/bitmask DP.

---



## II. Arrays & Two-Pointers
**Use when:** sorted arrays, pair-sum, remove duplicates in-place.
**Clues:** “closest pair”, “move zeros”.


**Two-sum (sorted)**
```python
i, j = 0, n-1
while i < j:
    s = arr[i] + arr[j]
    if s == target: return (i,j)
    if s < target: i += 1
    else: j -= 1
```

**Remove duplicates (in-place)**
```python
write = 0
for read in range(n):
    if read == 0 or arr[read] != arr[read-1]:
        arr[write] = arr[read]; write += 1
# length = write
```

**Example:**
> *Question:* Given a sorted array and a target value, find the indices of two numbers that sum to the target.
> *Answer:* Use two pointers from both ends. Complexity O(n).

---

## III. Sliding Window & Monotonic Queue
**Use when:** contiguous subarray, max/min in windows, smallest window containing something.
**Clues:** “longest substring”, “at most K distinct”, “max in subarray of size k”, “shortest window”.

**Fixed-size window sum / average**
```python
cur = sum(arr[:k])
best = cur
for i in range(k, n):
    cur += arr[i] - arr[i-k]
    best = max(best, cur)
```

**Alternate:**
```python
cur = 0; best = 0; left = 0
for right, val in enumerate(arr):
    cur += val
    while condition_not_met:
        cur -= arr[left]; left += 1
    best = max(best, cur)
```

**Minimum window substring (general pattern)**
1. Expand right until window valid. 2. Contract left to shrink while valid. 3. Track best.

**Monotonic deque (max in window)**
```python
from collections import deque
d = deque()
for i in range(n):
    while d and arr[d[-1]] <= arr[i]: d.pop()
    d.append(i)
    if d[0] <= i - k: d.popleft()
    if i >= k-1: result.append(arr[d[0]])
```

**Example:**
> *Question:* Find the maximum sum of any contiguous subarray of length k.
> *Answer:* Use a fixed-size sliding window; update sum in O(1).
---

## IV. Prefix Sums & Differences
**Use when:** Need to compute range sums, count subarrays with a specific property, or handle cumulative aggregates efficiently.  
**Clues:** Problems mentioning "subarray sum equals K", "range queries", "count intervals with sum divisible by K", or "cumulative sums".

### Prefix Sum Array
**Purpose:** Compute the sum of elements in a range `[i, j]` in O(1) time after O(n) preprocessing.  
**How it works:** Build a prefix sum array where `pref[i+1]` stores the sum of elements from `arr[0]` to `arr[i]`. The sum of range `[i, j]` is `pref[j+1] - pref[i]`.  
**Example:** Given `arr = [1, 2, 3, 4]`, compute the sum of elements from index 1 to 3 (i.e., `[2, 3, 4]`).  
```python
def create_prefix_sum(arr):
    n = len(arr)
    pref = [0] * (n + 1)
    for i in range(n):
        pref[i + 1] = pref[i] + arr[i]
    return pref

# Example usage
arr = [1, 2, 3, 4]
pref = create_prefix_sum(arr)  # pref = [0, 1, 3, 6, 10]
# Sum of arr[1:4] (indices 1 to 3) = pref[4] - pref[1] = 10 - 1 = 9
```

### Subarray Sum Equals K (Hashmap)
**Purpose:** Count the number of subarrays with a sum equal to `K`.  
**How it works:** Use a hashmap to store the frequency of cumulative sums. For each index, compute the cumulative sum and check if `cur_sum - K` exists in the hashmap to find valid subarrays.  
**Example:** Given `arr = [1, 1, 1]`, `K = 2`, find how many subarrays have a sum of 2.  
```python
def subarray_sum(arr, K):
    count = 0
    cur_sum = 0
    seen = {0: 1}  # Initialize with 0 sum having frequency 1
    for x in arr:
        cur_sum += x
        count += seen.get(cur_sum - K, 0)  # Add count of subarrays ending here
        seen[cur_sum] = seen.get(cur_sum, 0) + 1  # Update frequency of current sum
    return count

# Example usage
arr = [1, 1, 1]
K = 2
print(subarray_sum(arr, K))  # Output: 2 (subarrays [1,1] at indices 0-1 and 1-2)
```

## V. Hashmap / Frequency Patterns
**Use when:** Problems involve grouping elements, counting frequencies, finding unique elements, top-k frequent items, or grouping anagrams.  
**Clues:** Keywords like "group by", "count occurrences", "most frequent", "first unique", or "anagram".

### Top-K Frequent Elements (Heap)
**Purpose:** Find the top `k` most frequent elements in an array.  
**How it works:** Use a `Counter` to count frequencies, then maintain a min-heap of size `k` to keep track of the top `k` elements by frequency.  
**Example:** Given `arr = [1, 1, 1, 2, 2, 3]`, `k = 2`, find the two most frequent elements.  
```python
from collections import Counter
import heapq

def top_k_frequent(arr, k):
    # Count frequencies
    counter = Counter(arr)
    
    # Use min-heap to track top k elements
    heap = []
    for val, freq in counter.items():
        if len(heap) < k:
            heapq.heappush(heap, (freq, val))
        else:
            if freq > heap[0][0]:
                heapq.heapreplace(heap, (freq, val))
    
    # Extract values from heap
    return [val for _, val in heap]

# Example usage
arr = [1, 1, 1, 2, 2, 3]
k = 2
print(top_k_frequent(arr, k))  # Output: [1, 2] (1 appears 3 times, 2 appears 2 times)
```

### Group Anagrams
**Purpose:** Group a list of strings into anagrams (words with the same characters but in different order).  
**How it works:** Use a hashmap where the key is the sorted version of each word, and the value is a list of words that share that sorted key.  
**Example:** Given `words = ["eat", "tea", "tan", "ate", "nat", "bat"]`, group anagrams together.  
```python
from collections import defaultdict

def group_anagrams(words):
    mp = defaultdict(list)
    for w in words:
        key = ''.join(sorted(w))  # Sort characters to create a unique key
        mp[key].append(w)        # Group words with the same sorted key
    return list(mp.values())

# Example usage
words = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(group_anagrams(words))  # Output: [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
```

## VI. Sorting & Two-way greedy
**Clues:** “merge intervals”, “meeting rooms”, “schedule max tasks”.

- Sort by key and then sweep. Classic for intervals and scheduling.



**Interval scheduling (max non-overlapping)**
```python
intervals.sort(key=lambda x: x[1])
last_end = -inf
count = 0
for s,e in intervals:
    if s >= last_end:
        count += 1; last_end = e
```


> *Question:* Merge overlapping intervals.
> *Answer:* Sort by start; merge as you scan.
**Merge intervals**
```python
intervals.sort()
merged = []
for s,e in intervals:
    if not merged or merged[-1][1] < s: merged.append([s,e])
    else: merged[-1][1] = max(merged[-1][1], e)
```

---
## VII. Binary Search (and Binary Search on Answer)
**Use when:** Need to find an element in a sorted array, locate the boundary of a condition, or optimize a monotonic parameter (e.g., minimize the maximum or maximize the minimum).  
**Clues:** Problems mentioning “find smallest/largest that satisfies…”, “minimize maximum”, “search space”, or “sorted array”.

### Standard Binary Search (Find First True)
**Purpose:** Find the smallest index (or value) where a condition becomes true in a monotonic search space.  
**How it works:** Narrow the search space by half each iteration, adjusting the boundaries based on whether the condition is satisfied. Returns the first index where the condition is true.  
**Example:** Given a sorted array `arr = [1, 2, 2, 2, 3]` and target `2`, find the first occurrence of `2`.  
```python
def bisect_first(arr, target):
    def check(mid):
        return arr[mid] >= target  # Condition: element is at least target
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if check(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

# Example usage
arr = [1, 2, 2, 2, 3]
target = 2
result = bisect_first(arr, target)
print(result)  # Output: 1 (first occurrence of 2 is at index 1)
```

### Binary Search on Answer
**Purpose:** Find the smallest/largest value of a parameter that satisfies a feasibility condition, often when the feasibility check is O(n) and the parameter is monotonic.  
**How it works:** Define a search space for the answer (e.g., possible capacities, times, or thresholds), and use binary search to find the optimal value by testing feasibility.  
**Example:** Given `packages = [1, 2, 3, 4, 5]` and `D = 2` days, find the minimum ship capacity to deliver all packages within D days.  
```python
def ship_within_days(packages, D):
    def can_ship(capacity):
        days = 1
        curr_sum = 0
        for w in packages:
            if w > capacity:  # Package too heavy for capacity
                return False
            if curr_sum + w > capacity:
                days += 1
                curr_sum = w
            else:
                curr_sum += w
        return days <= D

    lo, hi = max(packages), sum(packages)  # Min and max possible capacities
    while lo < hi:
        mid = (lo + hi) // 2
        if can_ship(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

# Example usage
packages = [1, 2, 3, 4, 5]
D = 2
print(ship_within_days(packages, D))  # Output: 6 (minimum capacity to ship in 2 days)
```

---

## VIII. Heaps / Priority Queue
**Use when:** Need to maintain a dynamic set of elements to efficiently retrieve the smallest/largest element, merge sorted lists, or track running medians.  
**Clues:** Problems mentioning “k largest/smallest”, “median online”, “merge k sorted lists”, or “top k elements”.

### Min-Heap and Max-Heap Basics
**Purpose:** Efficiently retrieve the smallest (min-heap) or largest (max-heap) element in O(log n) time.  
**How it works:** Use Python’s `heapq` module for a min-heap by default. For a max-heap, push the negative of values to simulate. Supports push and pop operations.  
**Example:** Push elements to a min-heap and pop the smallest element.  
```python
import heapq

def heap_example():
    heap = []
    elements = [3, 1, 4, 1, 5]
    for x in elements:
        heapq.heappush(heap, x)  # Push to min-heap
    smallest = heapq.heappop(heap)  # Pop smallest element
    return heap, smallest

# Example usage
heap, smallest = heap_example()
print(smallest, heap)  # Output: 1 [1, 3, 4, 5] (smallest is 1, remaining heap)
```

### K Smallest Elements (Max-Heap)
**Purpose:** Find the k smallest elements in an array efficiently.  
**How it works:** Maintain a max-heap of size k. If a new element is smaller than the heap’s largest element, replace it. This keeps the k smallest elements in the heap.  
**Example:** Given `arr = [3, 2, 1, 5, 6, 4]` and `k = 2`, find the 2 smallest elements.  
```python
import heapq

def k_smallest(arr, k):
    heap = []
    for x in arr:
        if len(heap) < k:
            heapq.heappush(heap, -x)  # Push negative for max-heap
        else:
            if -x > heap[0]:  # If x is smaller than largest in heap
                heapq.heapreplace(heap, -x)
    return [-x for x in heap]  # Convert back to positive

# Example usage
arr = [3, 2, 1, 5, 6, 4]
k = 2
print(k_smallest(arr, k))  # Output: [1, 2] (the 2 smallest elements)
```
## IX. Stack / Monotonic Stack
**Use when:** Need to find the next greater/smaller element, compute spans, or solve problems like largest rectangle in histogram where maintaining a monotonic property is useful.  
**Clues:** Problems mentioning “next greater element”, “stock span”, “largest rectangle”, or “nearest smaller/larger”.

### Next Greater Element (Right)
**Purpose:** For each element in an array, find the next element to its right that is greater than it. If none exists, return -1.  
**How it works:** Use a monotonic stack to maintain indices of elements in decreasing order. When a larger element is found, pop elements from the stack and assign the current element as their next greater element.  
**Example:** Given `arr = [1, 3, 2, 4]`, find the next greater element for each position.  
```python
def next_greater_element(arr):
    n = len(arr)
    res = [-1] * n  # Initialize result with -1
    stack = []      # Stack to store indices
    for i in range(n):
        while stack and arr[i] > arr[stack[-1]]:  # Current element is greater
            res[stack.pop()] = arr[i]             # Assign next greater
        stack.append(i)                           # Push current index
    return res

# Example usage
arr = [1, 3, 2, 4]
print(next_greater_element(arr))  # Output: [3, 4, 4, -1]
# Explanation: 1->3, 3->4, 2->4, 4->-1 (no greater element)
```

### Largest Rectangle in Histogram
**Purpose:** Find the largest rectangle area in a histogram where each bar has a height given in an array.  
**How it works:** Use a monotonic stack to store indices of bars in increasing height order. For each bar, compute the area of rectangles ending at that bar by popping bars with greater heights from the stack.  
**Example:** Given `heights = [2, 1, 5, 6, 2, 3]`, find the largest rectangle area.  
```python
def largest_rectangle_area(heights):
    stack = [-1]  # Sentinel for easier width calculation
    max_area = 0
    for i in range(len(heights) + 1):
        curr_height = 0 if i == len(heights) else heights[i]
        while stack[-1] != -1 and curr_height < heights[stack[-1]]:
            height = heights[stack.pop()]
            width = i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    return max_area

# Example usage
heights = [2, 1, 5, 6, 2, 3]
print(largest_rectangle_area(heights))  # Output: 10 (rectangle with height 5, width 2)
```

---

## X. Linked List Patterns
**Use when:** Manipulating linked lists, such as reversing, detecting cycles, merging sorted lists, or finding intersections.  
**Clues:** Problems mentioning “reverse in-place”, “detect cycle”, “merge sorted lists”, “find middle”, or “reorder list”.

### Reverse Linked List (Iterative)
**Purpose:** Reverse a singly linked list in-place.  
**How it works:** Iterate through the list, reversing the `next` pointers by keeping track of the previous and next nodes. The final `prev` node becomes the new head.  
**Example:** Given a linked list `1 -> 2 -> 3 -> None`, reverse it to `3 -> 2 -> 1 -> None`.  
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next    # Save next node
        curr.next = prev   # Reverse the link
        prev = curr        # Move prev forward
        curr = nxt         # Move curr forward
    return prev           # New head

# Example usage
def create_list(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    curr = head
    for val in arr[1:]:
        curr.next = ListNode(val)
        curr = curr.next
    return head

def print_list(head):
    vals = []
    while head:
        vals.append(head.val)
        head = head.next
    return vals

# Example
head = create_list([1, 2, 3])
reversed_head = reverse_list(head)
print(print_list(reversed_head))  # Output: [3, 2, 1]
```

## XI. Trees
**DFS recursive (preorder/inorder/postorder)**
```python
def dfs(node):
    if not node: return
    # preorder: visit(node)
    dfs(node.left)
    # inorder: visit(node)
    dfs(node.right)
    # postorder: visit(node)
```

**BFS (level order)**
```python
from collections import deque
q = deque([root])
while q:
    sz = len(q)
    for _ in range(sz):
        node = q.popleft()
        # visit
        if node.left: q.append(node.left)
        if node.right: q.append(node.right)
```

**Tree DP (postorder)**
- Many problems require computing values from children up. Use recursion and return value(s) to parent.

**Lowest Common Ancestor (binary lifting idea)**
- For simple tests, use parent pointers + depth climb; for multiples, precompute binary lift.

---

## XII. Graphs
**Clues:** “shortest path”, “connected components”, “topological order”, “cycle detection”.

**Example:**
> *Question:* Number of islands in a grid.
> *Answer:* DFS over grid cells marking visited.


**Adjacency list**: `adj = [[] for _ in range(n)]` and `adj[u].append((v, w))` for weighted.

**BFS (shortest in unweighted)**
```python
from collections import deque
dist = [-1]*n
q = deque([start]); dist[start]=0
while q:
    u = q.popleft()
    for v in adj[u]:
        if dist[v] == -1:
            dist[v] = dist[u] + 1; q.append(v)
```

**DFS (iterative)**
```python
stack = [start]
visited = set([start])
while stack:
    u = stack.pop()
    for v in adj[u]:
        if v not in visited:
            visited.add(v); stack.append(v)
```

**Dijkstra (weighted shortest)**
```python
import heapq
INF = 10**18
dist = [INF]*n
dist[src] = 0
heap = [(0, src)]
while heap:
    d,u = heapq.heappop(heap)
    if d != dist[u]: continue
    for v,w in adj[u]:
        nd = d + w
        if nd < dist[v]:
            dist[v] = nd; heapq.heappush(heap, (nd, v))
```

**Topological sort (Kahn)**
```python
from collections import deque
in_deg = [0]*n
for u in range(n):
    for v in adj[u]: in_deg[v]+=1
q = deque([i for i in range(n) if in_deg[i]==0])
order = []
while q:
    u = q.popleft(); order.append(u)
    for v in adj[u]:
        in_deg[v]-=1
        if in_deg[v]==0: q.append(v)
# if len(order) < n => cycle
```

**Union-Find (Disjoint Set Union)**
```python
parent = list(range(n))
rank = [0]*n
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a,b):
    ra, rb = find(a), find(b)
    if ra == rb: return False
    if rank[ra] < rank[rb]: ra,rb = rb,ra
    parent[rb] = ra
    if rank[ra] == rank[rb]: rank[ra] += 1
    return True
```

**Minimum Spanning Tree (Kruskal)**
- Sort edges by weight and union endpoints with DSU.

---

## XIII. Dynamic Programming (templates)
**Clues:** “optimal”, “minimum/maximum ways”, “partition”, “longest subsequence”.

**Example:**
> *Question:* Coin change: fewest coins for amount.
> *Answer:* Bottom-up DP.
```python
dp=[inf]*(amount+1); dp[0]=0
for c in coins:
    for a in range(c, amount+1):
        dp[a] = min(dp[a], dp[a-c]+1)
```


**Memoization (top-down)**
```python
from functools import lru_cache
@lru_cache(None)
def dp(state):
    if base_case: return value
    ans = -inf
    for choice in options:
        ans = max(ans, dp(next_state) + value)
    return ans
```

**Bottom-up 1D (e.g., 1D knapsack)**
```python
dp = [0]*(W+1)
for item in items:
    for w in range(W, item.weight-1, -1):
        dp[w] = max(dp[w], dp[w-item.weight] + item.value)
```

**LIS (n log n)**
```python
from bisect import bisect_left
tails = []
for x in arr:
    i = bisect_left(tails, x)
    if i == len(tails): tails.append(x)
    else: tails[i] = x
# length = len(tails)
```

**Longest Common Subsequence (DP table)**
```python
m,n = len(A), len(B)
dp = [[0]*(n+1) for _ in range(m+1)]
for i in range(m-1,-1,-1):
    for j in range(n-1,-1,-1):
        if A[i]==B[j]: dp[i][j] = 1 + dp[i+1][j+1]
        else: dp[i][j] = max(dp[i+1][j], dp[i][j+1])
```

**DP on trees**
- Use postorder recursion; return states (e.g., include/exclude node).

**Bitmask DP (n ≤ 20)**
```python
# dp[mask] = best value for subset mask
for mask in range(1<<n):
    for i in range(n):
        if not (mask & (1<<i)): continue
        prev = mask ^ (1<<i)
        dp[mask] = min(dp[mask], dp[prev] + cost(prev, i))
```

---

## XIV. Backtracking (Permutations / Subsets)
**Clues:** “all permutations”, “generate subsets”, “N-Queens”.

**Subsets**
```python
def backtrack(i, path):
    if i == n: res.append(path[:]); return
    # skip
    backtrack(i+1, path)
    # take
    path.append(arr[i]); backtrack(i+1, path); path.pop()
```

**Permutations (swap method)**
```python
def perm(i):
    if i==n: res.append(arr[:]); return
    for j in range(i,n):
        arr[i],arr[j] = arr[j],arr[i]
        perm(i+1)
        arr[i],arr[j] = arr[j],arr[i]
```

---

## XV. Strings
**Palindrome check**
```python
i,j = 0, len(s)-1
while i<j:
    if s[i]!=s[j]: return False
    i+=1; j-=1
return True
```

**KMP prefix function**
```python
def prefix(s):
    n=len(s); pi=[0]*n
    for i in range(1,n):
        j = pi[i-1]
        while j>0 and s[i]!=s[j]: j = pi[j-1]
        if s[i]==s[j]: j+=1
        pi[i] = j
    return pi
```

**Rabin-Karp (rolling hash) pattern**
- Compute rolling hash over window and compare (double-check by direct string comparison on hash hit).

**Trie skeleton**
```python
class TrieNode:
    def __init__(self):
        self.ch = {}
        self.end = False

root = TrieNode()
# insert/search straightforward
```

---

## XVI. Fenwick Tree (Binary Indexed Tree)
**Point update / prefix sum**
```python
class BIT:
    def __init__(self,n):
        self.n=n; self.bit=[0]*(n+1)
    def add(self,i,val):
        while i<=self.n:
            self.bit[i]+=val; i+= i & -i
    def sum(self,i):
        s=0
        while i>0:
            s+=self.bit[i]; i-= i & -i
        return s
```

---

## XVII. Segment Tree (point update / range query)
**Simple iterative segtree**
```python
# build
size = 1
while size < n: size <<= 1
tree = [0]*(2*size)
for i in range(n): tree[size+i] = arr[i]
for i in range(size-1,0,-1): tree[i] = tree[2*i] + tree[2*i+1]
# update
pos += size; tree[pos] = val
while pos>1: pos//=2; tree[pos] = tree[2*pos] + tree[2*pos+1]
# query(l,r)
```

---

## XVIII. Number Theory & Math
- `gcd(a,b)` Euclid, `lcm(a,b) = a//gcd*a`.
- Modular exponentiation (fast pow): use pow(x,y,mod) in Python.
- Sieve of Eratosthenes for primes ≤ N.
- Prime factorization via trial division up to sqrt(n) (for n ≤ 1e12 maybe optimized).

---

## XIX. Geometry (basics)
- Cross product: `(x1*y2 - x2*y1)` sign indicates orientation.
- Distance squared to avoid sqrt.
- Bounding box checks, point in polygon (ray-casting), convex hull (Graham scan / monotone chain template).

---

## XX. Templates for debugging & testing
- Always test: empty input, minimal size (1), max size, repeated values, negative values, sorted/unordered, random small cases.
- Use asserts for invariants while coding.
- Print intermediate only when debugging locally.

---

## XXI. Final checklist BEFORE submit
- Complexity matches constraints.
- Handles edge cases (empty, single element, duplicates).
- Uses appropriate integer types (avoid overflow in other languages).
- No unnecessary global state between test cases.
- Behavior for multiple test cases handled correctly.


