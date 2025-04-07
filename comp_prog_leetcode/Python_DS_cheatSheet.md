Great question! Here's a **cheat sheet of must-know operations** on common Python data structures— **strings, lists, sets, dictionaries, and more** —that are  **super helpful for LeetCode or competitive programming** .

---

### ✅ **1. String Operations**

| Operation           | Code                                       | Description                                 |
| ------------------- | ------------------------------------------ | ------------------------------------------- |
| Length              | `len(s)`                                 | Get length of string                        |
| Access char         | `s[i]`                                   | Get i-th character                          |
| Slice               | `s[i:j]`                                 | Substring from i to j-1                     |
| Reverse             | `s[::-1]`                                | Reverse string                              |
| Join                | `"".join(list)`                          | Merge list to string                        |
| Split               | `s.split()`                              | Split string by space (or `s.split(',')`) |
| Find                | `s.find("ab")`                           | First index of substring                    |
| Count               | `s.count("a")`                           | Number of times substring occurs            |
| Replace             | `s.replace("a", "b")`                    | Replace all "a" with "b"                    |
| Check prefix/suffix | `s.startswith("ab")`,`s.endswith("z")` | Boolean check                               |

---

### ✅ **2. List Operations**

| Operation          | Code                              | Description                    |
| ------------------ | --------------------------------- | ------------------------------ |
| Length             | `len(lst)`                      | Number of elements             |
| Append             | `lst.append(x)`                 | Add at end                     |
| Pop                | `lst.pop()`/`lst.pop(i)`      | Remove last / i-th element     |
| Insert             | `lst.insert(i, x)`              | Insert x at index i            |
| Slice              | `lst[i:j]`                      | Sub-list                       |
| Reverse            | `lst[::-1]`or `lst.reverse()` | Reverse list                   |
| Sort               | `lst.sort()`/`sorted(lst)`    | In-place or return sorted list |
| Sum / Max / Min    | `sum(lst)`,`max(lst)`         | Aggregates                     |
| Remove             | `lst.remove(x)`                 | Remove first occurrence of x   |
| List comprehension | `[x*x for x in lst]`            | Fast transformation            |

---

### ✅ **3. Set Operations**

| Operation    | Code             | Description                            |
| ------------ | ---------------- | -------------------------------------- |
| Add          | `s.add(x)`     | Insert element                         |
| Remove       | `s.remove(x)`  | Remove element (KeyError if not exist) |
| Discard      | `s.discard(x)` | Safe remove (no error)                 |
| Union        | `s1              | s2 `or`s1.union(s2)`                 |
| Intersection | `s1 & s2`      | Common elements                        |
| Difference   | `s1 - s2`      | Elements in s1 not s2                  |
| Membership   | `x in s`       | Check if exists                        |
| Length       | `len(s)`       | Number of unique elements              |

---

### ✅ **4. Dictionary (Hash Map) Operations**

| Operation             | Code                                      | Description             |
| --------------------- | ----------------------------------------- | ----------------------- |
| Access                | `d[key]`                                | Get value by key        |
| Insert/Update         | `d[key] = val`                          | Add or update entry     |
| Get with default      | `d.get(key, default)`                   | Safer access            |
| Keys / Values / Items | `d.keys()`,`d.values()`,`d.items()` | Useful in loops         |
| Delete                | `del d[key]`                            | Remove entry            |
| Check key             | `key in d`                              | Membership test         |
| Loop                  | `for k, v in d.items():`                | Iterate key-value pairs |

---

### ✅ **5. Tuple Operations**

| Operation         | Code                                 | Description               |
| ----------------- | ------------------------------------ | ------------------------- |
| Create            | `t = (1, 2)`                       | Immutable list            |
| Access            | `t[0]`                             | Indexing works like lists |
| Unpack            | `a, b = t`                         | Decompose tuple           |
| In list of tuples | `sorted(data, key=lambda x: x[1])` | Often used in sorting     |

---

### ✅ **6. General Techniques for LeetCode**

| Technique           | Example                                | Usage                                           |
| ------------------- | -------------------------------------- | ----------------------------------------------- |
| 2-pointer           | `while left < right:`                | Used in sorted arrays or palindromes            |
| Sliding window      | `while window_size < k:`             | Fixed/variable window size                      |
| Hashing             | `set()`,`dict()`                   | For fast lookups                                |
| Stack               | `stack = []`                         | Track previous elements (e.g., monotonic stack) |
| BFS/DFS             | `queue`,`recursion`,`visited`set | Graph traversal                                 |
| Binary Search       | `while low <= high:`                 | Sorted arrays/search spaces                     |
| Greedy              | `sort first`+ decision logic         | Optimal local decisions                         |
| Dynamic Programming | `dp[i] = ...`                        | Optimal substructure + overlapping subproblems  |

---

### 📌 Bonus: Common Patterns

* **Check palindrome** : `s == s[::-1]`
* **Flatten nested list** : `[item for sub in nested for item in sub]`
* **Count freq** : `Counter(arr)`
* **Sort by custom rule** : `sorted(arr, key=lambda x: x[1])`
* **Find unique items** : `set(arr)`


Here’s the full **Python cheat sheet for LeetCode-style problems** with a **code example column** added for each utility:

---

### ✅ Python Utility Cheat Sheet for LeetCode

| Module          | Utility         | Description              | Code Sample                               |
| --------------- | --------------- | ------------------------ | ----------------------------------------- |
| `collections` | `defaultdict` | Dict with default values | `d = defaultdict(int); d['a'] += 1`     |
| Imp             | `Counter`     | Count elements           | `Counter([1,1,2,3])  # {1:2, 2:1, 3:1}` |
|                 |                 |                          |                                           |
|                 |                 |                          |                                           |

---

| Module    | Utility                    | Description         | Code Sample                                     |
| --------- | -------------------------- | ------------------- | ----------------------------------------------- |
| `heapq` | `heappush`/`heappop`   | Min heap ops        | `heap = []; heappush(heap, 2); heappop(heap)` |
|           | Max heap (trick)           | Use negative values | `heappush(heap, -val); -heappop(heap)`        |
|           | `heapify`                | List to heap        | `heap = [3,1,4]; heapify(heap)`               |
|           | `nlargest`/`nsmallest` | Top k values        | `heapq.nlargest(2, [4,1,3,2])`                |

---

| Module     | Utility                   | Description                 | Code Sample                                 |
| ---------- | ------------------------- | --------------------------- | ------------------------------------------- |
| `bisect` | `bisect_left`/`right` | Insert pos in sorted list   | `bisect.bisect_left([1,3,4], 2)`          |
|            | `insort_left`           | Insert while keeping sorted | `arr = [1,3]; bisect.insort_left(arr, 2)` |

---

| Module        | Utility          | Description       | Code Sample                              |
| ------------- | ---------------- | ----------------- | ---------------------------------------- |
| `itertools` | `combinations` | r-length combos   | `list(combinations([1,2,3], 2))`       |
|               | `permutations` | r-length perms    | `list(permutations([1,2], 2))`         |
|               | `product`      | Cartesian product | `list(product([1,2], repeat=2))`       |
|               | `accumulate`   | Running sums      | `list(accumulate([1,2,3]))  # [1,3,6]` |

---

| Module   | Utility            | Description         | Code Sample                         |
| -------- | ------------------ | ------------------- | ----------------------------------- |
| `math` | `gcd`,`lcm`    | Math operations     | `math.gcd(8, 12)  # 4`            |
|          | `isqrt`          | Integer square root | `math.isqrt(10)  # 3`             |
|          | `ceil`,`floor` | Rounding            | `math.ceil(1.2), math.floor(1.8)` |

---

| Module        | Utility        | Description      | Code Sample                                                |
| ------------- | -------------- | ---------------- | ---------------------------------------------------------- |
| `functools` | `lru_cache`  | Memoization      | `@lru_cache(None); def fib(n): return fib(n-1)+fib(n-2)` |
|               | `cmp_to_key` | Custom sorting   | `sorted(arr, key=cmp_to_key(cmp_fn))`                    |
|               | `reduce`     | Combine iterable | `reduce(lambda x,y: x+y, [1,2,3])  # 6`                  |

---

| Module       | Utility         | Description    | Code Sample                            |
| ------------ | --------------- | -------------- | -------------------------------------- |
| `operator` | `itemgetter`  | Sort by index  | `sorted(data, key=itemgetter(1))`    |
|              | `add`,`mul` | Functional ops | `reduce(operator.mul, [1,2,3])  # 6` |

---

| Module     | Utility                            | Description   | Code Sample                                                                          |
| ---------- | ---------------------------------- | ------------- | ------------------------------------------------------------------------------------ |
| `random` | `choice`,`shuffle`,`randint` | Random utils  | `random.random() # given 0,1 uniform; random.choice([1,2,3]); random.shuffle(arr)` |
| `string` | `ascii_lowercase`/`digits`     | Alphabet/nums | `string.ascii_lowercase[:5]  # 'abcde'`                                            |
| `time`   | `time.time()`                    | Time a block  | `start = time.time(); ...; print(time.time() - start)`                             |

---

Let me know if you want this as a downloadable Markdown or PDF or want LeetCode problem examples using these!
