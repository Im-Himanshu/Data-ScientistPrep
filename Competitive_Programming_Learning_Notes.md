For Revision of Competitive Programming refer this file [Revise Competitive Programming](./Revise_CompetiveProgramming.md)
In this I am discussing soft skills and tricks that I have learned in my experience.

# Important Lessons
1. **Read the Question Carefully**
   - Understand what is expected from the problem.
   - Example: Today I tackled the [Reverse Sub Singly-Linked List](https://leetcode.com/problems/reverse-linked-list-ii/?envType=problem-list-v2&envId=linked-list). I overlooked the boundary cases, which changed the whole solving criterion.

2. **Draft Linked List and Graph Questions**
   - Write the operations on paper before coding.
   - Clearly outline the line-by-line operations to avoid confusion during implementation.

3. use defaultdict for dictionary and hashmap in python
   - Default dict doesn't raise keyError when it is not present so lots of boundary cases are handled by default 
   - With the `{}` type default dict in python we always have to search over keys if the given key is present or not, which makes computation time higher.
---

### Tricks for Linked Lists

1. **Use Dummy Nodes**
   - Create a dummy head and tail node to simplify boundary case handling.
   - This makes the process easier; for singly linked lists, the previous pointer doesn't need to be managed.
   - This approach elegantly handles cases where the linked list is empty or has only one element:
     - Start the loop with `current.next is None` and use `current = current.next` to skip the dummy node.
   - Always delete the connection created to the dummy node at the end of the code to avoid any loop in the code - refer [problem-linkedlist hard -1](./Problem_solving_logs.md) 

2. **Avoid Index-Based Iteration**
   - Do not run an outer loop based on a fixed number of iterations with next operations.
   - Looping in a linked list should be based on the parameter being worked upon or a fixed index only.

3. **Sorting Linked Lists**
   - If sorting is needed, convert values to an array and use `arr.sort()`. This is more space complex but simpler than implementing a sort from scratch.
   - When implementing sorting, use:
     - **Merge Sort** for time complexity.
     - **Bubble Sort** for quicker coding time.
   - There is an implementation of linked list sorting in [code-samples/linked_list.py](./code_samples/linked_list.py) of this directory.

4. **Debugging Tips**
   - Linked lists can be tricky to debug; even a single logical error can cause issues.
   - Example: The above linked list merge sorting took 15 minutes to implement and 30 minutes to debug.

---

### Sorting Algorithms

1. **Bubble Sort** 
   - Time Complexity: O(n²), worst case.
   - Easy to implement.
   - Procedure: Traverse from left to right, moving the largest element to the end in each loop.
   - Suitable for single linked lists due to its one-directional traversal.

2. **Insertion Sort** 
   - Time Complexity: O(n²), worst case.
   - Easy to implement.
   - Procedure: Travel from the ith to 0 for increasing i, creating a sorted array on the left.
   - Not feasible for single linked lists due to reverse traversal requirement.

3. **Merge Sort** 
   - Time Complexity: O(n log n), divide and conquer approach.
   - Common and moderately complex to implement.
   - Procedure: Divide the list into two halves, recursively call Merge Sort on each half, and merge the two sorted lists.
   - Requires recursive calls, which can take additional memory.

---

## Python Tricks

1. **Using Heapq for Dynamic Lists**
   - For programs requiring sorting or top-k structures, use built-in functions like `heapq` for max/min operations.
   - [HeapQ Detailed Notes](./heapq_python_tutorial.md)
   - Be creaful if the first value of tuple is tie it jumps to second value so pass a unique value as second value.

   ```python
   # Heapq general usage 
   import heapq
   i =0
   # Assuming Node is a class
   nodes = [
       (3, i:= i+1, Node("A", 3)), # need to careful for ties on priority it will throw error
       (1, i:= i+1, Node("B", 1)),
       (4, i:= i+1, Node("C", 4)),
       (2, i:= i+1, Node("D", 2))
   ]  # Using tuple (priority, object)
   
   heapq.heapify(nodes)  # Transform list into a heap
   
   # Push a new Node into the heap
   heapq.heappush(nodes, (0, i:= i+1, Node("E", 0)))

   
   popped_node = heapq.heappop(nodes)  # Get the smallest node
   largest_nodes = heapq.nlargest(2, nodes)  # Get the 2 largest nodes
   smallest_nodes = heapq.nsmallest(2, nodes)  # Get the 2 smallest nodes
   ```
2. **Using defaultdict from collection for dictionary**

```python
from collections import defaultdict
# if someone tries to access a element that doesn't exist it return default value
# so the initiation of key-value can be very seamless 

dicti = defaultdict(int)
dicti[3] +=1 # till now key 3 doesn't exist so it throws 0 and add one to it


# Default value 0 (int() returns 0)
int_default = defaultdict(int)
print(int_default['a'])  # Output: 0

# Default value is an empty list
list_default = defaultdict(list)
print(list_default['b'])  # Output: []

# Default value is an empty dict
dict_default = defaultdict(dict)
print(dict_default['c'])  # Output: {}

# Custom default value (function returning 42)
def custom_value(): return 42
custom_default = defaultdict(custom_value)
print(custom_default['x'])  # Output: 42

# Using lambda to set a custom default value
lambda_default = defaultdict(lambda: 100)
print(lambda_default['y'])  # Output: 100

# Default value as an empty set
set_default = defaultdict(set)
print(set_default['z'])  # Output: set()

# Nested defaultdict with int as inner default
nested_default = defaultdict(lambda: defaultdict(int))
print(nested_default['a']['b'])  # Output: 0
```
```