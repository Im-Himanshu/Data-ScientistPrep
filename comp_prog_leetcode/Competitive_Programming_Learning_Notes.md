For Revision of Competitive Programming refer this file [Revise Competitive Programming](../Revise_CompetiveProgramming.md)
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
   - Always delete the connection created to the dummy node at the end of the code to avoid any loop in the code - refer [problem-linkedlist hard -1](Problem_solving_logs.md) 

2. **Avoid Index-Based Iteration**
   - Do not run an outer loop based on a fixed number of iterations with next operations.
   - Looping in a linked list should be based on the parameter being worked upon or a fixed index only.

3. **Sorting Linked Lists**
   - If sorting is needed, convert values to an array and use `arr.sort()`. This is more space complex but simpler than implementing a sort from scratch.
   - if it comes to that and have to implement sorting, use:
     - **Merge Sort** for time complexity.
     - **Bubble Sort** for quicker coding time.
   - There is an implementation of linked list sorting in [code-samples/linked_list.py](code_samples/linked_list.py) of this directory.

4. **Debugging Tips**
   - Linked lists can be tricky to debug; even a single logical error can cause issues.
   - Example: The above linked list merge sorting took 15 minutes to implement and 30 minutes to debug.

5. **Detecting Cycles**
   - Use slow pointer (step=1) and fast pointer (step=2) and if they meet somepoint there is a cycle.
   - Argument is their relative speed is = 1 and let say cycle size is k, they must be at same location after atmost k-step after entering cycle.
   - If fast pointer reach Null before that means no cycle

---
## Tricks for Trees Question

1. **Tree Traversal**
   - Understand the difference between **Inorder**, **Preorder**, and **Postorder** traversals.
   - Inorder: Left, Root, Right
   - Preorder: Root, Left, Right
   - Postorder: Left, Right, Root
   - Implement these traversals recursively and iteratively.
   - For iterative traversal, use a stack to store the nodes.

Question of tree traversal can get complex implementing the iterative version of it, so always try to implement the recursive version first and then move to iterative version.
Moreover some recursion can get tricky because of return statement and maintaining some global state. 
Use self.<somthing> in the recusrive function itself to maintain the answer and return it at the end of the function.

For example in the example below, I have stored the self.ans because the required output is completely different from return of the function.
it makes it unnecessary complicated to return the value just to reach the end of the function call stack.
```python
#https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree
class Solution:

    def dfs_rec(self, root, p, q):
        
        if root is None:
            return False
        
        left = self.dfs_rec(root.left, p,q)
        right = self.dfs_rec(root.right, p, q)
        mid = root==p or root ==q

        if mid + left+right >=2: # only one node will have this 
            self.ans = root
        
        return left or right or mid

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        self.dfs_rec(root, p, q)
        return self.ans


# second version where I didn't use self.ans and returned the value from the function itself
    def print_util(self, node_arr):
        value = ", ".join([str(node.val) for node in node_arr])
        print(value)

    def DFS_rec(self, root, p, q, crnt_path, first_path):
        crnt_path.append(root)
        if root.val == p.val or root.val == q.val:
            if len(first_path) == 0:
                for node in crnt_path:
                    first_path.append(node)                
            else: # means both node are discovered return the last common ancestor in the array
                prev = None
                max_i = min(len(first_path), len(crnt_path))
                for i in range(max_i):
                    node1 = crnt_path[i]
                    node2 = first_path[i]
                    if node1 != node2:
                        break
                    else:
                        prev = node1 # should be equal to node2, root should always be there
                return prev # this is the node we were looking for
        # if not equal keep searching
        
        found_node = None
        if root.left is not None:
            found_node = self.DFS_rec(root.left, p, q, crnt_path, first_path)
        if found_node is not None:
            return found_node
        
        if root.right is not None:
            found_node = self.DFS_rec(root.right, p, q, crnt_path, first_path)
        
        if found_node is not None:
            return found_node
        # not found this was dead end, clear the queue 
        crnt_path.pop(-1)
        return None





    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        node = self.DFS_rec(root, p ,q, [], [])

```



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
   - [HeapQ Detailed Notes](heapq_python_tutorial.md)
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
2. **Using Python sort with custom sort criterion**

```python

# method-1 Lambda function - works in most of the cases 
alist = [<foo-1>, <foo-2>] # list of foo object
alist.sort(key=lambda x: x.foo) # define lambda mapping to the value on which comparison has to be done


# method-2 - pass a custom function yourself
from functools import cmp_to_key
# Custom comparator
def custom_comparator(x, y):
    if x > y:
        return 1  # x should come after y
    elif x < y:
        return -1  # x should come before y
    else:
        return 0  # x and y are equal

array = [5, 2, 9, 1, 5, 6]
array.sort(key=cmp_to_key(custom_comparator))  # Ascending
print(array)  # [1, 2, 5, 5, 6, 9]


# method-3 is simple to code but difficult to understand 
# it is about overwritting the str class itself 

class LargerNumKey(str):
    def __lt__(x, y):
        # Compare x+y with y+x in reverse order to get descending order
        # do some comparison here, return true or false
        return x+y > y+x
# and pass this class as key argument in the sort function 
nums.sort(key=LargerNumKey)

```


