from code_samples.linked_list import ListNode

# Important Lessons
1. Read the Question carefully and see what it is expecting to do
   1. Today I did the [Reverse sub Singly-Linked list](https://leetcode.com/problems/reverse-linked-list-ii/?envType=problem-list-v2&envId=linked-list) - I didn't notice the boundary cases which changes the whole solving criterion
2. LinkedList and Graph question must be written on paper before jumping on to implementing them, line by line operation should be clearly written on paper to not require to think when writing code.





### Tricks for Linked list

1. Always create a dummy head and tail node and start process from this dummy to handle boundary cases better. Delete this boundary cases in the end of the code. For single linked list it gets even simpler because prev pointer have not to be handled.
   1. This approach very elegantly handeled cases where linked list is empty or has only one element, start loop with `crnt.next is None` and first line is `crnt  = crnt.next` which skips the dummy node is start
2. Never run a index based iteration based on next operator of linkedlist. For example: running an outer loop for fixed number of time based on next operation, while the inner loop is editing the nodes relationship. Loop in LinkedList has to be based on parameter been worked upon or a fixed index only.
3. If in any question we need to perform sorting on linkedlist, don't go on to implement it yourself but convert the values to array and use arr.sort() function, this would be more space complex but would be anyday better than implementing own sorting.
4. if it comes to implement sorting of linked list: use MergeSort when time-complexity is priority and BubbleSort if coding-time is priority.
5. If needed there is a implementation of LL sorting in `code-samples>linked_list.py` of this directory
6. Linked list can be very tricky to debug, even one logical error make it go haywire. Above LL merge-sorting took 15 min to implement and 30 min to debug.






### Sorting Algorithms:
1. `BubbleSort` O(n2), worst time complexity, easy to implement: 
   1. travel from left to right and take the largest element to the end of array in every loop
   2. Requires one direction traversal so can be used for single Linked list
2. `Insertion Sort` O(n2), worst time complexity, easy to implement:
   1. Travel from ith to 0 for increasing i, consider this as accumulating shorted array in the left.
   2. At ith Loop the array will be sorted till ith index and we are looking for correct poisition for i+1 th element in the sorted array.
   3. Requires reverse traversal so not feasible in the single linked list.
3. `Merge Sort` O(nLogn) Divide and conquer: most common and not soo complex to implement (though not trivial also)
   1. Divide the list in two half and now recursively  call the Merge sort on shorter list
   2. Merge the two smaller sorted an in traverse back step.
   3. Require a recursive call so takes memory to be loaded.
   4. 


## Python Tricks 

1. For program requiring sorting and top-k like structure use inbuilt functions for example heapq for getting max/min of a dynamic list
[HeapQ Tutorial](./heapq_python_tutorial.md)
```python
# Heapq general usage 
# assume node is a class
import heapq
nodes = [(3, Node("A", 3)), (1, Node("B", 1)), (4, Node("C", 4)), (2, Node("D", 2))] # using tuple (priority, object)
heapq.heapify(nodes)
# Push a new Node into the heap
heapq.heappush(nodes, (0, Node("E", 0)))
popped_node = heapq.heappop(nodes) # smallest node
largest_nodes = heapq.nlargest(2, nodes) # Get the 2 largest nodes
smallest_nodes = heapq.nsmallest(2, nodes) # Get the 2 smallest nodes
```