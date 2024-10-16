# HeapQ tutorial in python 
Important Points:
1. It's a functionality package that works inplace, over the user defined array. here are some important function in it.
2. Very good for solving complex quickly that require sorting of dynamic arrays and very well implemented
3. Restriction is it only works with arrays, for other structures it need to be managed with tuple (value, _ , _, _,) which will sort it based on value and we can access rest value of tuple for our purposes.
4. Heapq is biased for getting the lowest element in the heap, to reverse the behaviour pass the negative value of the priority or change the _lt_ operator behaviour.

###  Basic usage of heap for primitive datatypes 
```python
import heapq
values = [5, 1, 3, 7, 4, 2] # Initialize a list with some values
heapq.heapify(values) # Convert the list into a heap
heapq.heappush(values, 6) # Add a new value to the heap
smallest = heapq.heappop(values) # Remove and return the smallest element from the heap
n_smallest = heapq.nsmallest(3, values) # Get the nth smallest elements from the heap
```


## Heap with object nodes and comparative operator
we can store an object node in the array and ask heapq to sort them for us, but we either need to custom define the __lt__ operator in the definition of the object class.
Or has to pass this as a tuple of (priority, objectNode)
### Method-1- using tuple with tuple of (priority, objectNode)

```python
import heapq

# Instead of a custom Node class, use a tuple (priority, value)
nodes = [
    (3, "A"),
    (1, "B"),
    (4, "C"),
    (2, "D"),
]

# Heapify the list of tuples
heapq.heapify(nodes)
print("Heapified nodes:", nodes)

# Push a new tuple into the heap
heapq.heappush(nodes, (0, "E"))
print("After pushing E:", nodes)

# Pop the smallest (highest priority) tuple from the heap
popped_node = heapq.heappop(nodes)
print("Popped node:", popped_node)
print("Nodes after popping:", nodes)

# Push another tuple
heapq.heappush(nodes, (5, "F"))
print("After pushing F:", nodes)

# Pop all tuples in order of priority
print("Popping all nodes in priority order:")
while nodes:
    print(heapq.heappop(nodes))

```


Method-2: More general and scalable, using __lt__ operator in class definition 
``` python
import heapq

class Node:
    def __init__(self, value, priority):
        self.value = value
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority

    def __repr__(self):
        return f"Node(value={self.value}, priority={self.priority})"

# Create a list of Node objects
nodes = [
    Node("A", 3),
    Node("B", 1),
    Node("C", 4),
    Node("D", 2),
]

# Heapify the list of nodes
heapq.heapify(nodes)

# Push a new Node into the heap
new_node = Node("E", 0)
heapq.heappush(nodes, new_node)

# Pop the smallest (highest priority) node from the heap
popped_node = heapq.heappop(nodes)

# Push another node
heapq.heappush(nodes, Node("F", 5))

```