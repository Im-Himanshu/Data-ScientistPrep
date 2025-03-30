

### Problem-1
Linked list, Hard : [leet code link](https://leetcode.com/problems/reverse-nodes-in-k-group/submissions/1424776833/?envType=problem-list-v2&envId=linked-list)
In this i forgot to delete the additional next connection I set to the dummy node due to which there was a loop introduced in the ll.

**Always delete the extra connection to the dummy node, it may cause cycles in the list later on.**



### Problem-2
Dynamic Programming on Strings: [leet code](https://leetcode.com/problems/interleaving-string/)
Lesson about managing boundary cases in DP problem on strings.
this DP solution had me learn following 

1. Any DP problem melt like butter if written properly on a paper
2. Implementation of DP and handling boundary cases can be complicated, define all the boundary cases what does dp[i,j] stores exactly in words
3. the logic of DP should have been seprated from the initiation logic to make things handy
4. Initial exit condition to exit straightaway should always be thinked in final submissions
5. Thinking in terms of index vs actual number is complicated - get use to it
6. In case of string type question, 0th case has to be part of DP, so either handle it in the if else logic but that is complicated, so simply sotres it in the DP array and move the index

### Problem-3: 

Graph Problem with, Trickiest of the lot, requiring understanding of it all
[Cheapest flight with k-stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)
Self Notes:

this is one of the trickiest problem involving graph: this can be solved in n umber of ways all requiring subtle modification in the 
original graph algorithm code. Main feature is 

1. the queue is also used as a prev step configuration storage and same node is processed multiple times with multiple configurations
2. This is kind of a DP problem but because of ordering BFS (ie. level keep increasing) the reptitive case never occurs
3. additionally not all the n^3 cases like in Warshall algorithm need to be solved
4. Though the stand bellman Fordd algorithm would have beeen best for solving this.
5. Warshall is the standard solution of this problem.
6. This problem cannot be done without having clear understanding of how to write a DP problem in Graph,
7. try writting this problem as a topological sort graph and then as a DP problem the complxity of sub-problem shows up itself
8. This was implemented and inspired by this [article](https://leetcode.com/problems/cheapest-flights-within-k-stops/solutions/4202933/7-solutions-python-java-tle-bfs-dfs-dp-bellman-ford-dijkstra)

the above is little too much but reply of same argument, BF algo is to be looked into in this.
