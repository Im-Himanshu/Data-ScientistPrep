

Problem-1 Linked list, Hard : [leet code link](https://leetcode.com/problems/reverse-nodes-in-k-group/submissions/1424776833/?envType=problem-list-v2&envId=linked-list)
In this i forgot to delete the additional next connection I set to the dummy node due to which there was a loop introduced in the ll.

**Always delete the extra connection to the dummy node, it may cause cycles in the list later on.**