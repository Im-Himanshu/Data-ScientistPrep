# Definition for singly-linked list.
from timeit import dummy_src_name


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:

    def reverseNextk(self, nodeHead, k):
        dummy = ListNode(None, nodeHead)
        crnt = dummy
        total = 0
        # check if k element exist or not
        while (crnt.next):
            crnt = crnt.next
            total += 1
            if total >= k:
                break
        if total < k:
            # crnt mean end of the list
            # nodeHead remains the head of the UNreversed list also
            return nodeHead, crnt, None  # end of the computation

        dummy = ListNode(None, nodeHead)
        crnt = dummy
        next_n = crnt.next
        prev = None
        turns = 0
        while (turns < k and next_n is not None):
            # move pointer ahead
            prev = crnt
            crnt = next_n
            next_n = crnt.next

            # edit the current pointer
            crnt.next = prev
            turns += 1

        newStartNode = crnt
        newEndNode = nodeHead
        nextStartNode = next_n
        return newStartNode, newEndNode, nextStartNode

    def reverseKGroup(self, head, k: int):
        dummy = ListNode(None, head)
        prevEnd = dummy
        nextstartNode = dummy.next
        while (nextstartNode is not None):
            newStartNode, newEndNode, nextstartNode = self.reverseNextk(nextstartNode, k)
            prevEnd.next = newStartNode
            prevEnd = newEndNode
            prevEnd.next = None

        newEndNode
        i = 0
        crnt = dummy
        while (i < 10 and crnt.next is not None):
            crnt = crnt.next
            print(crnt.val)
            i = i + 1

        return dummy.next


def createNode(arr):
    dummy = ListNode(None, None)
    crnt = dummy

    for i in arr:
        newNode = ListNode(i, None)
        crnt.next = newNode
        crnt = newNode
    return dummy.next


arr = [1,2,3,4,5]
ll = createNode(arr)
Solution().reverseKGroup(ll, 1)


