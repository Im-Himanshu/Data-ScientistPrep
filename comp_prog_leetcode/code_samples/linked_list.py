# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:

    ## this function sort the linked list in merge sort way O(nLogn)
    def listMergeSort(self, head, list_size):
        if list_size <= 1:
            return head
        MergeA = head
        len_A = int(list_size / 2)

        MergeB = head
        prev = None
        i = 0
        while (i < len_A):
            prev = MergeB
            MergeB = MergeB.next
            i = i + 1
        len_B = list_size - len_A
        if prev != None:
            prev.next = None  # take both seperately
        sortA = self.listMergeSort(MergeA, len_A)
        sortB = self.listMergeSort(MergeB, len_B)

        # merging operation O(n)
        Merged = ListNode(None, None)
        crnt_merged = Merged
        crntA = sortA
        crntB = sortB

        while (crntA is not None and crntB is not None):
            if crntA.val < crntB.val:
                crnt_merged.next = crntA
                crntA = crntA.next
            else:
                crnt_merged.next = crntB
                crntB = crntB.next
            crnt_merged = crnt_merged.next
        if crntA is not None:
            crnt_merged.next = crntA
        if crntB is not None:
            crnt_merged.next = crntB

        return Merged.next

    def print_list(self, head):
        crnt = head
        alli = ""
        while crnt is not None:
            alli = alli + str(crnt.val) + ", "
            crnt = crnt.next
        print(alli)

    def sortList(self, head):
        vHead = ListNode(None, head)
        crntTop = vHead
        count = 0
        while (crntTop.next is not None):
            crntTop = crntTop.next
            count += 1
        self.print_list(head)
        new_head = self.listMergeSort(head, count)
        self.print_list(head)
        return new_head