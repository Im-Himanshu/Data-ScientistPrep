# Interview Process let se ee 

## Round 1 - Coding (1 hour) 

> 1 easy - 20-25 minutes (35 marks) - avoid very easy\
1 medium - 35-40 minutes (65 marks) \
>
>[Bonus] If time remains after 2 coding questions (typically 10-15 minutes), system knowledge - caching, threading (vs processes), async/sync difference, pubsub, map reduce paradigm etc. /

Passing criteria: 75% or can be based on interviewer \

Below is a sample criteria:
If only approach correct: easy (20) - medium (40) ⇒ correct use of data structures, big O complexity, efficient algorithm - avoid questions with clever tricks
Solves most test cases in easy (35) AND medium approach is correct (40)
Stress on writing code
Good solution: code that thinks of all test cases
Topics
Lists, arrays, searching, sorting
Basic use of heaps (PQs) and trees
Basic DP - fibonacci etc, rod cutting, unique path count etc.
Graphs - DFS, BFS and applications

## Round 2 - ML and DL (1 hour - 1 hour 30 minutes)

### Initial list of important topics. Add more topics/questions.
1. Stats - A/B, hypothesis testing, p-value or confidence intervals

- [x] This is a complete item
- [ ] This is an incomplete item
### ML algorithms
1. dimensionality reduction basics (PCA vs. SVD vs. 
> non-linear versions t-SNE)
2. imbalance classification ways to deal - LR (threshold), oversampling vs. undersampling strategies, choice of evaluation metrics etc.
3. Loss function: regularization - L1/L2, 
>Elastic Net
4. Evaluation metrics: ROC, PR curve, 
> F1, weighted F1, regression metrics, rmsle vs rmse
5. Tree based - decision tree, entropy, pruning, RF, boosting, 
>gradient boosting

### Probability	
1. calculation question on expected values or bayes theorem
(Advanced) Mutual information based problems - ex: find phrases

### DL algorithms
1. Cross entropy, softmax, residual connections
2. Activation functions - purpose
3. Initialization strategies - application based
4. Batch norm, regularization (dropout) purpose
5. Vanishing/exploding gradient and mitigation ways
6. Optimizers: Effect of batch size, momentum concepts, learning rate, SGD vs mini batch vs GD
7. Depthwise separable convolutions for reducing compute
8. (Advanced) bottleneck, inverted bottleneck designs and why they are used
9. (Advanced) upsampling - deconvolution vs. convolution
10. (Advanced) quantization purpose, pruning
Design questions
11. Prediction latency is important - SVM vs RF/Boosting? When accuracy is the same, what to use? [comparison](https://www.thekerneltrip.com/statistics/random-forest-vs-svm/#:~:text=What%20we%20can%20see%20is,considered%20when%20chosing%20the%20algorithm.)




to do again
1. SVM Maths and time complexity with relationship to the RF and CART