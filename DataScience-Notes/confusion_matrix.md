

# Correct order - My intuition is set on this

### **Confusion Matrix:**

A simple binary classification confusion matrix looks like this:

| Predicted \ Actual     | **Actual Positive**                 | **Actual Negative**              | Matrics                                |
|------------------------|-------------------------------------|----------------------------------|----------------------------------------|
| **Predicted Positive** | True Positive (TP)                  | False Positive (FP)              | **Precision** = $`\frac{TP}{TP + FP}`$ |
| **Predicted Negative** | False Negative (FN)                 | True Negative (TN)               | Nothing     |
| Matrics                | **Recall\TPR** = $`\frac{TP}{TP + FN}`$ | **(1-FPR)** = **Specificty** = $`\frac{TN}{TN + FP}`$ |                                        |

Specificty is equilvalent to Recall on negative class \
FPR = (1- Specificity) - used on ROC curve x-axis
ROC curve is drawn b/w Recall(y-axis) and FPR (x-axis)


Realization:
- Denominator of Recall and FPR i.e (TP+FN) and (TN+FP) is always constant/fixed. Because it is the count of Actual positive or negative example respectively in our data.
- So the only variable is TP in recall and FP in FPR, which has mostly direct relation if we change our model threshold. i.e. If FP dec. Tp also decreases or remain same or viceversa.
- Our overall aim is to Maximize TP and minimize FP for the best model. which happen when we move to the left top corner of the ROC curve. SO we try to shift to the left such that we don't go down (reduce recall).
- Today I have thinked of this as 2 old bucket (actual positive, actual negative) to two new bucket (pred )positive and negative class. so recall and FPR is wrt to the old buckets because the denominator is what we had in old buckets while numerator is wrt to new buckets.

Precision-recall Curve is drawn b/w (precision (y-axis)) and recall (x-axis).

![img_1.png](img_1.png)
---
![](../Assets/abTesting/img.png)



Sure! Here's a breakdown of the metrics and their definitions in terms of True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN):

---

### **Definitions:**
| Metric                          | Formula                                                                                      | Description                                                                                       |
|---------------------------------|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Precision (Positive Predictive Value)** | $`\text{Precision} = \frac{TP}{TP + FP}`$                                                    | Measures the proportion of positive predictions that are actually correct.                       |
| **Recall (True Positive Rate, Sensitivity)** | $`\text{Recall} = \frac{TP}{TP + FN}`$                                                       | Measures the proportion of actual positives that are correctly identified.                       |
| **False Positive Rate (FPR)**   | $`FPR = \frac{FP}{FP + TN}`$                                                                 | Measures the proportion of negatives incorrectly classified as positives.                        |
| **Specificity (True Negative Rate)** | $`\text{Specificity} = \frac{TN}{TN + FP}`$                                                  | Measures the proportion of actual negatives correctly classified.                                |
| **Accuracy**                    | $`\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}`$                                      | Measures the overall correctness of predictions.                                                 |
| **F1-Score**                    | $`F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}`$ | Harmonic mean of Precision and Recall.                                                           |
| **Balanced Accuracy**           | $` \text{Balanced Accuracy} = \frac{\text{TPR} + \text{Specificity}}{2} `$                   | Average of True Positive Rate (TPR) and Specificity.                                             |


---


Let me know if you’d like to calculate these metrics for a specific example!
