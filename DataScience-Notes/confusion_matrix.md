

# Correct order - My intuition is set on this

### **Confusion Matrix:**

A simple binary classification confusion matrix looks like this:

| Predicted \ Actual     | **Actual Positive**                 | **Actual Negative**              | Matrics                                |
|------------------------|-------------------------------------|----------------------------------|----------------------------------------|
| **Predicted Positive** | True Positive (TP)                  | False Positive (FP)              | **Precision** = $`\frac{TP}{TP + FP}`$ |
| **Predicted Negative** | False Negative (FN)                 | True Negative (TN)               | **TPR**=   $`\frac{TP}{TP + FN}`$      |
| Matrics                | **Recall** = $`\frac{TP}{TP + FN}`$ | **FPR** = $`\frac{TN}{TN + FP}`$ |                                        |

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
