Here's the updated **Markdown version** with **formulas**, **descriptions**, and added **pros and cons** where applicable:

---

# 📊 Offline Evaluation Metrics

Offline evaluation uses historical or validation data to assess the model's performance before it is deployed in real-world settings. Metrics vary by task, and understanding their trade-offs is crucial.

---

## 🔢 **Classification**

### 1. **Precision**
- **Formula**:
  
  $ math
  \text{Precision} = \frac{TP}{TP + FP}
  $ 
- **Description**: Fraction of predicted positives that are actually correct.
- **Pros**:
  - Great when **false positives are costly** (e.g., spam detection).
- **Cons**:
  - Ignores false negatives.

---

### 2. **Recall**
- **Formula**:
  $ 
  \text{Recall} = \frac{TP}{TP + FN}
  $ 
- **Description**: Fraction of actual positives correctly predicted.
- **Pros**:
  - Useful when **missing a positive is dangerous** (e.g., cancer diagnosis).
- **Cons**:
  - Can be high even with many false positives.

---

### 3. **F1 Score**
- **Formula**:
  
  $\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
  
- **Description**: Harmonic mean of precision and recall.
- **Pros**:
  - Balances precision and recall.
- **Cons**:
  - Can be misleading if the class distribution is highly imbalanced.

---

### 4. **Accuracy**
- **Formula**:
  $ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $ 
- **Description**: Overall fraction of correct predictions.
- **Pros**:
  - Easy to interpret.
- **Cons**:
  - **Misleading on imbalanced datasets**.

---

### 5. **ROC-AUC**
- **Description**: Area under the ROC curve; shows trade-off between true positive and false positive rates.
- **Pros**:
  - Independent of classification threshold.
- **Cons**:
  - Can be **optimistic for imbalanced datasets**.

---

### 6. **PR-AUC**
- **Description**: Area under the Precision-Recall curve.
- **Pros**:
  - Better than ROC-AUC for **imbalanced datasets**.
- **Cons**:
  - Harder to interpret than ROC.

---

### 7. **Confusion Matrix**
- **Description**: Matrix showing TP, FP, TN, FN counts.
- **Pros**:
  - Gives full error breakdown.
- **Cons**:
  - Not a scalar; harder to compare models.

---

## 📈 **Regression**

### 1. **Mean Squared Error (MSE)**
- **Formula**:
  $ 
  \text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
  $ 
- **Pros**:
  - Penalizes larger errors more.
- **Cons**:
  - Sensitive to outliers.

---

### 2. **Mean Absolute Error (MAE)**
- **Formula**:
  $ 
  \text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|
  $ 
- **Pros**:
  - Robust to outliers.
- **Cons**:
  - May not capture large deviations well.

---

### 3. **Root Mean Squared Error (RMSE)**
- **Formula**:
  $ 
  \text{RMSE} = \sqrt{\text{MSE}}
  $ 
- **Pros**:
  - Same units as the target.
- **Cons**:
  - Still sensitive to outliers.

---

## 🔍 **Ranking**

### 1. **Precision@k**
- **Formula**:
  $ 
  \text{Precision@k} = \frac{\text{Relevant in Top-}k}{k}
  $ 
- **Pros**:
  - Focuses on top-ranked results.
- **Cons**:
  - Ignores results beyond top-k.

---

### 2. **Recall@k**
- **Formula**:
  $ 
  \text{Recall@k} = \frac{\text{Relevant in Top-}k}{\text{Total Relevant}}
  $ 
- **Pros**:
  - Shows coverage of relevant results.
- **Cons**:
  - Can be artificially inflated by large k.

---

### 3. **MRR (Mean Reciprocal Rank)**
- **Formula**:
  $ 
  \text{MRR} = \frac{1}{N} \sum \frac{1}{\text{Rank of First Relevant Item}}
  $ 
- **Pros**:
  - Emphasizes early correct hits.
- **Cons**:
  - Only uses the rank of the **first** relevant item.

---

### 4. **mAP (Mean Average Precision)**
- **Formula**:
  $ 
  \text{mAP} = \frac{1}{Q} \sum_{q=1}^{Q} \text{AP}(q)
  $ 
- **Pros**:
  - Balances both precision and ranking.
- **Cons**:
  - Harder to compute and interpret.

---

### 5. **nDCG**
- **Formula**:
  $ 
  \text{nDCG}_k = \frac{DCG_k}{IDCG_k}
  $ 
- **Pros**:
  - Takes **position** and **relevance** into account.
- **Cons**:
  - Requires graded relevance scores.

---

## 🖼️ **Image Generation**

### 1. **FID (Fréchet Inception Distance)**
- **Description**: Measures similarity between generated and real images using mean and covariance of features.
- **Pros**:
  - Correlates well with human judgment.
- **Cons**:
  - Needs a pre-trained Inception model; sensitive to implementation.

---

### 2. **Inception Score**
- **Description**: Measures quality and diversity of generated images.
- **Pros**:
  - Easy to compute.
- **Cons**:
  - Can be **fooled by adversarial examples**; less reliable than FID.

---

## 📝 **Natural Language Processing (NLP)**

### 1. **BLEU**
- **Description**: Compares overlapping n-grams with reference translations.
- **Pros**:
  - Standard metric in machine translation.
- **Cons**:
  - Ignores synonyms and grammar.

---

### 2. **METEOR**
- **Description**: Considers synonyms, stemming, and word order.
- **Pros**:
  - Better correlation with human judgment.
- **Cons**:
  - Slower to compute.

---

### 3. **ROUGE**
- **Description**: Recall-oriented n-gram overlap; often used in summarization.
- **Pros**:
  - Captures content coverage well.
- **Cons**:
  - Ignores fluency or grammar.

---

### 4. **CIDEr**
- **Description**: Consensus-based scoring using TF-IDF weighting on captions.
- **Pros**:
  - Designed for image captioning.
- **Cons**:
  - Requires multiple human references.

---

### 5. **SPICE**
- **Description**: Compares scene graphs (semantic propositions).
- **Pros**:
  - Focuses on meaning, not form.
- **Cons**:
  - Complex and computationally expensive.

---

Let me know if you want this exported to a `.md`, `.pdf`, or `LaTeX` version!