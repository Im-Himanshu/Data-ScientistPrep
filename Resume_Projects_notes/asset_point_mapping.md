
Notes from Talk with Risav
![1.jpg](1.jpg)
![2.jpg](2.jpg)
![3.jpg](3.jpg)
![4.jpg](4.jpg)


# About project

The Project was about finding the device-type, location, sensor_type, equipment_type from a given sensor name.
This information was used to filter the menu to map the given sensor to given device in sensor discovery process.
The basic problem arises because sensor coming from different provider and being marked by different person can have different name as there is no specification on how to name it.
The name itself contains the basic info about the type of equipment it is part of and type of sensor it is but it is not clear.
To make it unambiguous, we have LLM to convert the given asset point/sensor name to its parent id.


Steps 

1. We have take the Pythia 70m - 210m model, which was open source for commercial use and fine tune it on this specific problem to output the four fields of interest.
[This course](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/), [its jupyter](https://github.com/pelinbalci/LLM_Notebooks/blob/main/LLM_Finetuning.ipynb), [Article on it](https://medium.com/@balci.pelin/llm-finetuning-410e8a2738ef)
This is basic step, which uses Tokenizer and Transformer from hugging face to load and fine tune the model. It also uses model evaluation from evaluator from hugging face to report BELU score and other important scores.
2. For this custom tokenizer was created that split the name of the device in varied length from 2-5 char and feed it LLM.


3. Handle the model observability part:
   - For These uses the Pytorch Gradient explainability part. - Read about this, implement this 


4. use of ditilBert, Decision Tree, or any other pre or post processor.


# Addressing Data Drift in production
1. Used `Evidently` for detecting data drift 
2. Matrix like KL-divergence, Jensen-Shannon Divergence or JS Divergence, Refer in the later section below for more details.
3. Question: How exactly was it used? What was the matrices being tracked?






# More About Pythia 

















# About Data-Drift Matrices

Data drift matrices like **Kullback-Leibler (KL) Divergence** and others help quantify how much the distribution of data has changed over time, which is essential in monitoring machine learning models for concept drift. Here are some key methods:

### 1. **Kullback-Leibler (KL) Divergence**  
   - Measures how one probability distribution \( P \) differs from a reference distribution \( Q \).  
   - Formula:  
     ```math
      D_{KL}(P || Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}
      ```
   - **Pros**: Works well for categorical data and probability distributions.  
   - **Cons**: Not symmetric $ D_{KL}(P || Q) \neq D_{KL}(Q || P) $, and requires $ Q(i) > 0 $ for all \( i \).  

### 2. **Jensen-Shannon (JS) Divergence**  
   - A symmetric version of KL divergence that also ensures finite values.  
   - Formula:  
     
   ``` math
     D_{JS}(P || Q) = \frac{1}{2} D_{KL}(P || M) + \frac{1}{2} D_{KL}(Q || M)  
   ```

 where $ M = \frac{P + Q}{2} $.  
   - **Pros**: Symmetric and always finite.  
   - **Cons**: More computationally expensive than KL divergence.  

### 3. **Wasserstein Distance (Earth Mover’s Distance - EMD)**  
   - Measures how much "work" is required to transform one distribution into another.  
   - Particularly useful for continuous distributions and numeric data.  
   - **Pros**: Works well for structured, numeric data.  
   - **Cons**: Computationally expensive for high-dimensional data.  
 

### 4. **Population Stability Index (PSI)**  
   - Measures drift in categorical or continuous variables.  
   - Formula:  
     ```math
     PSI = \sum (P(i) - Q(i)) \log \frac{P(i)}{Q(i)}
     ```
   - **Pros**: Common in credit risk and business analytics.  
   - **Cons**: Less informative for small sample sizes.  

Would you like help implementing any of these in Python?