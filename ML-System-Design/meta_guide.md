# Machine Learning (ML) System Design Interview

## What can you expect?

The ML system design interview will consist of a **45-minute session** and focuses on your ability to **build ML systems at scale**. A strong performance in this interview indicates to your interviewer that you’d be successful in most applied ML problems here at **Meta**.

You can expect your interviewer to ask:

- Design a personalized news ranking system.
- Design a product recommendation system.
- Design an evaluation framework for ads ranking.

---

## What do we look for?

The purpose of this interview is for the candidate to **pick any product feature** and understand how to **model it using ML**. 

We’re **not** looking for you to memorize every ML algorithm. Instead, you should:

- Use your **existing toolsets** to model the problem.
- Break down the components involved in building a **large-scale ML system**.

# **Signals we assess:** 

- **Problem Navigation**  
  Can you visualize and organize the entire problem and solution space? Can you connect the business context and needs to the ML decisions?

- **Training Data**  
  How would you identify methods to collect training data? How do you look at the constraints/risks with a proposed method?

- **Feature Engineering**  
  Can you come up with relevant ML features for your model? How do you identify the most important features for the specific task?

- **Modeling**  
  How do you explain modeling choices? Are you able to justify the decision to use a specific model? Can you explain the training process? Can you anticipate risks and mitigate them?

- **Evaluation & Deployment**  
  Can you design consistent evaluation and deployment techniques? How do you justify and articulate your choice of metrics to track?

---

## How to prep

To practice, take any well-known app and pick a system that could benefit from ML.

**Example**: How would you design a friend recommendation system for Twitter?  
Initially, this system may have used static rules for a small user base. Now, you want to deprecate those rules and scale it with ML for millions of users.

### Suggested Prep Steps:

- 🧠 **Brush up on basic ML theory and algorithm details**  
  Be comfortable with concepts like **overfitting** and **regularization**.

- 🔧 **Practice converting intuitive ideas into concrete features**  
  For example, “number of likes” is a good start, but better features may involve **normalization**, **smoothing**, and **bucketing**.

- 🚀 **Think about the end-to-end system in production**  
  What will you do if your model doesn’t perform well? How do you **debug**, **evaluate**, and **deploy** continuously?

- ❓ **Prepare questions for the interviewer**  
  What info will help you make better design decisions? What clarifications might help you narrow your focus?

- ⚖️ **Analyze your approach and trade-offs**  
  Be ready to explain pros and cons of various techniques.  
  Example: Explain the advantages of **logistic regression** vs. **SVM**.

- ✍️ **Work out problems on paper or a whiteboard**  
  Practice breaking down real-world problems into smaller components.

- 🎥 **Watch public videos**  
  Learn how large systems like **Google search ranking** work.

---

Good luck!
