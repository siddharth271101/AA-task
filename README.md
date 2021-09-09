# AA-task

<b>Problem Statement:</b>
- Imagine there is a file full of Twitter tweets by various users and you are provided a set of words that indicates racial slurs. Write a program that can indicate the degree of profanity for each sentence in the file.

<b>Approach:</b>
- Used TfidfVectorizer to vectorize the text before feeding into a SVM classifier to predict the results.
- The SVM classifier was trained 200k labelled samples of clean and profane text.
- Since SVM does not natively predict probabilities, Therefore the SVM is fit via the CalibratedClassifierCV class so that it returns a probability for each class instead of just a classification.

<b>Assumptions:</b>
- Due to limited time instead of a set of words that indicate racial slurs, I have used a labelled [dataset](https://github.com/vzhou842/profanity-check/blob/master/profanity_check/data/clean_data.csv). The dataset contains 2 columns, the first column denotes if the particular text is offensive or not, the second column has the actual text.

<b>Result:</b>  
![Output](https://github.com/siddharth271101/AA-task/blob/main/result/profanity_output.png)
