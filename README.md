# question-answering-system

This project is basically a question answering system based on a context.
This project adapts and implements the following research paper. 
 <a href="https://arxiv.org/abs/1603.01417">Dynamic Memory Networks for Visual and Textual Question Answering</a>

# Data Processing
This model is trained on babi dataset on tasks 1,2 and 3. We used Glove vectors of dimension 100 for word vectorization.
Following is the word cloud of task 1.

![1](https://github.com/divyakrishna-devisetty/question-answering-system/blob/master/screenshots/1.JPG)

# Training
Training set consists of 9k questions and answers. Validation set contains 1k question and answers.
Test set contains 1k question and answers.
Training is carried out on batch size of 100 for 20 epochs for task-1.
Recurrent cell size is 80.
Loss Function: soft_max cross entropy.
Optimizer: Adam

The model consists of 4 modules implemented as follows.

![2](https://github.com/divyakrishna-devisetty/question-answering-system/blob/master/screenshots/2.JPG)

# Demo

![3](https://github.com/divyakrishna-devisetty/question-answering-system/blob/master/screenshots/3.JPG)

# Packages used
<ul>
<li> Numpy</li>
<li> Pandas </li>
<li> Keras </li>
<li> TensorFlow </li>
</ul>

# Future Improvements

<ul>
<li> To extend the model for other babi tasks</li>
<li> To improve efficiency using regularization techniques </li>
</ul>








