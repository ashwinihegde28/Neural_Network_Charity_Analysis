# Neural_Network_Charity_Analysis
Neural Networks and Deep Learning Models


## Overview:
Neural networks is an advanced form of Machine Learning that can recognize patterns and features in the dataset. Neural networks are modeled after the human brain and contain layers of neurons that can perform individual computations. A great example of a Deep Learning Neural Network would be image recognition. The neural network will compute, connect, weigh and return an encoded categorical result to identify the image. <br>
AlphabetSoup, a philanthropic foundation is requesting for a mathematical, data-driven solution that will help determine which organizations are worth donating to and which ones are considered "high-risk". In the past, not every donation AlphabetSoup has made has been impactful as there have been applicants that will receive funding and then disappear. Beks, a data scientist for AlphabetSoup is tasked with analyzing the impact of each donation and vet the recipients to determine if the company's money will be used effectively. In order to accomplish this request, we are tasked with helping Beks create a binary classifier that will predict whether an organization will be successful with their funding. We utilize Deep Learning Neural Networks to evaluate the input data and produce clear decision making results.

### Proces undergone:
1. Compare the differences between the traditional machine learning classification and regression models and the neural network models.
2. Describe the perceptron model and its components.
3. Implement neural network models using TensorFlow.
4. Explain how different neural network structures change algorithm performance.
5. Preprocess and construct datasets for neural network models.
6.Compare the differences between neural network models and deep neural networks.
7.Implement deep neural network models using TensorFlow.
8.Save trained TensorFlow models for later use.

## Results:
### Data Preprocessing
1. What variable(s) are considered the target(s) for your model?    
Checking to see if the target is marked as successful in the DataFrame, indicating that it has been successfully funded by AlphabetSoup.  

2. What variable(s) are considered to be the features for your model?    
The IS_SUCCESSFUL column is the feature chosen for this dataset.

3. What variable(s) are neither targets nor features, and should be removed from the input data?    
The EIN and NAME columns will not increase the accuracy of the model and can be removed to improve code efficiency. <br>

![Deliverable1](https://github.com/ashwinihegde28/Neural_Network_Charity_Analysis/blob/main/Challenge/Resources/Deliverable1.PNG)<br>

### Compiling, Training, and Evaluating the Model
4. How many neurons, layers, and activation functions did you select for your neural network model, and why?    
In the optimized model, layer 1 started with 120 neurons with a relu activation.  For layer 2, it dropped to 80 neurons and continued with the relu activation.  From there, the sigmoid activation seemed to be the better fit for layers 3 (40 neurons) and layer 4 (20 neurons).    <br>

![Optimisation1](https://github.com/ashwinihegde28/Neural_Network_Charity_Analysis/blob/main/Challenge/Resources/Optimisation1.PNG)   <br>

5. Were you able to achieve the target model performance?   
The target for the model was 75%, but the best the model could produce was 72.8%.

6. What steps did you take to try and increase model performance?   
Columns were reviewed and the STATUS and SPECIAL_CONSIDERATIONS columns were dropped as well as increasing the number of neurons and layers.  Other activations were tried such as tanh, but the range that model produced went from 40% to 68% accuracy.  The linear activation produced the worst accuracy, around 28%.  The relu activation at the early layers and sigmoid activation at the latter layers gave the best results.  <br>
![optimisation2](https://github.com/ashwinihegde28/Neural_Network_Charity_Analysis/blob/main/Challenge/Resources/optimisation2.PNG)  <br> 
![optimisation3](https://github.com/ashwinihegde28/Neural_Network_Charity_Analysis/blob/main/Challenge/Resources/optimisation3.PNG) <br>



## Summary:  
- The relu and sigmoid activations yielded a 72.8% accuracy, which is the best the model could produce using various number of neurons and layers.
- The next step should be to try the random forest classifier as it is less influenced by outliers.
- Overall, Neural Networks are very intricate and would require experience through trial and error or many iterations to identify the perfect configuration to work with this dataset and best used with larger data set.
