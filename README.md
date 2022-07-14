# Neural_Network_Charity_Analysis
Neural Networks and Deep Learning Models


## Overview:
Neural networks is an advanced form of Machine Learning that can recognize patterns and features in the dataset. Neural networks are modeled after the human brain and contain layers of neurons that can perform individual computations. A great example of a Deep Learning Neural Network would be image recognition. The neural network will compute, connect, weigh and return an encoded categorical result to identify the image. <br>
AlphabetSoup, a philanthropic foundation is requesting for a mathematical, data-driven solution that will help determine which organizations are worth donating to and which ones are considered "high-risk". In the past, not every donation AlphabetSoup has made has been impactful as there have been applicants that will receive funding and then disappear. Beks, a data scientist for AlphabetSoup is tasked with analyzing the impact of each donation and vet the recipients to determine if the company's money will be used effectively. In order to accomplish this request, we are tasked with helping Beks create a binary classifier that will predict whether an organization will be successful with their funding. We utilize Deep Learning Neural Networks to evaluate the input data and produce clear decision making results.<br>
The files for the Challenge work can found in [Challenge Folder](https://github.com/ashwinihegde28/Neural_Network_Charity_Analysis/tree/main/Challenge) .

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
The IS_SUCCESSFUL column.  

2. What variable(s) are considered to be the features for your model?    
The IS_SUCCESSFUL column is the feature chosen for this dataset.

3. What variable(s) are neither targets nor features, and should be removed from the input data?    
The EIN and NAME columns will not increase the accuracy of the model hence removed to improve code efficiency. <br>

![Deliverable1](https://github.com/ashwinihegde28/Neural_Network_Charity_Analysis/blob/main/Challenge/Resources/Deliverable1.PNG)<br>

### Compiling, Training, and Evaluating the Model
4. How many neurons, layers, and activation functions did you select for your neural network model, and why?    
In the optimized model, layer 1 started with 120 neurons with a relu activation.  For layer 2, it dropped to 80 neurons and continued with the relu activation.  From there, the sigmoid activation seemed to be the better fit for layers 3 (40 neurons) and layer 4 (20 neurons).    <br>

![Optimisation1](https://github.com/ashwinihegde28/Neural_Network_Charity_Analysis/blob/main/Challenge/Resources/Optimisation1.PNG)   <br>
![Optimisation1](https://github.com/ashwinihegde28/Neural_Network_Charity_Analysis/blob/main/Challenge/Resources/Optimisation1Accuracy.PNG)   <br>


5. Were you able to achieve the target model performance?   
The target for the model was 75%, but the best the model could produce was 72.8%.

6. What steps did you take to try and increase model performance?   
Columns were reviewed and few columns were dropped as well as increasing the number of neurons and layers.  Other activations were tried such as sigmoid,relu,linear  but the range that model produced max of 73% accuracy approximately.  The linear activation produced the loss around 0.5 comapred to others. The relu activation at the early layers and sigmoid activation at the latter layers gave the best results of 0.7287.  <br>
1. First attempt:
- Dropped the non-beneficial ID columns, 'EIN','NAME','STATUS','SPECIAL_CONSIDERATIONS'initially and looked at APPLICATION_TYPE and CLASSIFICATION value counts for binning.
- Used 4 hidden layers and an outout layers with 110 units with relu activation, 80 units with relu activation, 40 units with sigmoid activation, 20 units with sigmoid activation and 1 output layer with single unit with liner activation respctively.
- Achieved Loss: 0.553674042224884, Accuracy: 0.7258309125900269
![optimisation2](https://github.com/ashwinihegde28/Neural_Network_Charity_Analysis/blob/main/Challenge/Resources/Optimisation1.PNG)  <br>
![optimisation2](https://github.com/ashwinihegde28/Neural_Network_Charity_Analysis/blob/main/Challenge/Resources/Optimisation1Accuracy.PNG)  <br> 

2. Second attempt:
- Dropped the non-beneficial ID columns, 'EIN','NAME','STATUS','SPECIAL_CONSIDERATIONS' initially and looked at APPLICATION_TYPE value counts for binning.
- Used 2 hidden layers and an outout layers with 100 units with relu activation, 50 units with relu activation and 1 unit with sigmoid activation respctively
- Achieved Loss: 0.5783612132072449, Accuracy: 0.7259474992752075 (73%)
![optimisation2](https://github.com/ashwinihegde28/Neural_Network_Charity_Analysis/blob/main/Challenge/Resources/optimisation2Layers.PNG)  <br>
![optimisation2](https://github.com/ashwinihegde28/Neural_Network_Charity_Analysis/blob/main/Challenge/Resources/optimisation2.PNG)  <br> 

3. Third attempt:
- Dropped the non-beneficial ID columns, 'EIN','NAME' in the initial dataframe and looked for AFFILIATION and CLASSIFICATION value counts for binning.
- Used 4 hidden layers and an outout layers with 125 units with relu activation, 50 units with relu activation, 40 units with sigmoid activation, 20 units with sigmoid activation and a single neuron with sigmoid activation respctively.
- Achieved accuracy: Loss: 0.5672652125358582, Accuracy: 0.72874635457992555 (73%)
![optimisation3](https://github.com/ashwinihegde28/Neural_Network_Charity_Analysis/blob/main/Challenge/Resources/optimisation3Layers.PNG) <br>
![optimisation2](https://github.com/ashwinihegde28/Neural_Network_Charity_Analysis/blob/main/Challenge/Resources/optimisation2.PNG)  <br>




## Summary:  
- The relu and sigmoid activations yielded a 72.8% accuracy, which is the best the model could produce using various number of neurons and layers.
- The next step should be to try the random forest classifier as it is less influenced by outliers.
- Overall, Neural Networks are very intricate and would require experience through trial and error or many iterations to identify the perfect configuration to work with this dataset and best used with larger data set.
