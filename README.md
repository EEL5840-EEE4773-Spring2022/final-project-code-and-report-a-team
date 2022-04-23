<!-- ABOUT THE PROJECT -->
## About The Project
The objective of this project was to develop an end to end machine learning algorithm that can classify handwritten characters into ten classes. This work was done as a course project for 'EEL5840: Fundamentals of Machine Learning' Course. Train dataset were collected by handwritten characters on physical paper. The train dataset consists of a total 6720 images with a dimension of 300x300 pixels each. We used a state-of-the-art CNN structure named ResNet-18 for final training. The final trained model was able to detect all ten classes with a validation accuracy of 99.48%. 

For complete technical description, please refer to this [report].


<!-- GETTING STARTED -->
## Getting Started
1. Included files in 'A-Team' Github repository: 

	- environment.yml
	- train.py
	- test.py
	- best_model_resnet18_e5.pt (Trained_Weights)

2. Setting up the working directory:

	- open the anaconda prompt
	- set up the working directory by writing the command:
		
		cd [full/path/to/project/in/your/local/machine]

3. Setting up the virtual conda environment:

	- now set up the virtual conda environment by writing following command:

		conda env create -f environment.yml
   - after successfully creating the virtual environment, write the following command:

     conda activate A-Team
            
4. Running the code:

	### Train
  
     -  now simply run the 'train.py' by writing following command:

      python train.py
          
     - When input prompt asks about the directory, please provide the directory of 'data_train.npy' and 'labels_train.npy'.
	
	### Test 
     
     - now test the model performance by writing following command:
		
		 python test.py 
     
     - Please rename the test data file to 'data_test.npy' and test label file to 'labels_test.npy', if these have any other names in your local file.
     
     - When input prompt asks about the directory, please provide the directory of 'data_test.npy', and 'labels_test.npy'.
     
     - When the input prompt asks about the full model path, please provide the path for the 'best_model_resnet18.pt' including the file name.
     
     Finally, after training test accuracy and numerical labels of test data will be printed.
     
     
<!-- Usage -->
## Usage

### Parameter Tuning
 
   - All the hyperparameters are initialized at the begining of train function. It can be changed easily.
      - EPOCHS (Number of Epochs).
      - BATCH_SIZE (Batch size).
      - START_LR, END_LR (Starting and End Learning Rate for ADAM)
      - NUM_ITER (Number of iterations for tuning the learning rate)   

<!-- Acknowledgement -->
## Acknowledgement

- Professor Catia Silva, who taught us machine learning.
- FICS GPU server, We used the server during training.


<!-- Authors -->
## Authors
'A-Team' team members

## Thank You.
