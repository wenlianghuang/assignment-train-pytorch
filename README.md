# Technical Assignment: Basic Shape Classifier Using PyTorch

## Objective:
Develop a Jupyter Notebook that demonstrates the training, testing, and validation of a simple CNN model to classify images into three categories: circles, squares, and triangles.

## Tools and Libraries Required:
* Python 3.x
* PyTorch
* torchvision
* matplotlib (for visualization)
* PIL (Python Imaging Library)
* Jupyter Notebook

## Dataset:
Generate a synthetic dataset using Python's PIL library. Each image will contain a single shape (circle, square, or triangle) against a plain background.

# Task Description:
## Data Generation and Preprocessing:
Write a function to generate images of circles, squares, and triangles. Each shape should be randomly placed within the image frame.

## Ensure a balanced dataset: generate an equal number of images for each shape.
Normalize the images and split the dataset into training, validation, and testing sets.
Create DataLoader for each dataset subset with a suitable batch size.

# Model Architecture:

## Design a simple CNN model that includes:
At least two convolutional layers.
ReLU activation functions.
MaxPooling layers.
One or two fully connected layers.
Dropout and batch normalization layers, if deemed necessary.
Print the model summary.

## Training the Model:

Define the loss function (e.g., cross-entropy loss).
Select an optimizer (e.g., Adam or SGD).
Train the model for a suitable number of epochs (e.g., 10-20 epochs).
Implement checkpoints to save the model at regular intervals.
Include inline comments explaining the choice of hyperparameters.

## Testing and Validation:

Evaluate the model on the test dataset.
Calculate and print the classification accuracy.
Display a confusion matrix.

## Visualization:

Plot training and validation loss over epochs.
Visualize some sample predictions with actual labels vs. predicted labels.

## Documentation and Code Quality:

Include detailed comments and explanations in the Jupyter Notebook, describing each step of the workflow.
Ensure the code is clean, modular, and well-organized.

## Deliverables:
A Jupyter Notebook containing the completed assignment.
A brief report summarizing the methodology, results, and any observations or conclusions.

## Evaluation Criteria:
Correct implementation of the CNN model.
Accuracy of the model on the test dataset.
Quality and clarity of the code and comments.
Effective use of PyTorch and other libraries.
Insightful analysis and presentation of the results.

# How to Submit Your Assignment
To ensure a smooth submission process and proper evaluation of your work, please follow the steps outlined below:

## Create a Repository:

Set up a new GitHub repository for this assignment in your github personal account. Name the repository as `ShapeClassifierPyTorch`.
Initialize the repository with a README.md file that briefly describes the project and its structure. Include any necessary instructions for running the notebook and any dependencies that need to be installed.

## Commit Your Jupyter Notebook:

Develop your solution in a Jupyter Notebook named Shape_Classifier.ipynb.
Ensure that your notebook includes comprehensive comments and explanations as specified in the assignment.
Commit and push your completed notebook to the repository. Make sure that the notebook is well-documented and easy to understand.

## Include Additional Resources:

If your project uses any additional scripts or data files, make sure they are included in the repository and properly referenced in the notebook.
Ensure that all necessary files to run the notebook and replicate the results are available in the repository.

## Prepare for Submission:

Double-check your repository to ensure that it is public and accessible. This allows the evaluation team to review your work without any issues.
Review your email to ensure that it includes the repository link and any other required information as specified in the assignment guidelines.

## Submit Your Work:

Send an email to service@smartsurgerytek.com with the subject line "Mid-Level Machine Learning Engineer Technical Assignment Submission".

## In the body of the email, include:
Your full name and contact information.
A brief introduction or cover letter that provides context about your submission.
The link to your GitHub repository containing the completed assignment.

## Follow Up:

After submitting your assignment, you may follow up with an email if you have not received an acknowledgment within a reasonable time frame (e.g., one week).
Be prepared to discuss your project and answer any questions that may arise during the review process.
By following these steps, you can ensure that your submission is organized and professional, reflecting your abilities and attention to detail. Good luck with your assignment!

