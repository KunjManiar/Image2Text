# Note:
The main topic of the project is <b>"Image Captioning using Neural Networks"</b>
This project is part of the Artificial Intelligence Course at VIT University.

# The students part of this project are:
* 17BCE2410 - Maniar Kunj Jignesh
* 17BCB0102 - Vaibhav Bhandari

# Introduction
In our project we aim to create an auto captioning model for an image. We achieve this by using Convolutional Neural Network for image feature classification and object localization. Then we use Recurrent Neural Networks for text generation. We use Long Short-Term Memory (LSTM) a type of RNN model which can learn from sequential data like a series of words and characters. These networks use hidden layers that link the input and output layers. Which creates a memory loop so that the model can learn from previous outputs. So basically, CNN works with spatial features and RNN helps in solving sequential data.  
This network is used, to give an image as an input, and get an accurate caption as the output. We also employed different searching algorithms to better compare their accuracy on different datasets.

# Install Dependancies
Run the requirements.txt file to install all the dependancies.
`pip install -r requirement.txt`

# The dataset used in the projects are
* FLICKR8K
    * Download from https://forms.illinois.edu/sec/1713398
* MSCOCO
    * Download from http://cocodataset.org/#home
    
    
    
# Execution
The Project can be run using the following code:
`python im2txt/run_inference.py --checkpoint_path="im2txt/model/Hugh/train/newmodel.ckpt-2000000" --vocab_file="im2txt/data
/Hugh/word_counts.txt" --input_files="im2txt/data/images/test.jpg"`
