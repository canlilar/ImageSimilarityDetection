
**Source:** https://towardsdatascience.com/image-similarity-detection-in-action-with-tensorflow-2-0-b8d9a78b2509 

# Step 0)
First we need to activate the virtual env we created in the main read me and activate it. Then we can install the dependencies. 
```
python3 -m venv /Users/canlilareden/Documents/venvs/image-sim-venv1
source /Users/canlilareden/Documents/venvs/image-sim-venv1/bin/activate
pip3 install -r method1/requirements.txt
```

# Step 1) Generate Image Feature Vectors: get_image_feature_vectors.py
The main purpose of this script is to generate image feature vectors by reading image files located in a local folder. It has two functions: load_img() and get_image_feature_vectors().

**load_img(path)** gets file names which are provided as an argument of the function. Then loads and pre-process the images so that we can use them in our MobilenetV2 CNN model.

### Pre-processing steps are as follows;

- Decoding the image to W x H x 3 shape tensor with the data type of integer.
- Resizing the image to 224 x 224 x 3 shape tensor as the version of the MobilenetV2 model we use expects that specific image size.
- Converting the data type of tensor to float and adding a new axis to make tensor shape 1 x 224 x 224 x 3. This is the exact input shape expected by the model.

**get_image_feature_vectors()** function is where I extract the image feature vectors. You can see below, step by step definition of what this function does;

- Loads the MobilenetV2 model using Tensorflow Hub
- Loops through all images in a local folder and passing them to load_img(path) function
- Infers the image feature vectors
- Saves each one of the feature vectors to a separate file for later use
- Here is the link to find other feature vector models to try out on Tensor hub: https://tfhub.dev/s?module-type=image-feature-vector

# Step 2 How to Use Spotify/Annoy Library to Calculate the Similarity Scores
- What is Spotify/Annoy Library?
Annoy (Approximate Nearest Neighbor Oh Yeah), is an open-sourced library for approximate nearest neighbor implementation. I will use it to find the image feature vectors in a given set that is closest (or most similar) to a given feature vector. There are just two main parameters needed to tune Annoy: 
1. the number of trees **n_trees**
2. the number of nodes to inspect during searching **search_k**.

- **n_trees** is provided during build time and affects the build time and the index size. A larger value will give more accurate results, but larger indexes.
- **search_k** is provided in runtime and affects the search performance. A larger value will give more accurate results, but will take longer time to return.

## Let’s Calculate Similarity Scores: cluster_image_feature_vectors.py
The main purpose of this script is to calculate image similarity scores using image feature vectors we have just generated in the previous chapter. It has two functions: 
1. match_id(filename)
2. cluster().

**cluster()** function does the image similarity calculation with the following process flow:
- Builds an annoy index by appending all image feature vectors stored in the local folder
- Calculates the nearest neighbors and similarity scores
- Saves and stores the information in a JSON file for later use.

**match_id(filename)** is a helper function as I need to match images with the product id’s to enable visual product search in my web application. There is a JSON file that contains all the product id information matched with the product image names. This function retrieves the product id information for a given image file name using that JSON file.

# Take aways

## Take aways from using the mobilenet_v2_140_224 pre-trained model from Tensor Hub
https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/5
The approach is highly precise (92%) however it lacks sensitivity (ranges from .48 to .51). This might be a result of the pretrained model we are using. We really need to maximize the sensitivity (recall) for this exercise because we need to focus on maximizing the quantity of reunified families even if sometimes we misfire. I suppose it doesn’t matter too much if we are wrong with a match because they’re going to qualify the match anyway before they actually unite the families right? 

## Take aways from using the Inception V2 pre-trained model from Tensor Hub
https://tfhub.dev/google/imagenet/inception_v2/feature_vector/5
The approach got us a slight boost in precision (93.9%) however it STILL lacks sensitivity (ranges from .48 to .51). Did not move the needle at all here.

## Take aways from using the ResNet V2 101 pre-trained model from Tensor Hub
Source: https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/5
This approach did worst of all. Precision was awful (.18). Recall remained constant at 0.508 no matter what threshold we used. Total failure. 

## Take aways from using the NASNet-A (mobile) pre-trained model from Tensor Hub
Source: https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/5
This approach did terible as well. Precision was bad (.694). Recall ranged from .495 to .507 depending on the threshold set. 
