# ImageSimilarityDetection
**Method1:** Image Similarity Detection in Action with Tensorflow 2.0 - Annoy and Angular. Here is the medium article for more details on the approach: https://towardsdatascience.com/image-similarity-detection-in-action-with-tensorflow-2-0-b8d9a78b2509. I got the code from the repo referenced in that article. 
**Method2:** Giovanni's method

## Usecase: Finger print similarity detection
**Research question:** Is it possible to predict if two people are related based on their fingerprints?
**Application:** Finding parents of lost childen such as refugees separated at the border or children who have been kidnapped. 
**Dataset:** https://data.mendeley.com/datasets/8rkbvxhhmj/1 (local path = /Users/canlilareden/Documents/fingerprint-data/Data Image Fingerprint)

## Requirements
Python 3.8 +
Linux
Additional requirements listed in the requirements.txt files in the respective folders

## Recommended getting started guide
First create a virtual environment for EACH method
Note: It's recommend to maintain all of your venvs in a directory OUTSIDE of your Github folder
    ```
    python3 -m venv /Users/canlilareden/Documents/venvs/image-sim-venv1
    python3 -m venv /Users/canlilareden/Documents/venvs/image-sim-venv2
    ```
Then activate the venv of your chosing as follows (example):
    ```
    source /Users/canlilareden/Documents/venvs/image-sim-venv1/bin/activate
    or 
    source /Users/canlilareden/Documents/venvs/image-sim-venv2/bin/activate
    ```
Then install the necessary dependencies for whatever method you are using:
For **method 2** you will need to install a Jupyter kernel
    ```
    <!-- python3 -m ipykernel install --user --name=method2 -->
    pip3 install jupyterlab
    jupyter-lab
    ```