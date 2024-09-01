# README for RunBEAR Function

## Overview

The `RunCNN` function is designed to perform spatial transcriptomics cluster evaluation using a Convolutional Neural Network (CNN). It leverages neighborhood averaging to preprocess input data before training the model. The function provides an end-to-end solution, including data preparation, model training, evaluation, and prediction generation.

## Prerequisites

This function requires the following R packages:
- `keras`
- `tensorflow`
- `caret`

Make sure that you have installed the necessary Python dependencies for TensorFlow and Keras, as they are integral to the CNN model training and evaluation.

## Input Data

The function expects the input data in the form of an RDS file containing spatial transcriptomics data. Specifically, the input matrix, coordinates, and labels are extracted from the `xenium.obj` object as follows:

- **Matrix**: `xenium.obj@assays$SCT@scale.data`  
- **Coordinates**: Centroid coordinates derived from `xenium.obj@images$crop@boundaries$centroids@coords`
- **Labels**: Cluster labels stored in `xenium.obj$niches`

## Parameters

- **matrix**: The expression matrix containing spatial transcriptomics data.
- **coordinates**: Data frame containing centroid coordinates for each spot.
- **labels**: Vector of cluster labels corresponding to each column in the matrix.
- **nsize**: The neighborhood size used for averaging. Default is `3`.

## Function Workflow

1. **Data Splitting**: 
   - The data is split into training and testing sets using a 70-30 split ratio.

2. **Neighborhood Averaging**:
   - The expression data for each spot in the training and testing sets is averaged with its neighbors, based on the specified `nsize`.

3. **Model Definition**:
   - A 1D Convolutional Neural Network (CNN) model is built using Keras. The model includes convolutional layers, max pooling layers, dropout layers, and dense layers, all configured to classify the cluster labels.

4. **Model Training**:
   - The model is trained using the processed training data, with early stopping based on validation loss to prevent overfitting.

5. **Evaluation and Prediction**:
   - The model is evaluated on the test set, and predictions are generated. The confusion matrix is calculated to assess the model's performance.

## Output

The function returns a list containing:
- **evaluation**: Evaluation metrics of the model on the test set.
- **predictions**: The predicted labels for the test set.
- **confusion_matrix**: The confusion matrix comparing predicted labels to actual labels.

## Example Usage

```r
result <- RunCNN(matrix = matrix, coordinates = coordinates, labels = labels, nsize = 3)
```

This will run the entire workflow and return the evaluation metrics, predictions, and confusion matrix.

## Notes

- Ensure that your R environment is properly configured to use TensorFlow and Keras.
- The `RunBEAR` function is designed for spatial transcriptomics data but can be adapted for other types of spatial data with appropriate modifications.

