library(keras)
library(tensorflow)
library(caret)  # for confusionMatrix function

xenium.obj <- readRDS("~/spatial_cluster_evaluation/manuscript_spatial_transcriptomics_cluster_evaluation_2024/XeniumData.RDS")

matrix <- xenium.obj@assays$SCT@scale.data
coordinates <- cbind(as.numeric(colnames(xenium.obj)),
                     xenium.obj@images$crop@boundaries$centroids@coords)

coordinates <- data.frame(coordinates)
coordinates[,2] <- round(as.numeric(coordinates[,2]))
coordinates[,3] <- round(as.numeric(coordinates[,3]))

labels <- xenium.obj$niches

RunBEAR <- function(matrix = matrix, coordinates = coordinates, labels = labels, nsize = 3) {
  set.seed(22)
  
  # Train/test split
  ind <- sample(2, ncol(matrix), replace = TRUE, prob = c(0.7, 0.3))
  
  # Isolate the input for training
  x_train <- matrix[, ind == 1]
  y_train <- labels[ind == 1]
  
  y_train <- as.vector(as.character(y_train))
  y_train_vector <- as.integer(as.factor(y_train)) - 1
  num_classes <- length(unique(y_train_vector))
  
  coords <- coordinates[ind == 1, ]
  avg_train <- x_train
  
  for (i in unique(y_train_vector)) {
    idx <- which(y_train_vector == i)
    wc <- coords[idx, ]
    
    for (j in seq_along(idx)) {
      roi <- wc[j, 2]
      coi <- wc[j, 3]
      neighs <- which((wc[, 2] %in% c((roi-nsize):(roi+nsize))) &
                        (wc[, 3] %in% c((coi-nsize):(coi+nsize))))
      if (length(neighs) > 1) {
        avg_train[, idx[j]] <- rowMeans(x_train[, idx[neighs]])
      }
    }
  }
  
  message("Finished training neighborhood averaging")
  
  x_test <- matrix[, ind == 2]
  y_test <- labels[ind == 2]
  
  y_test <- as.vector(as.character(y_test))
  y_test_vector <- as.integer(as.factor(y_test)) - 1
  
  wc <- coordinates[ind == 2, ]
  avg_test <- x_test
  
  for (j in seq_len(ncol(x_test))) {
    roi <- wc[j, 2]
    coi <- wc[j, 3]
    neighs <- which((wc[, 2] %in% c((roi-nsize):(roi+nsize))) &
                      (wc[, 3] %in% c((coi-nsize):(coi+nsize))))
    if (length(neighs) > 1) {
      avg_test[, j] <- rowMeans(x_test[, neighs])
    }
  }
  
  message("Finished testing neighborhood averaging")
  
  input_shape <- c(nrow(x_train), 1)
  
  x_train <- tensorflow::tf$reshape(x_train, shape = tensorflow::tf$cast(c(as.integer(ncol(x_train)), as.integer(nrow(x_train)), 1), tensorflow::tf$int32))
  x_test <- tensorflow::tf$reshape(x_test, shape = tensorflow::tf$cast(c(as.integer(ncol(x_test)), as.integer(nrow(x_test)), 1), tensorflow::tf$int32))
  
  # Convert y_train and y_test to one-hot encoding
  y_train <- to_categorical(y_train_vector, num_classes)
  y_test <- to_categorical(y_test_vector, num_classes)
  
  # Calculate class weights using original y_train_vector before one-hot encoding
  class_weights <- table(y_train_vector)
  class_weights <- max(class_weights) / class_weights
  
  # Convert to a named list
  class_weights_list <- as.list(as.numeric(class_weights))
  names(class_weights_list) <- as.character(seq(0, num_classes - 1))
  
  cnn_model <- keras_model_sequential() %>%
    layer_conv_1d(filters = 128, kernel_size = 3, activation = 'relu', 
                  input_shape = input_shape, 
                  kernel_regularizer = regularizer_l2(0.01)) %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_conv_1d(filters = 96, kernel_size = 3, activation = 'relu', 
                  input_shape = input_shape, 
                  kernel_regularizer = regularizer_l2(0.01)) %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_conv_1d(filters = 64, kernel_size = 3, activation = 'relu', 
                  kernel_regularizer = regularizer_l2(0.01)) %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_conv_1d(filters = 32, kernel_size = 3, activation = 'relu', 
                  kernel_regularizer = regularizer_l2(0.01)) %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_dropout(rate = 0.3) %>%
    layer_flatten() %>%
    layer_dense(units = 16, activation = 'relu', 
                kernel_regularizer = regularizer_l2(0.01)) %>%
    layer_dense(units = num_classes, activation = 'softmax')
  
  summary(cnn_model)
  
  legacy_adam <- tf$keras$optimizers$legacy$Adam(learning_rate = 3e-05)
  
  cnn_model %>% compile(
    loss = loss_categorical_crossentropy,
    optimizer = legacy_adam,
    metrics = c('accuracy')
  )
  
  message("Starting model training...")
  
  early_stopping <- callback_early_stopping(
    monitor = "val_loss",
    patience = 10,
    restore_best_weights = TRUE
  )
  
  cnn_history <- cnn_model %>% fit(
    x_train, y_train,
    batch_size = 32,
    epochs = 100,
    validation_split = 0.2,
    callbacks = list(early_stopping),
    class_weight = class_weights_list
  )
  
  message("Model training completed.")
  
  evaluation <- cnn_model %>% evaluate(x_test, y_test)
  
  message("Model evaluation completed.")
  print(evaluation)
  
  cnn_pred <- cnn_model %>%
    predict(x_test) %>% k_argmax()
  
  cnn_pred <- as.vector(cnn_pred)
  
  message("Predictions completed.")
  print(head(cnn_pred, 50))
  
  y_test_labels <- apply(y_test, 1, which.max) - 1
  y_test_labels <- factor(y_test_labels)
  cnn_pred <- factor(cnn_pred, levels = levels(y_test_labels))
  
  print("Levels in y_test_labels:")
  print(levels(y_test_labels))
  print("Levels in cnn_pred:")
  print(levels(cnn_pred))
  
  confusion <- confusionMatrix(cnn_pred, y_test_labels)
  
  message("Confusion matrix generated.")
  print(confusion)
  
  return(list(evaluation = evaluation, predictions = cnn_pred, confusion_matrix = confusion))
}

result <- RunBEAR(matrix = matrix, coordinates = coordinates, labels = labels, nsize = 3)


