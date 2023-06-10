install.packages("devtools")
library("devtools")
library(reticulate)

devtools::install_github("WojtAcht/hms")
library("hms")

library(keras) #might ask for miniconda, wath out for isolation from antivirus

# for images
install.packages("magick")
library(magick)
library(tools)
library("png")

keras <- import("keras")
tf <- import("tensorflow")
install.packages("abind")
library(abind)

# Load model
model <- tf$keras$models$load_model("C:\\backup\\Pulpit\\stochastyka\\NN-architecture-optimizer\\models\\hms_ready_model")

summary(model)


# params len
total_params <- 0
for (layer in model$layers) {
  if (grepl('dense',layer$get_config()$name)) {
    for (param in layer$trainable_weights) {
      param_shape <- dim(param$value())
      param_count <- prod(param_shape)
      total_params <- total_params + param_count
    }
  }
}

cat("Total trainable parameters:", total_params, "\n")




set_model_weights <- function(new_weights_list) {
  # Set weights in model
  index <- 1
  for (layer in model$layers) {
    #cat(summary(layer))
    if (grepl('dense',layer$get_config()$name)) {
      params_to_set <- list()
      # weights
      param <- layer$trainable_weights[[1]]
      param_shape <- dim(param$value())
      param_count <- prod(param_shape)

      #set weights
      chunk <- new_weights_list[index:(index + param_count - 1)]
      weights <- array(unlist(chunk),dim = param_shape)

      index <- index + param_count
      # set biases
      param = layer$trainable_weights[[2]]
      param_shape <- dim(param$value())
      param_count <- prod(param_shape)
      chunk <- new_weights_list[index:(index + param_count - 1)]
      biases <- array(unlist(chunk),dim = param_shape)

      params_to_set <- append(params_to_set,list(chunk))

      index <- index + param_count

      params_to_set[[1]] <- weights
      params_to_set[[2]] <- biases

      layer$set_weights(params_to_set)
    }
  }
}
# Load data for inference

load_data <- function(mono=FALSE, set_path)
{
  images <- list()
  labels <- list()

  # List all files in the set_path directory
  photo_files <- dir(set_path)
  i <- 1
  # Iterate through each photo file
  for (photo in photo_files) {
    array <- readPNG(file.path(set_path, photo))
    #averaging
    if (mono)
    {
      array <- apply(array, c(1, 2), mean)
    }

    images[[i]] <- array
    labels <- append(labels, as.integer(grepl("class1", photo)))
    i <- i + 1
  }
  print(length(images))

  num_samples <- length(labels)

  labels <- keras::to_categorical(labels, num_classes = 2)

  return(list(images = images, labels = labels))
}

dataset <- load_data(FALSE,"C:\\backup\\Pulpit\\stochastyka\\NN-architecture-optimizer\\set")

length(dataset[[1]])


score <- function(samples = 100, dataset)
{
  scores <- 0
  arrays<- NULL
  #TODO faster batch on GPU
  random_indexes <- sample(length(dataset[[1]]), samples)
  for (i in random_indexes)
  {
    if (is.null(arrays))
    {
      arrays <- array(dataset[[1]][[i]], dim = c(1, 50, 50, 3))
    }
    else
    {
      X <- array(dataset[[1]][[i]], dim = c(1, 50, 50, 3))
      arrays <- abind(arrays, X, along = 1)
    }
  }
  predictions <- predict(model, arrays)

  scores <- scores / samples
  return(scores)
}


neural_function <- function(new_weights_list)
{
  set_model_weights(new_weights_list)
  return(score(100,dataset))
}


lower <- rep(-512, total_params)
upper <- rep(512, total_params)
sigma <- list(rep(200, total_params), rep(100, total_params), rep(50, total_params))
ga_config <- list(
  list(
    pmutation = 0.4,
    mutation = rtnorm_mutation(lower, upper, sigma[[1]])
  ),
  list(
    pmutation = 0.2,
    mutation = rtnorm_mutation(lower, upper, sigma[[2]])
  ),
  list(
    pmutation = 0.2,
    mutation = rtnorm_mutation(lower, upper, sigma[[3]])
  )
)
HMS <- hms(
  #fitness = Eggholder,
  fitness = neural_function,
  minimize = FALSE,
  tree_height = 3,
  lower = lower,
  upper = upper,
  run_metaepoch = ga_metaepoch(ga_config),
  population_sizes = c(50, 30, 15),
  sigma = sigma,
  gsc = gsc_max_fitness_evaluations(200),
  sc = sc_max_metric(euclidean_distance, c(40, 20, 10)),
  lsc = lsc_metaepochs_without_improvement(25),
  monitor_level = "none",
  with_gradient_method = TRUE
)
HMS@best_solution






