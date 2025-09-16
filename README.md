# PyTorch Learning Repository

This repository contains Jupyter notebooks created while learning [**PyTorch**](https://pytorch.org/).  
Each notebook corresponds to a specific lesson or concept, covering topics from basic tensor operations to advanced deep learning techniques.  
Along with code, some theoretical explanations are included so learners can build a strong foundation.


## Contents

1. **01_tensors_in_pytorch.ipynb**  
   - *Theory:* Tensors are the core data structure in PyTorch, similar to NumPy arrays but with GPU acceleration.  
   - Covers creation, indexing, reshaping, and basic operations on tensors.  

2. **02_pytorch_autograd.ipynb**  
   - *Theory:* Autograd provides automatic differentiation for building and training neural networks.  
   - Demonstrates how gradients are calculated and propagated during backpropagation.  

3. **03_pytorch_training_pipeline.ipynb**  
   - *Theory:* A training pipeline typically includes dataset preparation, model definition, loss function, and optimization.  
   - Shows a simple step-by-step training loop.  

4. **04_pytorch_nn_module.ipynb**  
   - *Theory:* `nn.Module` is the base class for all PyTorch models. It helps structure code and manage parameters.  
   - Example of building custom models.  

5. **05_pytorch_training_pipeline_using_nn_module.ipynb**  
   - Extends the training pipeline with `nn.Module` for cleaner and reusable code.  

6. **06_dataset_and_dataloader_demo.ipynb**  
   - *Theory:* `Dataset` represents a collection of data, and `DataLoader` helps load data in batches with shuffling and parallelism.  

7. **07_pytorch_training_pipeline_using_dataset_and_dataloader.ipynb**  
   - Full training pipeline with datasets and dataloaders.  

8. **08_ann_fashion_mnist_pytorch.ipynb**  
   - *Theory:* Artificial Neural Networks (ANNs) are the foundation of deep learning.  
   - Trains an ANN on Fashion MNIST dataset.  

9. **09_ann_fashion_mnist_pytorch_gpu.ipynb**  
   - Demonstrates GPU acceleration with CUDA.  

10. **10_ann_fashion_mnist_pytorch_gpu_optimized.ipynb**  
    - Shows performance optimizations for ANN models.  

11. **11_ann_fashion_mnist_pytorch_gpu_optimized_v2.ipynb**  
    - Further optimized ANN training.  

12. **12_cnn_fashion_mnist_pytorch_gpu.ipynb**  
    - *Theory:* Convolutional Neural Networks (CNNs) are specialized for image tasks.  
    - Trains a CNN on Fashion MNIST dataset using GPU.  

13. **13_cnn_optuna.ipynb**  
    - *Theory:* Hyperparameter tuning is essential to improve model performance.  
    - Uses **Optuna** to optimize hyperparameters.  

14. **14_transfer_learning_fashion_mnist_pytorch_gpu.ipynb**  
    - *Theory:* Transfer learning uses pretrained models to speed up training and improve accuracy with limited data.  

15. **15_pytorch_rnn_based_qa_system.ipynb**  
    - *Theory:* Recurrent Neural Networks (RNNs) are designed for sequential data.  
    - Builds a simple QA system with RNNs.  

16. **16_pytorch_lstm_next_word_predictor.ipynb**  
    - *Theory:* LSTMs are a type of RNN that overcome vanishing gradient problems.  
    - Implements a next-word prediction model.  


## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/Rohini-Koli9/PyTorch.git

2. Install dependencies:

   ```bash
   pip install torch torchvision torchaudio matplotlib optuna
   ```
3. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```
4. Explore the notebooks in order for a structured learning path

## Purpose

This repository is maintained as a **learning resource** for PyTorch.
It provides both **theory** and **practical implementation** so learners can connect concepts with code.

## License

This project is for learning purposes. Feel free to fork and use the code for learning all credit goes to [@campusx-official](https://github.com/campusx-official).
