import tensorflow as tf
import numpy as np

def load_data():
    """
    Loads and preprocesses the CIFAR-10 dataset.
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
        pixel values are normalized to [0, 1].
        Labels are sparse (integers).
    """
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten labels to 1D array
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    load_data()
