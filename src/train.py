import os
import argparse
import tensorflow as tf
from data_loader import load_data
from model import create_model
import matplotlib.pyplot as plt

def plot_history(history, save_path='history.png'):
    """Plots and saves training history."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")

def train(epochs=20, batch_size=64, model_save_path='cif10.keras'):
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Create model
    model = create_model()
    model.summary()
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    # Train
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping]
    )
    
    # Save model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot history
    plot_history(history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CNN on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--model_path', type=str, default='cifar10_model.keras', help='Path to save model')
    
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, model_save_path=args.model_path)
