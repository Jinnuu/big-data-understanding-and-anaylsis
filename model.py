import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def create_cnn_lstm_model(input_shape, num_features, num_classes):
    """
    Create a CNN-LSTM model for weather prediction
    
    Args:
        input_shape (tuple): Shape of input data (time_steps, features)
        num_features (int): Number of input features
        num_classes (int): Number of output classes/predictions
    
    Returns:
        model: Compiled CNN-LSTM model
    """
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # LSTM layers for temporal dependencies
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(64),
        Dropout(0.2),
        BatchNormalization(),
        
        # Dense layers for prediction
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes)
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def prepare_data_for_cnn_lstm(data, time_steps):
    """
    Prepare data for CNN-LSTM model
    
    Args:
        data (numpy.ndarray): Input data
        time_steps (int): Number of time steps to look back
    
    Returns:
        X (numpy.ndarray): Input sequences
        y (numpy.ndarray): Target values
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """
    Train the CNN-LSTM model
    
    Args:
        model: CNN-LSTM model
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        batch_size: Batch size for training
        epochs: Number of epochs
    
    Returns:
        history: Training history
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    return history 