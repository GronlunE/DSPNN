import numpy as np
import tensorflow as tf
import pandas as pd
from scipy.io import loadmat
from config import *
from neuralnet import sylnet_model, TransformerBlock
from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError


def netTrain(tensor, syllables, epochs, batch_size, n_channels):
    """

    :param tensor:
    :param syllables:
    :param epochs:
    :param batch_size:
    :param n_channels:
    :return:
    """

    tensor[tensor == -np.inf] = 20*np.log10(eps)
    print("Tensor dimensions:", np.shape(tensor))
    print("Syllable dimensions:", np.shape(syllables))

    model = sylnet_model(tensor, n_channels)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mean_absolute_percentage_error',
        metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
    )

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_percentage_error', patience=10)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=r"resources/most_recent_model",
        monitor='val_mean_absolute_percentage_error',
        mode='min',
        save_best_only=True,
        custom_objects={'TransformerBlock': TransformerBlock}
        )

    # Train the model
    history = model.fit(tensor, syllables,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[earlystop, model_checkpoint_callback],
                        validation_split = 0.2)

    return model, history


def netTest(tensor, syllables, model, batch_size, language):
    """

    :param tensor:
    :param syllables:
    :param batch_size:
    :param model:
    :return:
    """

    print("\nLoading testing data...", "\n")

    # Find indices of tensors with duration less than or equal to T seconds
    csv = r"resources/csv" + "\\" + language + ".csv"
    df = pd.read_csv(csv)
    audio_durations = df['Audio duration'].to_numpy()
    T = tensor.shape[1] / 100
    valid_indices = np.where(audio_durations <= T)[0]

    # Filter to only keep valid indices
    tensor = tensor[valid_indices, :, :]
    syllables = syllables[valid_indices]

    # Remove any zeros from the annotations
    valid_indices = np.where(syllables != 0)[0]
    tensor = tensor[valid_indices, :, :]
    syllables = syllables[valid_indices]

    # Find the indices of the NaN values in the tensor
    nan_indices = np.argwhere(np.isnan(tensor))

    # Create a mask for the non-NaN values in the tensor
    mask = np.ones(tensor.shape[0], dtype=bool)
    mask[nan_indices[:, 0]] = False

    # Remove the NaN values from the tensor and update the label array
    tensor = tensor[mask]
    syllables = syllables[mask]

    print("Tensor dimensions:", np.shape(np.array(tensor)))
    print("Syllable dimensions:", np.shape(np.array(syllables)), "\n")

    syll_estimates = model.predict(tensor, batch_size=batch_size)

    mae = np.nanmean(np.abs(syll_estimates[:, 0, 0] - syllables))
    mape = np.nanmean(np.abs(syll_estimates[:, 0, 0] - syllables) / syllables) * 100

    print(f"\nMeanAbsoluteError: {mae}")
    print(f"MeanAbsolutePercentageError: {mape}\n")

    return mae, mape


def construct_data(languages=training_languages, samples=10000, test=False, root=training_loc):
    """

    :param test:
    :param root:
    :param languages:
    :param samples:
    :return:
    """
    total_tensor = np.empty((samples,400,40))
    total_syllables = np.empty((samples,1))
    for i in range(len(languages)):

        language = languages[i]

        mat_loc = root + "\\" + language
        data = loadmat(mat_loc)
        tensor = data["tensor"]
        syllables = np.transpose(data["syllables"])

        N = tensor.shape[0]

        if test:
            syllables = np.transpose(data["true_syllables"])
            samples = N

        ord_ = np.arange(N)
        np.random.shuffle(ord_)
        tensor = tensor[ord_, :, :]
        syllables = syllables[ord_]

        tensor = tensor[0:samples,:,:]
        syllables = syllables[0:samples]

        if i == 0:
            total_tensor = tensor
            total_syllables = syllables
        else:
            total_tensor = np.concatenate((total_tensor, tensor))
            total_syllables = np.concatenate((total_syllables, syllables))

    N = total_tensor.shape[0]
    ord_ = np.arange(N)
    np.random.shuffle(ord_)
    total_tensor = total_tensor[ord_, :, :]
    total_syllables = total_syllables[ord_]

    return total_tensor, total_syllables
