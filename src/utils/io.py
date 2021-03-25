import numpy as np
import torch
import torchaudio
from PIL import Image


def load_image(path):
    """Loads an image as a numpy array.

    Parameters
    ----------
    path : str
        Image path.

    Returns
    ------
    numpy.ndarray
        The image as numpy array.
    """
    with Image.open(path) as img_file:
        img = np.array(img_file)
    return img


def img_to_bw(input_path, output_path, threshold=127):
    """Converts a grayscale image to a binary (black and white) image.

    Parameters
    ----------
    input_path : str
        Path to the input image.
    output_path : str
        Target location of the output image.
    threshold : int
        Threshold value for converting the image: for pixel values [0, threshold] the converted pixel is black,
        otherwise it is white.
    """
    with Image.open(input_path) as img:
        img = np.array(img)
        if len(img.shape) > 2:
            img = np.min(img, axis=2)

        converted_img = Image.fromarray(np.uint8((img > threshold) * 255))
        converted_img.save(output_path)


def img_to_grayscale(input_path, output_path):
    """Converts an RGB image to a grayscale image.

    Parameters
    ----------
    input_path : str
        Path to the input image.
    output_path : str
        Target location of the output image.

    """
    with Image.open(input_path) as img:
        img.convert('L')
        img.save(output_path)


def load_audio(path):
    """Loads an audio file as a numpy array.

    Parameters
    ----------
    path : str
        Audio file path.

    Returns
    ------
    numpy.ndarray, int
        The audio as numpy array and its sampling rate. Samples are in range [-1,1].
    """
    audio, samplerate = torchaudio.load(path)
    return audio.numpy(), samplerate


def save_audio(path, audio, sample_rate):
    """Stores an audio sequence given as numpy array as audio file with a given sampling rate.

    Parameters
    ----------
    path : str
        Target file path.
    audio : numpy.ndarray
        Audio sequence.
    sample_rate : int
        The sampling rate of the audio sequence.

    Returns
    ------
    numpy.ndarray, int
        The audio as numpy array and its sampling rate.
    """
    # normalize the waveform to lie in range [-1, 1]
    upper = np.max(audio)
    lower = np.min(audio)
    audio = 2.0 * (audio - lower) / (upper - lower) - 1.0
    # save audio file
    torchaudio.save(path, torch.Tensor(audio), sample_rate)
