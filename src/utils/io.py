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
    return np.flipud(img)  # flip image s.t. origin is at bottom left corner


def img_to_bw(input_path, output_path=None, threshold=127, bg_color=255):
    """Converts a grayscale image to a binary (black and white) image.

    Parameters
    ----------
    input_path : str
        Path to the input image.
    output_path : str
        Target location of the output image. If None, image is not stored.
    threshold : int
        Threshold value for converting the image: for pixel values [0, threshold] the converted pixel is black,
        otherwise it is white.
    bg_color : int
        Background color (grayscale) to be applied if image has a transparency channel. 255 corresponds to white.
    """
    with Image.open(input_path) as img:
        if img.mode == 'LA':
            converted_img = Image.new('L', img.size, bg_color)
            converted_img.paste(img, mask=img.split()[1])

        np_img = np.uint8((np.array(converted_img) > threshold) * 255)
        converted_img = Image.fromarray(np_img)

    if output_path is not None:
        converted_img.save(output_path)

    return np.flipud(np_img)  # flip image s.t. origin is at bottom left corner


def img_to_grayscale(input_path, output_path):
    """Converts an RGB image to a grayscale image.

    Parameters
    ----------
    input_path : str
        Path to the input image.
    output_path : str
        Target location of the output image. If None, image is not stored.

    """
    with Image.open(input_path) as img:
        converted_img = img.convert('L')

    if output_path is not None:
        converted_img.save(output_path)

    return np.flipud(np.array(converted_img))  # flip image s.t. origin is at bottom left corner


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
