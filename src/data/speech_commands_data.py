import torch
from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommandsData:
    """
    This is a wrapper class around the SpeechCommands <https://arxiv.org/abs/1804.03209> dataset provided via
    torchaudio's SPEECHCOMMANDS interface.
    """

    def __init__(self, data_dir='../../data/audio/'):
        self.data_dir = data_dir
        # extract labels from the dataset
        test_set = SPEECHCOMMANDS(self.data_dir, subset='testing')
        self.labels = sorted(list(set(datapoint[2] for datapoint in test_set)))

    def get_subset(self, subset: str):
        """
        Get a subset of the speech commands dataset.

        Parameters
        ----------
        subset : str
            Subset identifier. Has to be in ['training', 'validation', 'testing'].

        Returns
        ----------
        torchaudio.datasets.SPEECHCOMMANDS
            The respective subset of the SC dataset.
        """
        return SPEECHCOMMANDS(self.data_dir, subset=subset)

    def get_dataloader(self, dataset, batch_size=256, shuffle=False):
        """
        Create a dataloader for subset of the SC dataset.

        Parameters
        ----------
        dataset : torchaudio.datasets.SPEECHCOMMANDS
            Subset of the SC dataset.
        batch_size : int
            Batch size..
        shuffle : bool
            Whether to shuffle the data before batching or not.

        Returns
        ----------
        torch.utils.data.DataLoader
            The dataloader.
        """
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )
        return data_loader

    @classmethod
    def pad_sequence(cls, sequences):
        """
        Applies zero padding to a list of sequences to get a batch of equal length sequences.

        Parameters
        ----------
        sequences : list
            List of batch_size number of sequences of shape (num_channels, length_i) where length_i can vary for
            each sequence.

        Returns
        ----------
        torch.Tensor
            A batch of equal length sequences of shape (batch_size, num_channels, length).
        """
        # ensure that all waveforms in the batch have the same length by applying zero-padding
        sequences = [sequence.T for sequence in sequences]
        sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.)
        return sequences.permute(0, 2, 1)

    def collate_fn(self, batch):
        """
        Takes a batch of SC data points and extracts the waveforms and targets.

        Parameters
        ----------
        batch : list
            List of batch_size data points of the SC dataset. Each data point is a
            tuple of (waveform, sample_rate, label, speaker_id, utterance_number).

        Returns
        ----------
        torch.Tensor, torch.Tensor
            A tensor of padded waveforms and a batch of targets.
        """

        # extract waveforms from batch and apply padding
        waveforms = self.pad_sequence([waveform for waveform, *_ in batch])

        # extract labels from batch and convert them to indices
        targets = torch.stack([torch.tensor(self.labels.index(label)) for _, _, label, *_ in batch])

        return waveforms, targets

    @classmethod
    def create_reduced_lists(cls, datadir='../../data/SpeechCommands/speech_commands_v0.02/', selected_words=None):
        """
        Extracts a subset of words from the testing and validation lists of the original SpeechCommands dataset and
        creates new lists with the relevant entries.

        Parameters
        ----------
        datadir : str
            Directory where the lists reside.
        selected_words : set
            Set of words to extract.

        """
        if selected_words is None:
            selected_words = {'yes', 'no', 'stop', 'go', 'four'}

        orig_lists = ['testing_list.txt', 'validation_list.txt']
        reduced_lists = ['testing_list_reduced.txt', 'validation_list_reduced.txt']

        for orig_list, reduced_list in zip(orig_lists, reduced_lists):
            with open(datadir + orig_list, 'r') as f_orig:
                with open(datadir + reduced_list, 'w') as f_reduced:
                    line = f_orig.readline()
                    while line:
                        word, *_ = line.split('/')
                        if word in selected_words:
                            f_reduced.write(line)
                        line = f_orig.readline()
