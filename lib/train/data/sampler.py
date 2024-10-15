import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np


def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. 

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, dataset, num_frames, processing=no_processing, frame_sample_mode='mean'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.dataset = dataset
        self.num_frames = num_frames
        # self.samples_per_epoch = samples_per_epoch
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.dataset.get_dataset_len()

    def __getitem__(self, index):
        return self.getitem(index)

    def getitem(self, seq_id):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """

        # sample a sequence from the given dataset
        # seq_id = self.sample_seq_from_dataset(self.dataset)
        seq_name_info, seq_len, seq_label = self.dataset.get_sequence_info(seq_id)


        if self.frame_sample_mode == 'RANDOMLY':
            frame_ids = self.get_frame_ids_randomly(seq_len)
        elif self.frame_sample_mode == 'UNIFORMLY':
            frame_ids = self.get_frame_ids_uniformly(seq_len)
        else:
            raise ValueError("Illegal frame sample mode")

        # rgb_frames, depth_frams = self.dataset.get_frames(seq_id, frame_ids)
        all_modality_frames = self.dataset.get_frames(seq_id, frame_ids)
        if len(all_modality_frames) == 2:
            data = TensorDict({'rgb_images': all_modality_frames[0],
                               'depth_images': all_modality_frames[1],
                               'label': seq_label,
                               'dataset': self.dataset.get_name()})
        elif len(all_modality_frames) == 3:
            dict = all_modality_frames
            dict['label'] = seq_label
            dict['dataset'] = self.dataset.get_name()
            data = TensorDict(dict)
        # make data augmentation
        data = self.processing(data)

        return data

    def get_frame_ids_uniformly(self, seq_len):
        """Sample frames from the input video."""
        sample_duration = self.num_frames
        frame_indices = []
        for i in range(sample_duration):
            start = int(seq_len * i / sample_duration)
            end = max(int(seq_len * i / sample_duration) + 1, int(seq_len * (i + 1) / sample_duration))
            possible_indices = list(range(start, end))

            if not possible_indices:
                chosen_index = start
            else:
                chosen_index = np.mean(possible_indices, keepdims=True, dtype=int).tolist()

            frame_indices = frame_indices + chosen_index

        return frame_indices

    def get_frame_ids_randomly(self, seq_len):
        """
        Get frame indices from a sequence using a specified interval-based method.

        Parameters:
        - seq_len (int): Total length of the sequence
        - sn (int): Number of frames to retrieve

        Returns:
        - list: List of retrieved frame indices
        """
        sample_duration = self.num_frames
        frame_indices = []
        for i in range(sample_duration):
            start = int(seq_len * i / sample_duration)
            end = max(int(seq_len * i / sample_duration) + 1, int(seq_len * (i + 1) / sample_duration))
            possible_indices = list(range(start, end))

            if not possible_indices:
                chosen_index = start
            else:
                chosen_index = random.choices(possible_indices, k=1)

            frame_indices = frame_indices + chosen_index

        return frame_indices
