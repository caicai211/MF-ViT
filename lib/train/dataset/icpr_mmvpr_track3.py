import csv
import os

from lib.train.admin import env_settings
from lib.train.data import opencv_loader
from lib.train.dataset.base_video_dataset import BaseVideoDataset


class ICPR_MMVPR_Track3(BaseVideoDataset):
    def __init__(self, root=None, split=None, modality='RGB', image_loader=opencv_loader):
        super().__init__(name='ICPR_MMVPR_Track3', root=root, image_loader=image_loader)
        root = env_settings().icpr_dir if root is None else root
        self.split = split + '_videofolder.txt'
        if split == 'train':
            self.img_root = os.path.join(self.root, 'training_set')
        elif split == 'test':
            self.img_root = os.path.join(self.root, 'test_set')
        else:
            raise NotImplemented('Unknown split')
        ground_true_path = os.path.join(self.img_root, self.split)
        self.modality = modality
        self.sequence_list, self.seq_lens, self.labels = self._get_sequence_list_and_label(ground_true_path)

    def get_dataset_len(self):
        return len(self.sequence_list)

    def _get_sequence_list_and_label(self, ground_true_path):
        def parse_line(line):
            parts = line[:-1].split(' ')
            label = int(parts[2]) if len(parts) == 3 else 0

            rgb_sequence_name = '/'.join(['rgb_data'] + [parts[0]])
            ir_sequence_name = '/'.join(['ir_data'] + [parts[0]])
            depth_sequence_name = '/'.join(['depth_data'] + [parts[0]])
            all_sequence_name = {}
            if self.modality == 'RGB':
                all_sequence_name['RGB'] = rgb_sequence_name
                seq_len = len(os.listdir(os.path.join(os.path.join(self.img_root, rgb_sequence_name))))
            elif self.modality == 'Depth':
                all_sequence_name['Depth'] = depth_sequence_name
                seq_len = len(os.listdir(os.path.join(os.path.join(self.img_root, depth_sequence_name))))
            elif self.modality == 'IR':
                all_sequence_name['IR'] = ir_sequence_name
                seq_len = len(os.listdir(os.path.join(os.path.join(self.img_root, ir_sequence_name))))
            elif self.modality == 'RTD':
                all_sequence_name['RGB'] = rgb_sequence_name
                all_sequence_name['IR'] = ir_sequence_name
                all_sequence_name['Depth'] = depth_sequence_name
                rgb_seq_len = len(os.listdir(os.path.join(os.path.join(self.img_root, rgb_sequence_name))))
                ir_seq_len = len(os.listdir(os.path.join(os.path.join(self.img_root, ir_sequence_name))))
                depth_seq_len = len(os.listdir(os.path.join(os.path.join(self.img_root, depth_sequence_name))))
                seq_len = int(parts[1])
                min_len = min(rgb_seq_len, depth_seq_len, ir_seq_len)
                assert min_len == seq_len
            elif self.modality == 'RGBT':
                all_sequence_name['RGB'] = rgb_sequence_name
                all_sequence_name['IR'] = ir_sequence_name
                rgb_seq_len = len(os.listdir(os.path.join(os.path.join(self.img_root, rgb_sequence_name))))
                ir_seq_len = len(os.listdir(os.path.join(os.path.join(self.img_root, ir_sequence_name))))
                seq_len = int(parts[1])
                min_len = min(rgb_seq_len, ir_seq_len)
                assert min_len >= seq_len
            elif self.modality == 'RGBD':
                all_sequence_name['RGB'] = rgb_sequence_name
                all_sequence_name['Depth'] = depth_sequence_name
                rgb_seq_len = len(os.listdir(os.path.join(os.path.join(self.img_root, rgb_sequence_name))))
                depth_seq_len = len(os.listdir(os.path.join(os.path.join(self.img_root, depth_sequence_name))))
                seq_len = int(parts[1])
                min_len = min(rgb_seq_len, depth_seq_len)
                assert min_len >= seq_len
            return all_sequence_name, seq_len, label

        all_sequence_list = []
        seq_lens = []
        labels = []
        with open(ground_true_path) as file:
            for line in file.readlines():
                all_sequence_name, seq_len, label = parse_line(line)
                all_sequence_list.append(all_sequence_name)
                labels.append(label)
                seq_lens.append(seq_len)
        return all_sequence_list, seq_lens, labels

    def _get_modal_sequence_path(self, seq_id, modality):
        return os.path.join(self.img_root, self.sequence_list[seq_id][modality])

    def _get_sequence_path(self, seq_id):
        """
        Get the path for the specified modality sequence.

        Parameters:
        - sequence_id (str): The ID of the sequence
        - modality (str): The type of modality (e.g., "RGB", "Depth")

        Returns:
        - str: The full path of the modality sequence
        """
        all_seq_name = self.sequence_list[seq_id]
        if self.modality == 'RTD':
            return (os.path.join(self.img_root, all_seq_name['RGB']),
                    os.path.join(self.img_root, all_seq_name['IR']),
                    os.path.join(self.img_root, all_seq_name['Depth']))
        elif self.modality == 'RGBT':
            return (os.path.join(self.img_root, all_seq_name['RGB']),
                    os.path.join(self.img_root, all_seq_name['IR']))
        elif self.modality == 'RGBD':
            return (os.path.join(self.img_root, all_seq_name['RGB']),
                    os.path.join(self.img_root, all_seq_name['Depth']))
        else:
            return os.path.join(self.img_root, all_seq_name[self.modality])

    def get_sequence_info(self, seq_id):
        """
        Get the length and label of the sequence.

        Parameters:
        - seq_id (str): The ID of the sequence

        Returns:
        - tuple: (length of the sequence, label of the sequence)
        """

        return self.sequence_list[seq_id], self.seq_lens[seq_id], self.labels[seq_id]

    def _get_frame_path(self, seq_path, frame_id, suffix='.jpg'):
        return os.path.join(seq_path, '{:06}'.format(frame_id + 1) + suffix)  # frames start from 1

    def _get_frame(self, seq_path, frame_id, suffix='.jpg'):
        return self.image_loader(self._get_frame_path(seq_path, frame_id, suffix))

    def get_name(self):
        return 'ICPR_MMVPR_Track3'

    def get_frames(self, seq_id, frame_ids):
        """
        Get frames for the specified modality.

        Parameters:
            - seq_id (str): The ID of the sequence
            - frame_ids (list of int): List of frame IDs to retrieve
        Returns:
            - list: List of frame paths
        """
        if self.modality == 'RTD':
            rgb_seq_path, ir_seq_path, depth_seq_path = self._get_sequence_path(seq_id)
            rgb_frame_list = [self._get_frame(rgb_seq_path, f_id, '.jpg') for f_id in frame_ids]
            ir_frame_list = [self._get_frame(ir_seq_path, f_id, '.jpg') for f_id in frame_ids]
            depth_frame_list = [self._get_frame(depth_seq_path, f_id, '.png') for f_id in frame_ids]
            all_frames = {"rgb_images": rgb_frame_list, "ir_images": ir_frame_list, "depth_images": depth_frame_list}
            return all_frames
        elif self.modality == 'RGBT':
            rgb_seq_path, ir_seq_path = self._get_sequence_path(seq_id)
            rgb_frame_list = [self._get_frame(rgb_seq_path, f_id, '.jpg') for f_id in frame_ids]
            ir_frame_list = [self._get_frame(ir_seq_path, f_id, '.jpg') for f_id in frame_ids]
            all_frames = {"rgb_images": rgb_frame_list, "ir_images": ir_frame_list, "depth_images": None}
            return all_frames
        elif self.modality == 'RGBD':
            rgb_seq_path, depth_seq_path = self._get_sequence_path(seq_id)
            rgb_frame_list = [self._get_frame(rgb_seq_path, f_id, '.jpg') for f_id in frame_ids]
            depth_frame_list = [self._get_frame(depth_seq_path, f_id, '.png') for f_id in frame_ids]
            all_frames = {"rgb_images": rgb_frame_list, "ir_images": None, "depth_images": depth_frame_list}
            return all_frames
        else:
            suffix = '.png' if self.modality == 'Depth' else '.jpg'
            modal_seq_path = self._get_sequence_path(seq_id)
            modal_frame_list = [self._get_frame(modal_seq_path, f_id, suffix) for f_id in frame_ids]
            if self.modality == 'RGB':
                return {"rgb_images": modal_frame_list, "ir_images": None, "depth_images": None}
            elif self.modality == 'IR':
                return None, {"rgb_images": None, "ir_images": modal_frame_list, "depth_images": None}
            elif self.modality == 'Depth':
                return {"rgb_images": None, "ir_images": None, "depth_images": modal_frame_list}

