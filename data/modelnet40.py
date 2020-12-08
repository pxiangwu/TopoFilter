import os
import torch
import torch.utils.data as data
import numpy as np
import data.provider as provider


def modelnet40_load(root_dir, load_train_data=True, load_test_data=True):
    """
    Load modelnet40 data.
    """
    # ModelNet40 official train/test split
    train_files = []
    test_files = []

    if load_train_data:
        train_files = provider.getDataFiles(
            os.path.join(root_dir, 'modelnet40_ply_hdf5_2048/train_files.txt'))
    if load_test_data:
        test_files = provider.getDataFiles(
            os.path.join(root_dir, 'modelnet40_ply_hdf5_2048/test_files.txt'))

    return train_files, test_files


class ModelNet40(data.Dataset):
    def __init__(self, num_ptrs=1024, random_selection=False, random_rotation=False,
                 random_jitter=False, random_scale=False, random_translation=False, random_dropout=False,
                 split='train', train_ratio=0.8, root_dir='/data/local/pw241/data'):
        self.npoints = num_ptrs
        self.random_rotation = random_rotation
        self.random_selection = random_selection
        self.random_jitter = random_jitter
        self.random_scale = random_scale
        self.random_translation = random_translation
        self.random_dropout = random_dropout

        if root_dir is None:
            self.root_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.root_dir = root_dir

        if split == 'train' or split == 'val':
            data_files, _ = modelnet40_load(self.root_dir, load_train_data=True, load_test_data=False)
        else:
            _, data_files = modelnet40_load(self.root_dir, load_train_data=False, load_test_data=True)

        point_sets, labels = [], []
        for idx in range(len(data_files)):
            data_file_path = os.path.join(self.root_dir, data_files[idx])
            current_data, current_label = provider.loadDataFile(data_file_path)

            point_sets.append(current_data)
            labels.append(current_label)

        point_sets = np.concatenate(point_sets, axis=0)
        labels = np.concatenate(labels, axis=0)

        self.point_sets = point_sets
        self.labels = labels
        self.num_classes = np.unique(self.labels).size

        # create train val split
        res_point_sets = []
        res_labels = []

        if split == 'train' or split == 'val':
            for i in range(self.num_classes):
                select_idx = np.where(self.labels == i)[0]
                select_data = self.point_sets[select_idx]
                select_labels = self.labels[select_idx]

                select_num = len(select_idx)
                train_num = int(select_num * train_ratio)

                if split == 'train':
                    res_point_sets.append(select_data[:train_num])
                    res_labels.append(select_labels[:train_num])
                else:
                    res_point_sets.append(select_data[train_num:])
                    res_labels.append(select_labels[train_num:])

            self.point_sets = np.concatenate(res_point_sets, 0)
            self.labels = np.concatenate(res_labels, 0)

        self.labels = np.squeeze(self.labels).tolist()

    def __getitem__(self, index):
        selected_point_set = self.point_sets[index]

        # randomly sample npoints from the selected point cloud
        if self.random_selection:
            choice = np.random.choice(selected_point_set.shape[0], self.npoints, replace=False)
            sampled_point_set = selected_point_set[choice, :]
        else:
            sampled_point_set = selected_point_set[0:self.npoints, :]

        # random rotation for data augmentation
        if self.random_rotation:
            sampled_point_set = provider.random_rotation_about_origin_instance(sampled_point_set)
            # sampled_point_set = provider.rotate_point_cloud_instance(sampled_point_set)

        if self.random_jitter:
            sampled_point_set = provider.jitter_point_cloud_instance(sampled_point_set)
            # sampled_point_set = np.clip(sampled_point_set, -1.0, 1.0)

        if self.random_scale:
            sampled_point_set = provider.random_scale_point_cloud_instance(sampled_point_set)

        if self.random_translation:
            sampled_point_set = provider.random_translation_point_cloud_instance(sampled_point_set)

        if self.random_dropout:
            sampled_point_set = provider.random_point_dropout_instance_instance(sampled_point_set)

        label = self.labels[index]

        return sampled_point_set.astype(np.float32), label, index

    def __len__(self):
        return len(self.labels)

    def update_corrupted_label(self, noise_label):
        self.labels[:] = noise_label[:]

    def update_selected_data(self, selected_indices):
        self.point_sets = self.point_sets[selected_indices]

        self.labels = np.array(self.labels)
        self.labels = self.labels[selected_indices]
        self.labels = self.labels.tolist()

    def ignore_noise_data(self, noisy_data_indices):
        total = len(self.point_sets)
        remain = list(set(range(total)) - set(noisy_data_indices))
        remain = np.array(remain)

        self.point_sets = self.point_sets[remain]
        self.labels = np.array(self.labels)
        self.labels = self.labels[remain]
        self.labels = self.labels.tolist()

    def get_data_labels(self):
        return self.labels


if __name__ == '__main__':
    from data.show3d_balls import showpoints
    a = ModelNet40(split='train')
    pc, label, _ = a[1500]
    print(label)

    showpoints(pc)

