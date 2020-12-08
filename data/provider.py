import os
import sys
import numpy as np
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# # Download dataset for point cloud classification
# DATA_DIR = os.path.join(BASE_DIR, '../data')
# if not os.path.exists(DATA_DIR):
#     os.mkdir(DATA_DIR)
# if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
#     www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#     zipfile = os.path.basename(www)
#     os.system('wget %s; unzip %s' % (www, zipfile))
#     os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
#     os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_instance(points):
    """
    Randomly rotate the individual point cloud to augment the dataset
    Rotation is along the up direction
    Input:
        Nx3 array
    Output:
        Nx3 array, representing the rotated point cloud
    """
    rotated_data = np.zeros(points.shape, dtype=np.float32)

    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    shape_pc = points[:]
    rotated_data[:] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def random_rotation_about_origin_instance(points):
    rotation_angle_x = np.random.uniform() * 2 * np.pi
    rotation_angle_y = np.random.uniform() * 2 * np.pi
    rotation_angle_z = np.random.uniform() * 2 * np.pi

    rotated_data = random_rotation_about_origin(points, rotation_angle_x, rotation_angle_y, rotation_angle_z)

    return rotated_data


def random_rotation_about_origin(points, rotation_angle_x, rotation_angle_y, rotation_angle_z):
    rotated_pc = rotate_x_dir(points, rotation_angle_x)
    rotated_pc = rotate_y_dir(rotated_pc, rotation_angle_y)
    rotated_pc = rotate_z_dir(rotated_pc, rotation_angle_z)

    return rotated_pc


def rotate_y_dir(points, rotation_angle):
    rotated_data = np.zeros(points.shape, dtype=np.float32)

    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    shape_pc = points[:]
    rotated_data[:] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def rotate_z_dir(points, rotation_angle):
    rotated_data = np.zeros(points.shape, dtype=np.float32)

    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    shape_pc = points[:]
    rotated_data[:] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def rotate_x_dir(points, rotation_angle):
    rotated_data = np.zeros(points.shape, dtype=np.float32)

    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    shape_pc = points[:]
    rotated_data[:] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def jitter_point_cloud_instance(point_data, sigma=0.01, clip=0.05):
    N, C = point_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += point_data
    return jittered_data


def random_scale_point_cloud_instance(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, 3)
    batch_data[:, 0] *= scales[0]
    # batch_data[:, 1] *= scales[1]  # vertical direction
    batch_data[:, 2] *= scales[2]
    return batch_data


def random_translation_point_cloud_instance(batch_data, offset_low=-0.2, offset_high=0.2):
    offsets = np.random.uniform(offset_low, offset_high, 3)
    batch_data[:, 0] += offsets[0]
    batch_data[:, 1] += offsets[1]  # vertical direction
    batch_data[:, 2] += offsets[2]
    return batch_data


def random_point_dropout_instance_instance(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    dropout_ratio =  np.random.random()*max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        batch_pc[drop_idx, :] = batch_pc[0, :]  # set to the first point
    return batch_pc


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    f.close()
    return (data, label)


def loadDataFile(filename):
    return load_h5(filename)


def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    f.close()
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)
