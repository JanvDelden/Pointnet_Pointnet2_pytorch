import torch


def get_device(cuda_preference=True):
    print('cuda available:', torch.cuda.is_available(),
          '; cudnn available:', torch.backends.cudnn.is_available(),
          '; num devices:', torch.cuda.device_count())

    use_cuda = False if not cuda_preference else torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    device_name = torch.cuda.get_device_name(device) if use_cuda else 'cpu'
    print('Using device', device_name)
    return device


def gen_split(percentages=(0.5, 0.2),
              paths=("/content/Pointnet_Pointnet2_pytorch/data/trainsplit.npy",
                     "/content/Pointnet_Pointnet2_pytorch/data/valsplit.npy"),
              sample_number=255,
              shuffle=True,
              seed=1):
    import random
    import numpy as np
    if shuffle:
        random.seed(seed)
        indices = range(sample_number)
        indices = np.array(random.sample(indices, sample_number))
    else:
        indices = np.arange(0, sample_number)
    start, percentage = 0, 0
    for i, (path) in enumerate(paths):
        percentage += percentages[i]
        stop = np.floor(percentage * sample_number).astype(int)
        index_subset = indices[start:stop]
        np.save(path, index_subset)
        start = stop



