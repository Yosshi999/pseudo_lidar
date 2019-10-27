IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath, train_file, calib=True):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disparity/'
    if calib:
        calib_fold = 'calib/'

    with open(train_file, 'r') as f:
        train_idx = [x.strip() for x in f.readlines()]

    left_train = [filepath + '/' + left_fold + img + '.png' for img in train_idx]
    right_train = [filepath + '/' + right_fold + img + '.png' for img in train_idx]
    disp_train_L = [filepath + '/' + disp_L + img + '.npy' for img in train_idx]
    if calib:
        calib_train = [filepath + '/' + calib_fold + img + '.txt' for img in train_idx]
        return left_train, right_train, disp_train_L, calib_train

    return left_train, right_train, disp_train_L
