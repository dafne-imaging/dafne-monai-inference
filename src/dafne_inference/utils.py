import numpy as np
from skimage.transform import resize

def resample_image(image, shape, anisotrophy_flag):
    '''
    Docstring per resample_image
    
    :param image: image to resample
    :param shape: image shape
    :param anisotrophy_flag: True if image is anisotrophy
    '''
    if image.ndim == 4:
        image_list = image
        is_multichannel = True
    else:
        image_list = [image]
        is_multichannel = False

    resized_channels = []

    for image_c in image_list:
        # image_c: np.array(Depth, Height, Width)
        
        if anisotrophy_flag:
            resized_slices = []
            target_2d_shape = shape[1:]

            for i in range(image_c.shape[0]):
                image_c_2d_slice = image_c[i, :, :]
                image_c_2d_slice = resize(
                    image_c_2d_slice,
                    target_2d_shape,
                    order=3,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                resized_slices.append(image_c_2d_slice)
            resized = np.stack(resized_slices, axis=0)
            
            if resized.shape[0] != shape[0]:
                resized = resize(
                    resized,
                    shape,
                    order=1,
                    mode="constant",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
            resized_channels.append(resized)

        else:
            resized = resize(
                image_c,
                shape,
                order=3,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            resized_channels.append(resized)
    
    if is_multichannel:
        return np.stack(resized_channels, axis=0)
    else:
        return resized_channels[0]

def _center_crop(vol, target_shape):
    slices = tuple(
        slice((vol.shape[i] - target_shape[i]) // 2,
              (vol.shape[i] - target_shape[i]) // 2 + target_shape[i])
        for i in range(vol.ndim)
    )
    return vol[slices]


def _resample_prediction(pred_vol, target_shape, anisotrophy_flag):
    n_class = int(pred_vol.max())
    resampled = np.zeros(target_shape, dtype=np.int8)

    if anisotrophy_flag:
        shape_2d = target_shape[1:]
        depth = pred_vol.shape[0]
        resampled_2d = np.zeros((depth, *shape_2d), dtype=np.int8)

        for cls in range(1, n_class + 1):
            for d in range(depth):
                mask = (pred_vol[d] == cls).astype(float)
                if not np.any(mask):
                    continue
                resized = resize(mask, shape_2d, order=0, mode='edge',
                                 cval=0, clip=True, anti_aliasing=False)
                resampled_2d[d][resized >= 0.5] = cls

        if depth != target_shape[0]:
            for cls in range(1, n_class + 1):
                mask = (resampled_2d == cls).astype(float)
                if not np.any(mask):
                    continue
                resized = resize(mask, target_shape, order=0, mode='constant',
                                 cval=0, clip=True, anti_aliasing=False)
                resampled[resized >= 0.5] = cls
        else:
            resampled = resampled_2d
    else:
        for cls in range(1, n_class + 1):
            mask = (pred_vol == cls).astype(float)
            if not np.any(mask):
                continue
            resized = resize(mask, target_shape, order=0, mode='edge',
                             cval=0, clip=True, anti_aliasing=False)
            resampled[resized >= 0.5] = cls

    return resampled