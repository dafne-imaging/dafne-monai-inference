import numpy as np
import torch 

from skimage.transform import resize
from monai.transforms.utils import generate_spatial_bounding_box

from monai.transforms import (MapTransform, 
                            CropForegroundd, 
                            NormalizeIntensity, 
                            SpatialCrop)


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



class PreprocessAnisotropy(MapTransform):
    def __init__(self, 
                 keys, 
                 target_spacing,
                 clip_values=None, 
                 normalize_values=None, 
                 model_mode="train",
                 spatial_dims:int=3):
        super().__init__(keys)
        
        self.target_spacing = target_spacing
        self.spatial_dims = spatial_dims
        self.keys = keys

        self.low = clip_values[0] if clip_values else 0
        self.high = clip_values[1] if clip_values else 0
        self.mean = normalize_values[0] if normalize_values else 0
        self.std = normalize_values[1] if normalize_values else 1

        self.training = (model_mode == 'train')

        self.crop_foreg = CropForegroundd(keys=['image'], source_key='image', allow_smaller=True)
        self.normalize_intensity = NormalizeIntensity(nonzero=True, channel_wise=True)

    def calculate_new_shape(self, spacing, shape):
        aspect_ratio = np.array(spacing) / self.target_spacing
        new_shape = np.round(aspect_ratio * np.array(shape)).astype(int)
        return new_shape
    
    def check_anisotrophy(self, spacing):
        def check(s):
            if s is None:
                return False
            return np.max(s) / np.min(s) >= 3
        return check(spacing) or check(self.target_spacing)
    
    def __call__(self, data):
        d = dict(data)
        image = d['image']

        current_spacing = d['image_meta_dict']['pixdim']
        current_spacing = current_spacing[1:] # no channel spacing
        image_spacing = np.array(current_spacing).tolist()

        if self.training:
            cropped_data = self.crop_foreg({"image": image})
            if 0 in cropped_data["image"].shape[1:]:
                pass
            else:
                image = cropped_data["image"]
        else:
            d["original_shape"] = np.array(image.shape[1:])
            box_start, box_end = generate_spatial_bounding_box(image, allow_smaller=True) #return pixels coordinates where the foreground is located
            temp_image = SpatialCrop(roi_start=box_start, roi_end=box_end)(image) #return cropped image, 0 if the box_start=box_end
            if 0 in temp_image.shape[1:]:
                d["bbox"] = np.vstack([box_start, box_end])
                d["crop_shape"] = np.array(image.shape[1:])
            else: 
                image = temp_image
                d["bbox"] = np.vstack([box_start, box_end])
                d["crop_shape"] = np.array(image.shape[1:])

        original_shape = image.shape[1:]

        resample_flag = False
        anisotrophy_flag = False

        if isinstance(image, torch.Tensor): image = image.numpy()

        if 0 not in original_shape and not np.allclose(self.target_spacing, image_spacing, atol=1e-4):
            resample_flag = True
            anisotrophy_flag = self.check_anisotrophy(image_spacing)
            
            resample_shape = self.calculate_new_shape(image_spacing, original_shape)
            
            if self.spatial_dims == 3:
                image = resample_image(image, resample_shape, anisotrophy_flag)
            elif self.spatial_dims == 2: 
                resized_channels = []
                for img_c in image:
                    res = resize(
                        img_c,
                        resample_shape,
                        order=3,
                        mode="edge",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                    resized_channels.append(res)
                image = np.stack(resized_channels, axis=0)

        d["resample_flag"] = resample_flag
        d["anisotrophy_flag"] = anisotrophy_flag

        if self.low != 0 or self.high != 0:
            image = np.clip(image, self.low, self.high)
            image = (image - self.mean) / self.std
        else:
            image = self.normalize_intensity(image.copy())

        d["image"] = image

        return d