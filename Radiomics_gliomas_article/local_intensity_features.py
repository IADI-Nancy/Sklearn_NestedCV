import numpy as np
import SimpleITK as sitk
from scipy.ndimage.filters import convolve


def local_intensity_features(image, mask):
    label_stats = sitk.LabelStatisticsImageFilter()

    # Convolution kernel creation
    dist = (3/(4*np.pi))**(1/3)*10
    radius_voxel = max(np.floor(dist / np.array(image.GetSpacing())))
    kernel = sitk.Image([int(radius_voxel) * 2 + 1] * 3, sitk.sitkUInt32)
    kernel.SetSpacing(image.GetSpacing())
    kernel_center = kernel.TransformPhysicalPointToIndex(np.array(kernel.GetSpacing()) * (np.array(kernel.GetSize()) - 1) / 2)
    kernel[kernel_center] = 1
    kernel = sitk.SignedMaurerDistanceMap(kernel, squaredDistance=False, useImageSpacing=True) <= dist

    # Convolution with image
    convolved_array = convolve(sitk.GetArrayFromImage(image).astype(np.float64), weights=sitk.GetArrayFromImage(kernel),
                               mode='constant', cval=0)
    normalization_array = convolve(np.ones_like(sitk.GetArrayFromImage(image), dtype=float),
                                   weights=sitk.GetArrayFromImage(kernel), mode='constant', cval=0)
    convolved_array /= normalization_array
    convolved_image = sitk.GetImageFromArray(convolved_array)
    convolved_image.CopyInformation(image)

    # === Local Intensity Peak ===
    # Select voxel with maximum intenisty in image
    label_stats.Execute(image, mask)
    maximum_mask = (image == label_stats.GetMaximum(1)) * mask
    # Among these voxels select the maximum value in convolved_image
    label_stats.Execute(convolved_image, maximum_mask)
    local_intensity_peak = label_stats.GetMaximum(1)

    # === Global Intensity Peak ===
    label_stats.Execute(convolved_image, mask)
    global_intensity_peak = label_stats.GetMaximum(1)

    return {'local_intensity_peak': local_intensity_peak, 'global_intensity_peak': global_intensity_peak}