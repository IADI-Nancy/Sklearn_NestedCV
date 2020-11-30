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

    # === Local Intensity Peak ===
    label_stats.Execute(image, mask)
    maximum_image = (image == label_stats.GetMaximum(1)) * mask
    max_idx = np.argwhere(sitk.GetArrayFromImage(maximum_image).transpose((2, 1, 0)) != 0)
    local_int_peak_list = []
    for idx in max_idx:
        buffer_image = sitk.Image(maximum_image)
        buffer_image[[int(_) for _ in idx]] = 2
        local_int_peak_mask = sitk.SignedMaurerDistanceMap(buffer_image == 2, squaredDistance=False,
                                                           useImageSpacing=True) <= dist
        label_stats.Execute(image, local_int_peak_mask)
        local_int_peak_list.append(label_stats.GetMean(1))
    local_intensity_peak = max(local_int_peak_list)

    # === Global Intensity Peak ===
    convolved_array = convolve(sitk.GetArrayFromImage(image).astype(np.float64), weights=sitk.GetArrayFromImage(kernel),
                               mode="constant", cval=np.nan)
    convolved_array /= np.sum(sitk.GetArrayFromImage(kernel))
    # Border values set to nan we need to calculate mean on this voxel only with voxels in the image and not full kernel
    nan_in_mask = np.logical_and(np.isnan(convolved_array), sitk.GetArrayFromImage(mask) == 1)
    nan_in_mask_idx = np.argwhere(nan_in_mask)
    for idx in nan_in_mask_idx:
        buffer_image = sitk.Image(mask)
        buffer_image[[int(_) for _ in idx[::-1]]] = 2
        global_int_peak_mask = sitk.SignedMaurerDistanceMap(buffer_image == 2, squaredDistance=False,
                                                            useImageSpacing=True) <= dist
        label_stats.Execute(image, global_int_peak_mask)
        convolved_array[tuple(idx)] = label_stats.GetMean(1)
    convolved_image = sitk.GetImageFromArray(convolved_array)
    convolved_image.CopyInformation(image)
    label_stats.Execute(convolved_image, mask)
    global_intensity_peak = label_stats.GetMaximum(1)

    return {'local_intensity_peak': local_intensity_peak, 'global_intensity_peak': global_intensity_peak}