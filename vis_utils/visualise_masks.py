import torch
import numpy as np
import cv2
import torch.nn.functional as F
from skimage.morphology import skeletonize
import kornia.morphology
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR

def overlay_mask_mult(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.95):
    """
    Overlay a multi-class mask on an image.

    :param image: Tensor of shape (1, H, W, 3) representing the image.
    :param mask: Tensor of shape (1, 4, H1, W1) representing the multi-class mask.
    :param alpha: Transparency factor for overlay.
    :return: Overlayed image as a NumPy array.
    """

    # Get the height and width of the image
    h, w = image.shape[1], image.shape[2]

    # Step 1: Get the class with the highest score in the mask (argmax across the class dimension)
    combined_mask = torch.argmax(mask, dim=1).unsqueeze(0)  # Shape (H1, W1)

    # Step 2: Resize the mask to match the image size using torch interpolation
    combined_mask = F.interpolate(combined_mask.float(), size=(h, w), mode='nearest').squeeze(0).long()

    # Step 3: Normalize mask values and apply the colormap (viridis)
    # Instead of using cm.viridis (from Matplotlib), we will use PyTorch to apply the colormap.
    colormap = torch.tensor([
        [70, 70, 70],  # Class 0 - outer retina
        [255, 0, 255],  # Class 1 - left tool
        [255, 0, 0],  # Class 2 - right tool
        [0, 180, 180]  # Class 3 - retina
    ], dtype=torch.uint8, device='cuda')

    # Map each pixel in the combined mask to its corresponding color
    color_mask = colormap[combined_mask]  # Shape (H, W, 3)

    # Step 4: Overlay the mask onto the image
    # The mask is overlayed onto the image using alpha blending.
    overlayed_image = (alpha * color_mask.float() + (1 - alpha) * image.squeeze(0)).clamp(0, 255).byte()

    return overlayed_image


def overlay_mask_bin(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.95):
    """
    Overlay a binary mask on an image.

    :param image: Tensor of shape (1, H, W, 3) representing the image.
    :param mask: Tensor of shape (1, H1, W1) representing the binary mask.
    :param alpha: Transparency factor for overlay.
    :return: Overlayed image as a NumPy array.
    """

    # Get the height and width of the image
    h, w = image.shape[1], image.shape[2]

    # Step 2: Resize the mask to match the image size using torch interpolation
    combined_mask = F.interpolate(mask.float(), size=(h, w), mode='nearest').squeeze(0).long()

    # Step 3: Normalize mask values and apply the colormap (viridis)
    # Instead of using cm.viridis (from Matplotlib), we will use PyTorch to apply the colormap.
    colormap = torch.tensor([
        [0, 0, 0],  
        [204, 255, 0],  
    ], dtype=torch.uint8, device='cuda')


    # Map each pixel in the combined mask to its corresponding color
    color_mask = colormap[combined_mask]  # Shape (H, W, 3)

    # Step 4: Overlay the mask onto the image
    # The mask is overlayed onto the image using alpha blending.
    overlayed_image = (alpha * color_mask.float() + (1.) * image.squeeze(0)).clamp(0, 255).byte()

    return overlayed_image





def extract_tool_tip_and_focus_region(image: torch.Tensor, binary_mask: torch.Tensor, tool_type: str = "left"):
    """
    Estimate and overlay the tip of a binary tool mask.

    :param image: Tensor of shape (1, H, W, 3) representing the image
    :param binary_mask: Tensor of (1, H1, W1) representing the binary mask of the tool of interest
    :param tool_type: representing left/right tool
    """

    device = binary_mask.device
    binary_mask = (binary_mask * 255).byte()  # Scale to 0-255 for visualization
    h, w = binary_mask.size()[1:3]

    # Skeletonize using Kornia
    skeleton = skeletonize(binary_mask.float().unsqueeze(0))
    skeleton = (skeleton.squeeze(0).squeeze(0) * 255).byte()

    # Create skeleton overlay in red
    skeleton_colored = torch.zeros(h, w, 3, dtype=torch.uint8, device=device)  # RGB Tensor

    indices = (skeleton > 0).nonzero(as_tuple=True)  # Get pixel locations where skeleton > 0

    if tool_type == 'left':
        skel_color = [255, 0, 0]
    else:
        skel_color = [0, 0, 255]


    skeleton_colored[indices[0], indices[1], :] = torch.tensor(skel_color, device=device, dtype=skeleton_colored.dtype)

    # Resize skeleton overlay to match image size
    skeleton_colored_res = (F.interpolate(skeleton_colored.permute(2, 0, 1).unsqueeze(0).float(),size=(image.shape[1], image.shape[2]),
                                          mode='nearest').squeeze(0).byte().permute(1, 2, 0))  # (H, W, 3)


    # Overlay the skeleton on the original image
    overlayed_image = (0.9 * image.float() + 0.9 * skeleton_colored_res.float()).clamp(0, 255).byte()



    # Get the leftmost point
    if tool_type == 'left':
        # Find rightmost point of the skeleton
        y_indices, x_indices = torch.where(skeleton_colored_res[:, :, 0] > 100)
        min_x_idx = torch.argmax(x_indices)  # Find index of maximum x value
    else:
        # Find leftmost point of the skeleton
        y_indices, x_indices = torch.where(skeleton_colored_res[:, :, 2] > 100)
        min_x_idx = torch.argmin(x_indices)  # Find index of minimum x value


    tip_x, tip_y = x_indices[min_x_idx].item(), y_indices[min_x_idx].item()  # Get coordinates

    # Draw the tip (green dot) on the overlayed image
    overlayed_image[:,tip_y-5:tip_y+5, tip_x-5:tip_x+5] = torch.tensor([0, 255, 0], device=device)  # Green dot


    return overlayed_image

 



## external code

def skeletonize(data: Tensor, kernel: Tensor | None = None) -> Tensor:
    r"""Return the skeleton of a binary image.

    In morphological terms, it involves reducing foreground regions in a
    binary image while preserving the extent and connectivity of the original
    region and throwing away most of the foreground pixels.

    The kernel must have 2 dimensions.

    Args:
        data: Image with shape :math:`(B, 1, H, W)`. [C = 1]
        kernel: Positions of non-infinite elements of a flat structuring element.
                Non-zero values give the set of neighbors of the center over which
                the operation is applied. Its shape is :math:`(k_x, k_y)`.
                For full structural elements use torch.ones_like(structural_element).
    Returns:
        Skeleton of the image with shape :math:`(B, 1, H, W)`. [C = 1]

    Example:
        >>> data = torch.rand(1, 1, 20, 20)
        >>> kernel = torch.ones(3, 3)
        >>> skeleton_img = skeletonize(data, kernel)
    """

    KORNIA_CHECK_IS_TENSOR(data)
    KORNIA_CHECK(len(data.shape) == 4, "Input size must have 4 dimensions.")

    data = data.int()
    output = torch.zeros(data.shape, device=data.device, dtype=torch.int64)

    if kernel is None:
        kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], device=data.device, dtype=torch.int64)

    KORNIA_CHECK_IS_TENSOR(kernel)
    KORNIA_CHECK(len(kernel.shape) == 2, "Kernel size must have 2 dimensions.")

    while True:
        opened_img = kornia.morphology.opening(data, kernel)
        subtracted_img = torch.sub(data, opened_img)
        subtracted_img[subtracted_img < 0] = 0
        eroded_img = kornia.morphology.erosion(data, kernel)
        output = torch.bitwise_or(output, subtracted_img)
        data = eroded_img.clone()
        if torch.count_nonzero(data) == 0:
            break

    return output.view_as(data)
