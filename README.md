# Auto-liquify_kindaof

This is a naive try at making an autoliquify for images that have a silluette after Img2Img in stable diffusion, it uses a simple linear grid for moving pixels around and a machine learning for optimization.

Needs pytorch, Pillow and tqdm and a GPU with at least 8GB of vram

You place the alpha channel of images in folder in_alpha0 and in_alpha1, the originals from blender and the ones after img2img, you can put the original images too, but it doesn't look nice

arguments:

resolution: -1, for the same resolution as input.

steps: number of backprop steps.

contour_lambda: how much weight is given to the contours vs the original images, with 1 it doens't search for original images.

grid_smoothing: how much smoothing in the grid to apply.

lr: learning rate.
