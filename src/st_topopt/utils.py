from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps


class PillowGIFWriter:
    """Class for writing a set of images to a GIF file using Pillow."""

    def __init__(self,  target_size: int = 800) -> None:
        self.frames = []
        self.fps = 1
        self.target_size = target_size

    @staticmethod
    def density_to_grayscale_img(rho: np.ndarray) -> np.ndarray:
        """Convert a float density array to a grayscale uint8 array."""
        rho_img = ((1 - rho) * 255).astype(np.uint8)
        return rho_img

    def resize_(self, img: Image) -> Image:
        """Use PIL to resize an image to a maximum dimension. Interpolation is nearest
        neighbor."""
        H, W = img.size
        max_dim = max(H, W)
        scale_factor = self.target_size / max_dim
        return img.resize((int(H * scale_factor), int(W * scale_factor)), Image.NEAREST)

    def add_frame_number(self, img: Image) -> None:
        """Draw the frame number to the top left corner of the image."""
        border_width = 20
        img = ImageOps.expand(img, border=border_width, fill=255)
        draw = ImageDraw.Draw(img)
        draw.text((border_width, 0), f"It. {len(self.frames)}")
        return img

    def add_frame(self, rho: np.ndarray) -> None:
        """Add a frame to the GIF. 'rho' is a 2D array of floats in [0, 1]."""
        assert rho.ndim == 2
        rho_img = self.density_to_grayscale_img(rho)
        pil_img = Image.fromarray(rho_img, "L").convert("P")
        pil_img = self.resize_(pil_img)
        pil_img = self.add_frame_number(pil_img)
        self.frames.append(pil_img)

    def reset_(self):
        self.frames = []

    def save_bytes_(self, buffer: BytesIO):
        """Save the GIF to a BytesIO object."""
        self.frames[0].save(
            buffer,
            format="gif",
            save_all=True,
            append_images=self.frames[1:],
            optimize=False,
            duration=self.fps * 1000,  # ms
            loop=0,
        )


def matshow_to_image_buffer(mat: np.ndarray, display_cmap: bool = False, **kwargs):
    """Given a 2D array, return a BytesIO object containing a PNG image of the array.
    'display_cmap' determines whether to display the colorbar."""

    def get_figsize_from_array(arr: np.ndarray, max_figsize: int = 16) -> tuple:
        """Given a 'max_figsize' and a 2D array, return the appropriate figsize
        maintaing the aspect ratio."""
        height, width = arr.shape
        scale_factor = max(height, width) // max_figsize
        figsize = (width // scale_factor, height // scale_factor)
        return figsize

    figsize = get_figsize_from_array(mat)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.matshow(mat, **kwargs)
    if display_cmap:
        plt.colorbar(im, ax=ax)
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    return buf
