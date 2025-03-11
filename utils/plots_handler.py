import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from PIL import Image
import numpy as np
from utils.config import OUTPUT_DIR

def plot_sparse_structure(sparse_matrix, title, cmap='Greys'):
    """
    Visualizes sparse matrices clearly by filling squares.
    Nonzero entries in black, zeros in white.
    Uses imshow for a crisp rendering, especially for large matrices.
    """
    # Convert to dense binary matrix (handle both sparse and dense)
    if issparse(sparse_matrix):
        binary_matrix = (sparse_matrix != 0).astype(int).toarray()
    else:
        binary_matrix = (np.array(sparse_matrix) != 0).astype(int)
    
    # Determine matrix dimensions
    n_rows, n_cols = binary_matrix.shape
    # Dynamically adjust figure size: use a scaling factor (0.1 inches per cell)
    # with limits to avoid extremes.
    fig_width = min(max(n_cols * 0.1, 10), 20)
    fig_height = min(max(n_rows * 0.1, 8), 16)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Render the matrix using imshow with no interpolation between pixels
    ax.imshow(binary_matrix, cmap=cmap, interpolation='nearest')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Columns", fontsize=14)
    ax.set_ylabel("Rows", fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def fig_to_pil_image(fig):
    """
    Convert a Matplotlib figure to a PIL image.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img.convert('RGB')

def resize_images_to_height(images, target_height):
    """
    Resize each PIL image in the list to the target height (keeping aspect ratio).
    """
    resized = []
    for img in images:
        w, h = img.size
        new_width = int(w * target_height / h)
        resized.append(img.resize((new_width, target_height), Image.Resampling.LANCZOS))
    return resized

def horizontally_concatenate(images, bg_color=(255,255,255)):
    """
    Concatenate a list of PIL images horizontally.
    All images are assumed to have the same height.
    """
    widths = [img.width for img in images]
    total_width = sum(widths)
    max_height = images[0].height
    composite = Image.new('RGB', (total_width, max_height), color=bg_color)
    x_offset = 0
    for img in images:
        composite.paste(img, (x_offset, 0))
        x_offset += img.width
    return composite

def vertically_concatenate(images, bg_color=(255,255,255)):
    """
    Concatenate a list of PIL images vertically.
    All images are assumed to have the same width.
    """
    max_width = images[0].width
    total_height = sum(img.height for img in images)
    composite = Image.new('RGB', (max_width, total_height), color=bg_color)
    y_offset = 0
    for img in images:
        composite.paste(img, (0, y_offset))
        y_offset += img.height
    return composite

def save_all_plots(permuted_matrices, canonical_matrices, file_path, filename="experiment_composite_plots.PNG"):
    """
    Save a composite page into a single PNG file.
    
    Top row: All permuted figures (the first element is the original figure) arranged horizontally.
    Bottom row: All canonical figures arranged horizontally.
    
    The two rows are then concatenated vertically.
    """
    # Convert permuted figures to PIL images (top row).
    top_figures = [plot_sparse_structure(matrix, f"Permuted Matrix {i}") 
                   for i, matrix in enumerate(permuted_matrices)]
    # Convert canonical figures to PIL images (bottom row).
    bottom_figures = [plot_sparse_structure(matrix, f"Canonical Matrix {i}") 
                      for i, matrix in enumerate(canonical_matrices)]
    
    # Convert figures to PIL images.
    top_imgs = [fig_to_pil_image(fig) for fig in top_figures]
    bottom_imgs = [fig_to_pil_image(fig) for fig in bottom_figures]
    
    # Resize top row images so that they all have the same height.
    min_height_top = min(img.height for img in top_imgs)
    top_row_resized = resize_images_to_height(top_imgs, min_height_top)
    top_row_composite = horizontally_concatenate(top_row_resized)
    
    # Resize bottom row images if any, so that they all have the same height.
    if bottom_imgs:
        min_height_bottom = min(img.height for img in bottom_imgs)
        bottom_row_resized = resize_images_to_height(bottom_imgs, min_height_bottom)
        bottom_row_composite = horizontally_concatenate(bottom_row_resized)
    else:
        bottom_row_composite = None

    # Pad the narrower row if widths differ.
    top_width = top_row_composite.width
    if bottom_row_composite is not None:
        bottom_width = bottom_row_composite.width
        if top_width > bottom_width:
            padded_bottom = Image.new('RGB', (top_width, bottom_row_composite.height), color=(255,255,255))
            padded_bottom.paste(bottom_row_composite, (0, 0))
            bottom_row_composite = padded_bottom
        elif bottom_width > top_width:
            padded_top = Image.new('RGB', (bottom_width, top_row_composite.height), color=(255,255,255))
            padded_top.paste(top_row_composite, (0, 0))
            top_row_composite = padded_top
    
    # Vertically stack the two rows.
    if bottom_row_composite is not None:
        final_composite = vertically_concatenate([top_row_composite, bottom_row_composite])
    else:
        final_composite = top_row_composite
    
    # Create a figure to display the composite image.
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(np.array(final_composite), cmap='Greys', interpolation='nearest')
    ax.axis('off')
    fig.tight_layout()

    # Generate filename components
    base_name = os.path.basename(file_path)
    base_name = os.path.splitext(base_name)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}.png")
    
    fig.savefig(filename, dpi=300)
    
    print(f"Composite plot has been saved to {filename}")
