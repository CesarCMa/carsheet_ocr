"""
Table structure and cell detection using kernel-based morphological operations,
inspired by the DEXTER paper (P.R. Nandhinee et al., 2022).

This script:
1. Reads an input image containing a table.
2. Applies Otsu's thresholding and inversion to separate foreground (text/lines)
   from background.
3. Detects horizontal and vertical lines via parameterized kernels.
4. Detects row and column separators in borderless/partially-bordered tables
   via sliding-window convolution.
5. Merges all detected lines and separators into a single mask.
6. Finds contours on the merged mask and outputs bounding boxes for each cell.

"""

# %%
import cv2
import numpy as np
import argparse

# %%
def otsu_threshold(img_gray):
    """Apply Otsu's thresholding."""
    _, thresh = cv2.threshold(img_gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def get_structural_lines(thresh_inv, kw=0.15, kh=0.1):
    """
    Detect horizontal and vertical lines using parameterized morphological kernels.
    thresh_inv: binary inverted image (foreground=255, background=0)
    kw: kernel width ratio for horizontal lines
    kh: kernel height ratio for vertical lines
    """
    h, w = thresh_inv.shape
    # horizontal kernel: 1 × int(w * kw)
    horiz_len = max(1, int(w * kw))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    # vertical kernel: int(h * kh) × 1
    vert_len = max(1, int(h * kh))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))

    # Erode then dilate to extract lines
    horiz_lines = cv2.erode(thresh_inv, horizontal_kernel)
    horiz_lines = cv2.dilate(horiz_lines, horizontal_kernel)
    vert_lines = cv2.erode(thresh_inv, vertical_kernel)
    vert_lines = cv2.dilate(vert_lines, vertical_kernel)

    return horiz_lines, vert_lines

def get_separators(thresh, slw):
    """
    Detect row/column separators by sliding-window convolution.
    thresh: binary Otsu image (foreground=0, background=255)
    slw: sliding-window width (in px)
    """
    # Invert so text=255, background=0
    inv = cv2.bitwise_not(thresh)
    h, w = inv.shape

    # Column separators: kernel shape = (h, slw)
    K_col = np.ones((h, slw), dtype=np.uint8)
    conv_v = cv2.filter2D(inv, -1, K_col)
    # threshold to keep only strong vertical white patches
    _, col_sep = cv2.threshold(conv_v,
                               slw * 255 * 0.9,
                               255,
                               cv2.THRESH_BINARY)

    # Row separators: kernel shape = (slw, w)
    K_row = np.ones((slw, w), dtype=np.uint8)
    conv_h = cv2.filter2D(inv, -1, K_row)
    _, row_sep = cv2.threshold(conv_h,
                               slw * 255 * 0.9,
                               255,
                               cv2.THRESH_BINARY)

    return row_sep, col_sep

def detect_cells(image,
                 kw=0.15,
                 kh=0.1,
                 min_cell_width=15,
                 min_cell_height=15):
    """
    Full pipeline: detect table cell bounding boxes in the input image.
    Returns a list of (x, y, w, h).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = otsu_threshold(gray)              
    thresh_inv = cv2.bitwise_not(thresh)       

    horiz, vert = get_structural_lines(thresh_inv, kw, kh)

    h, w = thresh.shape
    if h > w:
        slw = 1
    elif h < 360:
        slw = 4
    else:
        slw = 2

    row_sep, col_sep = get_separators(thresh, slw)

    combined = cv2.bitwise_or(horiz, vert)
    combined = cv2.bitwise_or(combined, row_sep)
    combined = cv2.bitwise_or(combined, col_sep)

    contours, _ = cv2.findContours(combined,
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw >= min_cell_width and ch >= min_cell_height:
            boxes.append((x, y, cw, ch))

    return boxes

def draw_boxes(image, boxes):
    """Draw bounding boxes on a copy of the image."""
    out = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out

def visualize_lines(image, horiz_lines, vert_lines):
    """Visualize detected horizontal and vertical lines on the original image.
    Horizontal lines are shown in red, vertical lines in blue."""
    vis = image.copy()
    
    horiz_color = np.zeros_like(vis)
    vert_color = np.zeros_like(vis)
    
    horiz_color[horiz_lines > 0] = [0, 0, 255]  
    vert_color[vert_lines > 0] = [255, 0, 0]    

    alpha = 0.5 
    vis = cv2.addWeighted(vis, 1, horiz_color, alpha, 0)
    vis = cv2.addWeighted(vis, 1, vert_color, alpha, 0)
    
    return vis

# %%
# Configuration parameters
image_path = "explorations/image_10.jpg"  # Path to the input image
kw = 0.15  # Horizontal kernel width ratio
kh = 0.1   # Vertical kernel height ratio 
min_w = 15 # Minimum cell width in pixels
min_h = 15 # Minimum cell height in pixels
output_path = "explorations/cells_detected.png" # Path to save output visualization

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Cannot read image at {image_path!r}")

boxes = detect_cells(img,
                        kw=kw,
                        kh=kh,
                        min_cell_width=min_w,
                        min_cell_height=min_h)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = otsu_threshold(gray)
thresh_inv = cv2.bitwise_not(thresh)
horiz, vert = get_structural_lines(thresh_inv, kw, kh)
lines_vis = visualize_lines(img, horiz, vert)
cv2.imwrite("explorations/lines_detected.png", lines_vis)

vis = draw_boxes(img, boxes)
cv2.imwrite("explorations/cells_detected.png", vis)

# %%