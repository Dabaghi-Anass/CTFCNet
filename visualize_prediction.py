import cv2
import numpy as np
import os

# =========================
# CONFIG
# =========================

COLOR_MAP = {
    (255, 255, 0):  "High-density residential buildings",  # yellow
    (0, 0, 255):    "Apartment-type buildings",            # blue
    (0, 255, 0):    "Factory buildings",                   # green
    (0, 255, 255):  "Detached buildings",                  # cyan
    (255, 0, 0):    "Complex buildings",                   # red
}

ALPHA = 0.5  # transparency

# =========================
# FUNCTION
# =========================

def overlay_mask(image_path, mask_path, output_path):

    # Read images
    image = cv2.imread(image_path)
    mask  = cv2.imread(mask_path)

    if image is None or mask is None:
        raise ValueError("Image or mask not found.")

    # Convert BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask  = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    overlay = image.copy()

    # For each defined color
    for color, label in COLOR_MAP.items():
        color_array = np.array(color)

        # Create binary mask for this color
        binary_mask = np.all(mask == color_array, axis=-1)

        # Apply colored overlay
        overlay[binary_mask] = (
            overlay[binary_mask] * (1 - ALPHA)
            + color_array * ALPHA
        )

        # Optional: write label name once if class exists
        if np.any(binary_mask):
            y, x = np.where(binary_mask)
            cv2.putText(
                overlay,
                label,
                (x[0], y[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Convert back to BGR for saving
    overlay = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, overlay)
    print(f"Saved to {output_path}")


image_path = "pretrained/UBTdataset/test/img1/1_754.png"
mask_path  = "predictions/1_754.png"

overlay_mask(image_path, mask_path, "overlay_result.png")
