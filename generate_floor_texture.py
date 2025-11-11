
import numpy as np
from PIL import Image
import os

def generate_texture(width=1024, height=1024):
    """Generates a chevron/zigzag pattern based on the user-provided image."""
    # Colors
    black = (20, 20, 20)
    white = (230, 230, 230)
    green = (58, 243, 76)

    # Pattern parameters
    # Width of one full 'V' shape
    period = 128
    # Thickness of the black and white stripes
    stripe_thickness = 64
    # Steepness of the chevron
    slope = 2.0

    # Create coordinate grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # --- Chevron Pattern Logic ---
    # 1. Create a triangular wave based on the x-coordinate.
    #    `x % period` gives a sawtooth wave, `abs(...)` folds it into a triangle.
    triangular_wave = np.abs((x % period) - period / 2)
    
    # 2. Combine the y-coordinate with the triangular wave.
    #    This shifts the y-value up and down, creating diagonal bands that change direction.
    val = y + slope * triangular_wave

    # 3. Determine the color based on which stripe the value falls into.
    pattern = (val // stripe_thickness) % 2

    # 4. Create the image from the pattern.
    image_data = np.where(pattern[..., np.newaxis] == 0, black, white).astype(np.uint8)

    # --- Add Gridlines ---
    grid_size = 256
    # Make gridlines semi-transparent by blending with the image data
    grid_mask = (y % grid_size < 2) | (x % grid_size < 2)
    image_data[grid_mask] = (image_data[grid_mask] * 0.5 + np.array(green) * 0.5).astype(np.uint8)

    return Image.fromarray(image_data, 'RGB')

if __name__ == "__main__":
    # Save the image in the same directory as the MJCF models
    # This makes file paths in the XML simple
    script_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(script_dir, 'src/reachy_mini/descriptions/reachy_mini/mjcf/')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    file_path = os.path.join(save_dir, 'floor_texture.png')
    
    print("Generating floor texture...")
    img = generate_texture()
    img.save(file_path)
    print(f"Successfully saved texture to {file_path}")
