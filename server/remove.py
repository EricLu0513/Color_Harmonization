from rembg import remove
from PIL import Image
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Remove background from an image.")
parser.add_argument("--input_image", type=str, help="Path to the input image file.")
parser.add_argument("--output_image", type=str, help="Path to save the output image file.")
parser.add_argument("--mode", type=int, default=0, help= "0:reserve foreground, 1:reserve background")
args = parser.parse_args()
output_path = args.output_image + ".png"

input_image = Image.open(args.input_image).convert('RGBA')
output_image = remove(input_image)

if args.mode == 1:
    orig_np = np.array(input_image)
    removed_np = np.array(output_image)

    # Step 3: Create mask for foreground（前景 pixel alpha > 0）
    foreground_mask = removed_np[:, :, 3] > 0  # 透明度非 0 即為前景

    # Step 4: 把原圖中前景部分設為透明（或白色）
    result_np = orig_np.copy()
    result_np[foreground_mask] = [0, 0, 0, 0]  # 透明，如果想白色背景就改成 [255, 255, 255, 255]

    # Step 5: Save the result
    result_image = Image.fromarray(result_np)
    result_image.save(output_path)
else:
    output_image.save(output_path)


