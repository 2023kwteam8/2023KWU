from rembg import remove
from PIL import Image

input_path = "image/2_resize/crop.jpg"  # input image path
output_path = "image/3_rmbg/output.png"  # output image path

input = Image.open(input_path)  # load image
output = remove(input)  # remove background
output.save(output_path)  # save image
