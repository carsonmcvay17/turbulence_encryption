import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import imageio
from PIL import Image


from turbencrypt.img_analysis import img_analysis

image_dir="raw_images/image_3.jpg"
data_path = "movie_files/midre_shark"
base_img = imageio.imread(image_dir)
base_img = Image.fromarray(base_img)
base_img = base_img.convert("RGB")

desired_size = (256,256)
base_img = jnp.array(base_img.resize(desired_size))


model = img_analysis()

image_files = [f for f in os.listdir(data_path) if f.startswith('my_fig')]

mse_list = []

for image_file in image_files:
    image_path = os.path.join(data_path, image_file)
    image = imageio.imread(image_path)
    image = Image.fromarray(image)
    image = image.convert("RGB")
    image = jnp.array(image.resize(desired_size))
    mse = model.mse(image, base_img)
    mse_list.append(mse)

# plot
plt.plot(mse_list)
plt.xlabel('Image Index')
plt.ylabel('MSE')
plt.title('MSE between Images and Reference Image')
plt.show()