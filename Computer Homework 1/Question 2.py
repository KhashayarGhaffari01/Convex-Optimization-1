# %%
import numpy as np
import PIL
from PIL import Image
import random


# Q2 Part 1


# This function take image file location and save it as grayscale image matrix.
def image_to_array(image_location):
    image = PIL.Image.open(image_location).convert('L')
    return np.asarray(image)


# This function calculate of low rank approximation of matrix 'a' to rank 'k'.
# u, s, vh are SVD components of matrix 'a'.
def low_rank(u, s, vh, k):
    return u[:, 0:k] @ np.diag(s)[0:k, 0:k] @ vh[0:k, :]


# Show new image from matrix 'a'.
def show_image(a, save_name):
    Image.fromarray(np.array(a, dtype='uint8'), 'L').show('Low Image')
    # Save the image:
    Image.fromarray(np.array(a, dtype='uint8'), 'L').save(save_name)


# This function take two image as matrix and return the PSNR of noisy image.
def psnr(original, compressed):
    if np.allclose(original, compressed):
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(np.mean((original - compressed) ** 2)))
    return psnr


image_name = 'q2_pic.jpg'
u, s, vh = np.linalg.svd(image_to_array(image_name))
show_image(low_rank(u, s, vh, 13), "rank 13.jpg")


# d = np.arange(1000)
# e = np.zeros(1000)
# for k in range(1000):
#     e[k] = psnr(image_to_array(image_name), low_rank(u, s, vh, k))
# plt.plot(d, e)
# plt.show()
# %%


# This function add gaussian noise to matrix 'a'.
def gaussian_noise(a):
    gauss_matrix = np.random.normal(0, 32, size=a.shape)
    return a + gauss_matrix


# This function add salt & pepper noise to matrix 'a'.
def salt_pepper_noise(a):
    b = np.zeros(a.shape)
    b = b + a
    row_max, col_max = a.shape
    pixel_numbers = row_max * col_max
    white_pixels = random.randint(0.03 * pixel_numbers, 0.06 * pixel_numbers)
    black_pixels = random.randint(0.03 * pixel_numbers, 0.06 * pixel_numbers)
    for i in range(white_pixels):
        x = random.randint(0, row_max - 1)
        y = random.randint(0, col_max - 1)
        b[x, y] = 255
    for i in range(black_pixels):
        x = random.randint(0, row_max - 1)
        y = random.randint(0, col_max - 1)
        b[x, y] = 0
    return b


salt_pepper_image = salt_pepper_noise((image_to_array(image_name)))
gaussian_image = gaussian_noise(image_to_array(image_name))
us, ss, vhs = np.linalg.svd(salt_pepper_image)
ug, sg, vhg = np.linalg.svd(gaussian_image)
show_image(salt_pepper_image, "q2_pic denoise salt 0 original noise.jpg")
show_image(gaussian_image, "q2_pic denoise gaussian 0 original noise.jpg")
show_image(low_rank(us, ss, vhs, 15), "q2_pic denoise salt 15.jpg")
show_image(low_rank(us, ss, vhs, 25), "q2_pic denoise salt 25.jpg")
show_image(low_rank(us, ss, vhs, 65), "q2_pic denoise salt 65.jpg")
show_image(low_rank(ug, sg, vhg, 20), "q2_pic denoise gaussian 20.jpg")
show_image(low_rank(ug, sg, vhg, 35), "q2_pic denoise gaussian 35.jpg")
show_image(low_rank(ug, sg, vhg, 60), "q2_pic denoise gaussian 60.jpg")


# salt_pepper_psnr = np.zeros(400)
# gaussian_psnr = np.zeros(400)
# for i in range(400):
#     salt_pepper_psnr[i] = psnr(image_to_array(image_name), low_rank(us, ss, vhs, i))
#     gaussian_psnr[i] = psnr(image_to_array(image_name), low_rank(ug, sg, vhg, i))
# plt.plot(salt_pepper_psnr, color='green')
# plt.plot(gaussian_psnr, color='red')
# plt.show()
