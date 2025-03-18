import cv2
import matplotlib.pyplot as plt

image = cv2.imread("Image for task 1.jpg")


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis("off")
plt.show()

#