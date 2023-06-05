"""
Verify dataset uploaded to the hub is good
"""

from datasets import load_dataset

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

dataset = load_dataset('zachary-shah/musdb18-spec-pix2pix10-samples')

i = 0

plt.subplot(1,2,1)
plt.imshow(dataset["train"]['original_image'][i])
plt.title("original image")
plt.subplot(1,2,2)
plt.imshow(dataset["train"]['edited_image'][i])
plt.title("edited image")
plt.suptitle(dataset["train"]['edited_prompt'][i])
plt.show()