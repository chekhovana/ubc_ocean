import pyvips
import matplotlib.pyplot as plt
# 0	./31793
# 0	./50878
# 0	./61852
# 6.2G	./45630

filename = 'data/original/train_images/31793.png'
# filename = 'data/original/train_thumbnails/31793_thumbnail.png'
res = pyvips.Image.new_from_file(filename)
res = res.resize(1 / 200).numpy()
plt.imshow(res)
plt.show()