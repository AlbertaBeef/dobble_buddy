
# example of combination image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# import matplotlib

import os, shutil

deck = 'dobble_deck07_cards_55'
dataPath = 'dobble_dataset/' + deck + '/'


augmentedPath = 'dobble_dataset/' + deck  + '-augmented/'

if os.path.exists(augmentedPath):
	shutil.rmtree(augmentedPath) 
os.mkdir(augmentedPath)


total_images_to_augment = 100
image_size = 300

for folder in os.listdir(dataPath):
	print("[INFO] generating images in folder " + folder)

	for file in os.listdir(dataPath + '/' + folder):
		# load each image
		img = load_img(dataPath + '/' + folder + '/' + file)
		# convert to numpy array
		data = img_to_array(img)
		# expand dimension to one sample
		samples = expand_dims(data, 0)
		# create image data augmentation generator
		datagen = ImageDataGenerator(
			width_shift_range=0.3,
			height_shift_range=0.3, 
			brightness_range=[0.3,1.0], 
			zoom_range=[0.7,1.5]
		)

		# prepare iterator
		it = datagen.flow(samples, batch_size=1)

		outputPath = augmentedPath + folder + '/'
		
		if os.path.exists(outputPath):
			shutil.rmtree(outputPath) 
		os.mkdir(outputPath)

		for i in range(1, total_images_to_augment + 1):
			# generate batch of images
			batch = it.next()
			# convert to unsigned integers for viewing
			image = batch[0].astype('uint8')

			fig = pyplot.figure(frameon=False)
			#fig.set_size_inches(w,h)
			ax = pyplot.Axes(fig, [0., 0., 1., 1.])
			ax.set_axis_off()
			fig.add_axes(ax)

			# plot raw pixel data
			ax.imshow(image)
			fig.savefig(fname = outputPath + "card" + folder + "_{:03d}.tif".format(i))
			# show the figure
			#pyplot.show()

			# the figure will remain open, using memory, open unless explicitly closed with the following: (you'll get a warning if you don't include it)
			pyplot.close('all')
			