from classify import *
from coordinate_solver import *
from image_morpher import *
from obfuscator import *

make_net()
image_urls = get_urls()
def show_2(image_loc, window_size = 8):
	global dogs, prob, top5
	rawim, im = prep_image(image_urls[image_loc])
	im2 = im.reshape([3,224,224])
	o = ImageObfuscator(im2, window_size)
	o.create_images()

	prob =np.vstack([np.array(lasagne.layers.get_output(output_layer, o.images[150*k:150*k+150], deterministic=True).eval()) for k in range(o.images.shape[0]/150 + 1)])
	top5 = np.argsort(prob[0])[-1:-6:-1]
	dogs = prob[:,top5[0]].reshape([rawim.shape[0]/window_size,rawim.shape[0]/window_size])
	dogs -= dogs.min() - 0.0001
	dogs /= dogs.max()
	# dogs = 1.01 - dogs
	dogs = np.minimum(-np.log(dogs+0.0001)*dogs,100)
	# dogs -= np.median(dogs)
	dogs = np.maximum(dogs, 0.005)
	plt.gray()
	plt.figure(1)
	plt.subplot(121)
	plt.imshow(dogs)

	plt.subplot(122)
	plt.imshow(rawim)
	plt.show()

dogs, prob, top5 = None,None,None
image_loc = 999
rawim, im = prep_image(image_urls[image_loc])
iu = ImageUniformer(rawim, 8, dogs.copy())
iu.prep_image()

plt.figure(1)
plt.subplot(131)
plt.imshow(dogs)
plt.subplot(132)
plt.imshow(iu.image)
plt.subplot(133)
plt.imshow(iu.new_image)
plt.show()
plt.imshow()
