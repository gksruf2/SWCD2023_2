import os
import shutil
from PIL import Image

IMG_EXTENSIONS = [
'.jpg', '.JPG', '.jpeg', '.JPEG',
'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
	images = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir
	new_root = './dataset/fashion/In-shop Clothes Retrieval Benchmark/Img/'
	if not os.path.exists(new_root):
		os.mkdir(new_root)

	train_root = './dataset/fashion/train/'
	if not os.path.exists(train_root):
		os.mkdir(train_root)

	test_root = './dataset/fashion/test/'
	if not os.path.exists(test_root):
		os.mkdir(test_root)

	train_images = []
	train_f = open('./dataset/fashion/train.lst', 'r')
	for lines in train_f:
		lines = lines.strip()
		if lines.endswith('.jpg'):
			train_images.append(lines)

	test_images = []
	test_f = open('./dataset/fashion/test.lst', 'r')
	for lines in test_f:
		lines = lines.strip()
		if lines.endswith('.jpg'):
			test_images.append(lines)

	#print(train_images, test_images)
	

	for root, _, fnames in sorted(os.walk(dir)):
		for fname in fnames:
			if is_image_file(fname):
				path = os.path.join(root, fname)
				path_names = path.split('/') 
				
				if path_names[5] != 'img':
					# img_highres -> no use
					break
				
				#['.', 'dataset', 'fashion', 'In-shop Clothes Retrieval Benchmark', 'Img', 'img', 'MEN', 'Denim', 'id_00000080', '01_1_front.jpg']
				# path_names[2] = path_names[2].replace('_', '')
				path_names[8] = path_names[8].replace('_', '')	# id
				path_names[9] = path_names[9].split('_')[0] + "_" + "".join(path_names[9].split('_')[1:])
				path_names = "".join([path_names[2],path_names[6],path_names[7],path_names[8],path_names[9]])

				new_path = os.path.join(root, path_names)
				img = Image.open(path)
				imgcrop = img.crop((40, 0, 216, 256))
				if path_names in train_images:
					imgcrop.save(os.path.join(train_root, path_names))
					print(path_names, "train data saved")
				elif path_names in test_images:
					imgcrop.save(os.path.join(test_root, path_names))
					print(path_names, "test data saved")

make_dataset('./dataset/fashion/In-shop Clothes Retrieval Benchmark/Img/')