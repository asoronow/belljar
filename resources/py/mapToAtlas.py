import os
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from imutils import build_montages
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# Links in case we should need to redownload these, will not be included
nisslDownloadLink = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/ara_nissil_10.nrrd"
annotationDownloadLink = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_10.nrrd"

# TODO: Remove this test case, build handlers for this
pathParent = Path(__file__)
pngPath = os.path.join(pathParent.parents[1], "nrrd/png/half")
fileList = os.listdir(os.path.join(pathParent.parents[1], "nrrd/png/half")) # path to flat pngs
absolutePaths = [os.path.abspath(os.path.join(pngPath, name)) for name in fileList]

class AutoEncoder:
    @staticmethod
    def build(width, height, depth, filters=(32,64), latentDim=16):
        '''
        Intializes the model. Model architecture constructed from example by Adrian Rosebrock.
        '''

        inputShape = (height, width, depth) # the input image shape
        chanDim = -1 # where the channel dimension is, -1=last, 0=first

        inputs = tf.keras.layers.Input(shape=inputShape) # the 
        x = inputs
        
        for f in filters:
            x = tf.keras.layers.Conv2D(f, (3,3), strides=2, padding="same")(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        volumeSize = tf.keras.backend.int_shape(x)
        x = tf.keras.layers.Flatten()(x)
        latent = tf.keras.layers.Dense(latentDim, name="encoded")(x)

        x = tf.keras.layers.Dense(np.prod(volumeSize[1:]))(latent)
        x = tf.keras.layers.Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        for f in filters[::-1]:
            x = tf.keras.layers.Conv2DTranspose(f, (3,3) , strides=2, padding="same")(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)

        x = tf.keras.layers.Conv2DTranspose(depth, (3,3), padding="same")(x)
        outputs = tf.keras.layers.Activation("sigmoid", name="decoded")(x)

        autoencoder = tf.keras.models.Model(inputs, outputs, name='autoencoder')

        return autoencoder

def visualize_predictions(decoded, gt, samples=10):
	# initialize our list of output images
	outputs = None
	# loop over our number of output samples
	for i in range(0, samples):
		# grab the original image and reconstructed image
		original = (gt[i] * 255).astype("uint8")
		recon = (decoded[i] * 255).astype("uint8")
		# stack the original and reconstructed image side-by-side
		output = np.hstack([original, recon])
		# if the outputs array is empty, initialize it as the current
		# side-by-side image display
		if outputs is None:
			outputs = output
		# otherwise, vertically stack the outputs
		else:
			outputs = np.vstack([outputs, output])
	# return the output images
	return outputs

def loadImages(paths):
    '''Loads images given file paths'''
    images = []
    for p in paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256,256))
        images.append(np.array(img))
    images = np.array(images)
    images = images.reshape(images.shape[0], 256, 256, 1)
    images = images.astype('float32')
    images /= 255
    return images

trainX, testX = train_test_split(absolutePaths, test_size=0.2)
trainX, testX = loadImages(trainX), loadImages(testX)

def train(trainX, testX):
    '''Trains the model'''
    EPOCHS = 100
    INIT_LR = 1e-3
    BS = 64

    # construct our convolutional autoencoder
    print("[INFO] building autoencoder...")
    autoencoder = AutoEncoder.build(256, 256, 1)
    opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    autoencoder.compile(loss="mse", optimizer=opt)
    # train the convolutional autoencoder
    H = autoencoder.fit(
        trainX, trainX,
        validation_data=(testX, testX),
        epochs=EPOCHS,
        batch_size=BS)

    print("[INFO] making predictions...")
    decoded = autoencoder.predict(testX)
    vis = visualize_predictions(decoded, testX)
    cv2.imwrite("test.jpg", vis)

    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.jpg")
    print("[INFO] saving autoencoder...")
    autoencoder.save("hemisphere_model.h5", save_format="h5")

def euclidean(a, b):
	# compute and return the euclidean distance between two vectors
	return np.linalg.norm(a - b)

def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    
    similiarity = np.dot(a, b.T)/(a_norm * b_norm) 
    dist = 1. - similiarity
    
    return dist

indexes = list(range(0, trainX.shape[0]))


print("[INFO] loading autoencoder model...")
autoencoder = tf.keras.models.load_model("hemisphere_model.h5")
# create the encoder model which consists of *just* the encoder
# portion of the autoencoder
encoder = tf.keras.models.Model(inputs=autoencoder.input,
	outputs=autoencoder.get_layer("encoded").output)
# quantify the contents of our input images using the encoder
print("[INFO] encoding images...")
features = encoder.predict(trainX)

data = {"indexes": indexes, "features": features}
def perform_search(queryFeatures, index, maxResults=10):
    # initialize our list of results
    results = []
    # loop over our index
    for i in range(0, len(index["features"])):
        # compute the euclidean distance between our query features
        # and the features for the current image in our index, then
        # update our results list with a 2-tuple consisting of the
        # computed distance and the index of the image
        d = cosine_distance(queryFeatures, index["features"][i])
        results.append((d, i))
    # sort the results and grab the top ones
    results = sorted(results)[:maxResults]
    # return the list of results
    return results


features_test = encoder.predict(testX)

'''
# randomly sample a set of testing query image indexes
queryIdxs = list(range(0, testX.shape[0]))
queryIdxs = np.random.choice(queryIdxs, size=10,
	replace=False)
# loop over the testing indexes
for i in queryIdxs:
	# take the features for the current image, find all similar
	# images in our dataset, and then initialize our list of result
	# images
	queryFeatures = features_test[i]
	results = perform_search(queryFeatures, data, maxResults=10)
	images = []
	# loop over the results
	for (d, j) in results:
		# grab the result image, convert it back to the range
		# [0, 255], and then update the images list
		image = (trainX[j] * 255).astype("uint8")
		image = np.dstack([image] * 3)
		images.append(image)
	# display the query image
	query = (testX[i] * 255).astype("uint8")
	cv2.imshow("Query", query)
	# build a montage from the results and display it
	montage = build_montages(images, (256, 256), (2, 5))[0]
	cv2.imshow("Results", montage)
	cv2.waitKey(0)
'''
dapi = loadImages(["M107_s002.png"])
dapi_features = encoder.predict(dapi)[0]
results = perform_search(dapi_features, data, maxResults=10)
images = []
for (d, j) in results:
    # grab the result image, convert it back to the range
    # [0, 255], and then update the images list
    image = (trainX[j] * 255).astype("uint8")
    image = np.dstack([image] * 3)
    images.append(image)
# display the query image
query = (dapi[0] * 255).astype("uint8")
cv2.imshow("Query", query)
# build a montage from the results and display it
montage = build_montages(images, (256, 256), (2, 5))[0]
cv2.imshow("Results", montage)
cv2.waitKey(0)