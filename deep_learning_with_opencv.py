import numpy as np
import time 
import argparse
import cv2

#To run the program type: -
#python deep_learning_with_opencv.py --image Photos\Greenland_dog.jpg --prototxt bvlc_googlenet.prototxt --model bvlc_googlenet.caffemodel --labels synset_words.txt

#Setting argument for input of model, prototxt, image and labels.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True,
	help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())



#read image using opencv (path to image)
image = cv2.imread(args["image"])

#structuring labels by replace ' ' with ','  
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

#print(classes)

#feeding image using the dnn module
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

print("[INFO] loading model...")
#Setting the prototxt (GoogleNet) and caffe model 
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

net.setInput(blob)
#Setting starting time
start = time.time()

#processing the image / classifying the image
preds = net.forward()

#Setting ending time of the process
end = time.time()

print("[INFO] classification took {:.5} seconds".format(end - start))

idxs = np.argsort(preds[0])[::-1][:5]

# loop over the top-5 predictions and display them
for (i, idx) in enumerate(idxs):
	# draw the top prediction on the input image
	if i == 0:
		text = "Label: {}, {:.2f}%".format(classes[idx],
			preds[0][idx] * 100)
		cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
	# display the predicted label + associated probability to the
	# console	
	print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
		classes[idx], preds[0][idx]))
 
# display the output image
cv2.imshow("Image", image)

cv2.waitKey(0)