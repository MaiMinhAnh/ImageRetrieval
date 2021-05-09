from detector import FeatureDetector
import glob
import cv2

cd = FeatureDetector((8, 12, 3))
output = open("index.csv", "w")

for imgPath in glob.glob("dataset/coba/*"):
	imgID = imgPath[imgPath.rfind("/")+1:]
	image = cv2.imread(imgPath)

	feature = cd.describe(image)

	feature = [str(f) for f in feature]
	output.write("%s,%s\n" % (imgID, ",".join(feature)))

output.close()