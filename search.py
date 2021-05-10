
from detector import FeatureDetector
from searcher import Searcher 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cd = FeatureDetector((8, 12, 3))

fig = plt.figure(figsize=(10, 10))
axes = []

#read and display image query
query = cv2.imread("dataset/test/137100.jpg")
axes.append(fig.add_subplot(3,2, 1))
axes[-1].set_title("query")
plt.imshow(cv2.cvtColor(query, cv2.COLOR_BGR2RGB))
plt.axis('off')

#tinh dac trung anh query
queryFeatures = cd.describe(query)

#tim anh
data = Searcher('index.csv')
results = data.search(queryFeatures)

#hien thi anh tim kiem duoc
i=1
for score, images in results:

        print(images)
        img = mpimg.imread('dataset/'+images)
        
        axes.append(fig.add_subplot(3,2, i+1))

        axes[-1].set_title(round(score,2))
        plt.imshow(img)
        plt.axis('off')
        i=i+1
plt.show()

        