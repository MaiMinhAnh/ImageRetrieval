from detector import FeatureDetector
from searcher import Searcher 
import argparse
import cv2
import matplotlib.pyplot as plt

cd = FeatureDetector((8, 12, 3))
query = cv2.imread("dataset/test/1.jpeg")

queryFeatures = cd.describe(query)

data = Searcher('index.csv')
results = data.search(queryFeatures)
cv2.imshow("query", query)

plt.figure(figsize = (10, 10))
for (score, resultID) in results:
	
	result = cv2.imread("dataset/"+resultID)
	print(resultID)



	temp = "score: " +f'{score}'
	cv2.imshow(temp, result)
	cv2.waitKey()