import cv2
import numpy as np
import sklearn

# Load the video
cap = cv2.VideoCapture("test_3.mp4")

# Get the video frames
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# Extract features from the frames (e.g. color histograms)
features = []
for frame in frames:
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    features.append(hist.flatten())

# Perform k-NN clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
kmeans.fit(features)

# Get the cluster labels for each frame
labels = kmeans.predict(features)

# Group the frames by their cluster labels
clusters = {}
for i, label in enumerate(labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(frames[i])

print("")
for cluster_index, cluster_frames in clusters.items():
    print("Cluster index:", cluster_index)
    print("Frames in cluster:", len(cluster_frames))
    print("Frames:", cluster_frames)