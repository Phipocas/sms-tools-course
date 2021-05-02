import soundDownload as SD
import soundAnalysis as SA
import numpy as np

key = "chnqxOZ35DAhmdHPAQaOs0bOXFbUNVc5Os0MimA6"

# # Part 1 - Download Sounds/ Mine the data

# SD.downloadSoundsFreesound(queryText="bassoon", tag="single-note", duration=(0, 5), API_Key=key, outputDir="Data", topNResults=20, featureExt='.json')

# SD.downloadSoundsFreesound(queryText="guitar", tag="single-note", duration=(0, 5), API_Key=key, outputDir="Data", topNResults=20, featureExt='.json')

# SD.downloadSoundsFreesound(queryText="naobo", tag=None, duration=(0, 1.5), API_Key=key, outputDir="Data", topNResults=20, featureExt='.json')


# Part 2 - Plot based on features

# SA.descriptorPairScatterPlot("Data", descInput=(3, 1), anotOn=0)

# Part 3 - Cluster with K-means

num_clusters = 10
accuracy = np.array([])
desc_list = [i for i in range(10)]
for i in range(num_clusters):  
    accuracy = np.append(accuracy, SA.clusterSounds("Data", nCluster = num_clusters, descInput=[0,1,2,3,4,5,6,7,8,9]))
acc_mean = np.mean(accuracy)

print("\n")
print("\n")
print(f"Your mean accuracy was {acc_mean}")



#Part 4 - Classify a new sound

# SD.downloadSoundsFreesound(queryText="naobo", tag=None, duration=(0, 1.5), API_Key=key, outputDir="Data", topNResults=25, featureExt='.json')
# SA.classifySoundkNN("Data/399492_7575123-lq.json", "Data/inst", 11, descInput = [3, 1])
# SA.classifySoundkNN("Data/372652_2475994-lq.json", "Data/inst", 11, descInput = [3, 1])
# SA.classifySoundkNN("Data/222293_2385996-lq.json", "Data/inst", 11, descInput = [3, 1])

