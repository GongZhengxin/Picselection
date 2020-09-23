**picselect.py**  servers as chaffer for images in train set of  every category within ImageNet2012. The first standard of selection is that resolution of each selected image must be larger than 375x375. The second operation is to sort the order accoridng to image activtion of Resnet152, and for every quartile of images shuffle to fetch the first 30 images.

**StimuliFiles**  stores the image paths of each category as .stim.csv file, whose resolution meet the first standard. Now there is 100 files.

**FilteredStimFiles**  stores the random selected images meet the second standard as .txt file. 120 image paths are stored in each file.

**OriginalImages**  stores the orginal images from ImageNet2012 which match the list in *FilteredStimFiles*. There are 20 folds correspond to the firt 20 categories.  

