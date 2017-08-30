# Semi-Supervised Change Detection by Auxiliary Variational Autoencoder

## Installation

Set up a virtualenv for Python 3. Execution was tested on Python 3.4.2.

```
pip3 install virtualenv
virtualenv auxcd_virtualenv
source auxcd_virtualenv/bin/activate
```

Install required Python modules:

```pip3 install -r requirements.txt```

You must also install libgdal and its Python bindings. The pip install of the Python GDAL module fails on some systems, and so is not included in requirements.txt. You can try installing it with ```pip3 install gdal```, and if this fails, try the workaround mentioned in [Python GDAL package missing header file when installing via pip](https://gis.stackexchange.com/questions/28966/python-gdal-package-missing-header-file-when-installing-via-pip), or you can try installing from your distribution's package manager.

Theano may require that you [install libgpuarray and pygpu manually](http://deeplearning.net/software/libgpuarray/installation.html).

## Usage

This program requires the following inputs:
* GeoTIFF images of the same location from two different times. The projection of these images must contain only translation components, and not any rotation or skew.
* A CSV containing bitemporal class labels for the GeoTIFF images in the following format (header row formatting is arbitrary):
```
X,Y,label_t0,label_t
sample1_x, sample1_y, sample1_label_t0, sample1_label_t1
sample2_x, sample2_y, sample2_label_t0, sample2_label_t1
sample3_x, sample2_y, sample3_label_t0, sample3_label_t1
...
```
Where x and y are given as geographic coordinates (not pixel coordinates). The CSV file can be generated with QGIS as described in the next section. There should be at least 80 labels with a class balance between change and non-change (so, a minimum of 40 change and 40 non-change samples). If ```--no-test``` is enabled, all samples will be used for training and the 80 can be reduced to 64.

Provided the above inputs, the command:

```python3 run.py image0.tif image1.tif labels.csv```

Will train a model from scratch and then detect changes between the images located at ```image0.tif``` and ```image1.tif```, using the CSV file located at ```labels.csv```. Output will be stored in the results directory and includes the following files:
* ```change_vae_svm.png```: Changes detected by the VAE after thresholding with an SVM.
* ```change_vae_svm_filtered.png```: VAE+SVM changes after filtering.
* ```change_vae_svm_overlay_t(n).png```: VAE+SVM changes highlighted in red over the image at t=n.
* ```change_vae_svm_filtered_overlay_t(n).png```: Filtered VAE+SVM changes highlighted in red over the image at t=n.

Additionally, using the ```--baseline``` flag will train baseline SVMs on raw RGB data for comparison and produce the following files:
* ```change_rgb_svm.png```: Changes detected by the baseline SVM.
* ```change_rgb_svm_filtered.png```: Baseline changes after filtering.
* ```change_rgb_svm_overlay_t(n).png```: Baseline changes highlighted in red over the image at t=n.
* ```change_rgb_svm_filtered_overlay_t(n).png```: Filtered baseline changes highlighted in red over the image at t=n.

The following additional modifiers are available:
* ```--save name, -s name```: Saves the model as ```name``` instead of the default, ```last```.
* ```--load name, -l name```: Loads the pretrained model ```name``` instead of training from scratch. Even if ```--save``` was not provided, the most recently trained model can be loaded with ```--load last```.
* ```--overwrite, -o```: Required if ```--save``` is used with a name (other than ```last```) that would write over a previously saved model.

## How to generate the CSV file with QGIS

Create a new QGIS project. Add both images to the project as raster layers. They should be registered so that unmoved objects appear in the same locations in both images and their projection information should contain only translation (no rotation or skew).

Then, generate a set of random points from "Vector > Research Tools > Random points". The number of points should be large enough to contain both change and non-change locations in sufficient amounts to meet the sample requirements. If the number of points turns out to be too low, however, you can generate another set of random points and then merge both layers together when you finish labeling.

![](/assets/qgis1.png)

Enter a filename for the shapefile, make sure the "Add result to canvas" box is checked, and click OK to add the points as a layer. Then, right click on the new layer and select "Toggle Editing" and then "Open Attribute Table". To add fields for the label at each timestep, use the "Open field calculator" button (marked by an abacus). The only option you might want to change in the menu that follows is the "Expression" box, which you can use to set a default label. This is useful if the majority of your image is a certain class.

![](/assets/qgis2.png)

After creating one label field for each timestep, begin labeling points until you have met the sample requirement, and then delete the remainder of the points. A quick way to do this is:
1. Separate the Attribute table and the main QGIS windows so that they do not overlap.
2. Zoom in to the desired level of detail.
3. Select the next unlabeled point by clicking its row number on the far left side of the window.
4. Press Control+J to pan to the selected point.
5. Determine the class of the top layer, and then untick its visibility checkbox and determine the class of the bottom layer. Alternatively, pre-set the top layer transparency from its "Properties" menu so that you can see both layers at once.
6. If these classes differ from the default values, double click their fields in the Attribute table and change them.
7. Repeat from step 2 until you meet the sample requirements. To check how many change and non-change points you have, you can create a temporary field which checks for equality of the label fields and then sort by that field, as shown below.

![](/assets/qgis3.png)

![](/assets/qgis4.png)

After you finish, delete any columns other than the label fields, right click the layer containing the labels, and select "Save As...". Enter a filename and use the following configuration (which should be the default configuration in QGIS 2.4.0):

![](/assets/qgis5.png)

![](/assets/qgis6.png)

The path to the CSV file which is generated by this process should be given as the third argument to the ```run.py``` script, and the paths to the images corresponding to the first and second columns of the CSV should be given as the first and second arguments, respectively.
