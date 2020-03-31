# AI-Photographer 
This project is for a paper under review "AI-Photographer: A real-time end-to-end motion blur prevention system" by Nahli et al

## System Overview and motivations

Regardless of amateurs an expert photographer is aware of all the photography techniques, his accumulated experience allows him to shoot artistic and sharp photography in whatever scene type, lighting or motion conditions. He usually shoots with the manual mode which gives him the full control on the exposure triangle elements in Figure.1 (ISO, Shutter Speed and Aperture). By balancing and steering these three key values he is able to artistically drive his camera and prevent all image corruption phenomena, including Under-exposure, Over-exposure, Noise and Motion Blur.
Motivated by fact that the majority of the world population are amateur photographers, we launched an AI-Photographer project, within a deep neural network has been trained to extract the latent patterns from expert photographers shooting data to learn their skills and accumulated experience. We then deploy our trained system on smartphone device, our device embedded AI-Photographer system proved its capability to offer a real-time motion blur prevention service and an artistic shooting just like an expert photographer do.

## Dependencies

You can install all dependencies by running one of the following commands

You need a [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) to use the environment setting.

```python
# Use TensorFlow without GPU
conda env create -f environments.yml 

# Use TensorFlow with GPU
conda env create -f environment-gpu.yml
```

Or you can manually install the required libraries (see the contents of the environemnt*.yml files) using pip.


## Usage



### To train the model

You'll need the data folder which contains the training images.

```python
python model.py
```

This will generate a file `model-<epoch>.h5` whenever the performance in the epoch is better than the previous best.  For example, the first epoch will generate a file called `model-000.h5`.

### Run the trained model

choose a scene and press the Autonomous Mode button.  Then, run the model as follows:

```python
python shoot.py model-<epoch>.h5
```





