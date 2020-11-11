# U-Net

![badge_original_author](https://img.shields.io/badge/original%20author-jakeret-red.svg)
![badge_tensorflow](https://img.shields.io/badge/tensorflow-%3E%3D1.0.0-orange.svg)

#### &nbsp; U-Net implemented with TensorFlow >= 1.0.0

&nbsp; This project is a reimplementation of the [Jakeret's U-Net](https://github.com/jakeret/tf_unet). <br/>
&nbsp; I reimplemented the code in my own style and modified it that can train multiple patch sizes at once. <br/><br/>

## How to use

``` bash
# Crop image
python "ImageCropping.py"

# Train U-Net
python "Training.py"

# Run TensorBoard
tensorboard --logdir="$TRAINING_LOG_PATH"

# Run demo
python "Demo.py"
```

&nbsp; You can change the settings using **config.ini** and **Setting.py**.