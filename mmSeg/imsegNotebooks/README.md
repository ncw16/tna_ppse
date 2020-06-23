# dhSegment

[![Documentation Status](https://readthedocs.org/projects/dhsegment/badge/?version=latest)](https://dhsegment.readthedocs.io/en/latest/?badge=latest)

**dhSegment** is a tool for Historical Document Processing. Its generic approach allows to segment regions and
extract content from different type of documents. See
[some examples here](https://dhsegment.readthedocs.io/en/latest/intro/intro.html#use-cases).

The complete description of the system can be found in the corresponding [paper](https://arxiv.org/abs/1804.10371).

It was created by [Benoit Seguin](https://twitter.com/Seguin_Be) and Sofia Ares Oliveira at DHLAB, EPFL.

## Installation and usage
The [installation procedure](https://dhsegment.readthedocs.io/en/latest/start/install.html)
and examples of usage can be found in the documentation (see section below).

## Demo
Have a try at the [demo](https://dhsegment.readthedocs.io/en/latest/start/demo.html) to train (optional) and apply dhSegment in page extraction using the `demo.py` script.

## Documentation
The documentation is available on [readthedocs](https://dhsegment.readthedocs.io/).

##
If you are using this code for your research, you can cite the corresponding paper as :
```
@inproceedings{oliveiraseguinkaplan2018dhsegment,
  title={dhSegment: A generic deep-learning approach for document segmentation},
  author={Ares Oliveira, Sofia and Seguin, Benoit and Kaplan, Frederic},
  booktitle={Frontiers in Handwriting Recognition (ICFHR), 2018 16th International Conference on},
  pages={7--12},
  year={2018},
  organization={IEEE}
}
```

# For the image segmentation of Merchant's marks
Use PNGconverter.ipynb to produce PNG images for the segmentation algorithm
It is possible to use JPG, but PNG is recommended

## Creating the labelled data

Use labelling_mark_data.ipynb to produce the labels for training and evaluation. The labels will be created in prize_papers/all_labels. You will need to organise the labels along with the original images into train, evaluation and test folders along with a class.txt file.

The demo provided on installation of dhSegment has a good example

## For training, make sure you download the pretrained weights

"pretrained_models" has the scripts to download the weights for VGG16 and RESNET50
