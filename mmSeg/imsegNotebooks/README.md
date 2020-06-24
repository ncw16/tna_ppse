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

To train a model run
```
python train.py with <config.json>
```
where config.json is the configuration of training. There are two examples in this repo.

If you are training multiple times you will need to export your model to a different directory. "mark_model" must not exist before you run train.py

## Trained model

To use a model already trained on prize papers (50 training, 20 evaluation, 13 testing). Download the files and folders in this link and save in a "mark_model" folder

https://imperialcollegelondon.box.com/s/ef97mbrqa2z8353cgeeispao5ng9kjqn

## To run the segmentation algorithm

run
```
python test.py
```
## To run on the HPC

There are pbs file examples

# Creating the mark database

Once you have used dhSegment to segment initial regions, you use the create_final_db.ipynb to create the mark database. You must supply a name for the folder to store the database.

# Evaluating performance

Ensure that you have a complete output from the image segmentation. This should consist of mark images and a CSV file. You will also need a CSV of the ground truth.

The notebooks in eval_notebooks can be used to evaluate the performance.

When using them, make sure you provide the correct directories to your outputs and ground truth, as well as the original prize paper images.
