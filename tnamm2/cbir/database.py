import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image, ImageOps
import scipy.ndimage as scindi
from .dataset import Dataset
import pickle
from . import utils



class Database(object):
    def __init__(self, dataset, encoder):
        # public:
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, str):
            self.dataset = Dataset(dataset)
        else:
            raise TypeError("Invalid dataset of type %s" % type(dataset))
        self.encoder = encoder

        # private:
        self._database = {}
        self._image_ids = {}
        return

    def get_image_id(self, image_path):
        # normalise the path first
        image_path = os.path.abspath(image_path)

        # lookup if image has already been hahed
        if image_path not in self._image_ids:
            # otherwise, store it
            self._image_ids[image_path] = hash(image_path)

        return self._image_ids[image_path]

    def index(self):
        """
        Generates the inverted index structure using tf-idf.
        This function also calculates the weights for each node as entropy.
        """
        # create inverted index
        print("\nGenerating index...")
        utils.show_progress(self.embedding, self.dataset.image_paths)
        return

    def is_indexed(self, image_path):
        image_id = self.get_image_id(image_path)
        return image_id in self._database

    def embedding(self, image_path):
        image_id = self.get_image_id(image_path)
        # check if has already been indexed
        if image_id not in self._database:
            # if not, calculate the embedding and index it
            image = self.dataset.read_image(image_path)
            
            if image.shape[0]<80 or image.shape[1]<80:
                print('Reshaping ' + image_path)
                image = scindi.zoom(image, (2, 2, 1), order = 1)

            self._database[image_id] = self.encoder.embedding(image)
        return self._database[image_id]

    def score(self, db_image_path, query_image_path):
        """
        Measures the similatiries between the set of paths of the features of each image.
        """
        # get the vectors of the images
        d = self.embedding(db_image_path)
        q = self.embedding(query_image_path)
        d = d / np.linalg.norm(d, ord=2)
        q = q / np.linalg.norm(q, ord=2)
        # simplified scoring using the l2 norm
        score = np.linalg.norm(d - q, ord=2)
        return score if not np.isnan(score) else 1e6

    def retrieve(self, query_image_path, n=4):
        # propagate the query down the tree
        scores = {}
        for db_image_path in self.dataset.image_paths:
            scores[db_image_path] = self.score(db_image_path, query_image_path)

        # sorting scores
        sorted_scores = {k: v for k, v in sorted(
            scores.items(), key=lambda item: item[1])}
        return sorted_scores

    def save(self, path=None):
        if path is None:
            path = "data"

        # store indexed vectors in hdf5
        with open(os.path.join(path, "index.pickle"), "wb") as f:
            pickle.dump(self._database, f)

        return True

    def load(self, path="data"):
        # load indexed vectors from pickle
        try:
            with open(os.path.join(path, "database.pickle"), "rb") as f:
                database = pickle.load(f)
                self._database = database
        except:
            print("Cannot load index file from %s/index.pickle" % path)
        return True

    def show_results(self, query_path, scores_dict, n=4, figsize=(10, 4)):
        fig, ax = plt.subplots(1, n + 1, figsize=figsize)
        ax[0].axis("off")
        ax[0].imshow(self.dataset.read_image(query_path))
        ax[0].set_title("Query image")
        img_ids = list(scores_dict.keys())
        scores = list(scores_dict.values())
        for i in range(1, len(ax)):
            ax[i].axis("off")
            ax[i].imshow(self.dataset.read_image(img_ids[i]))
            ax[i].set_title("#%d. %s Score:%.3f" %
                            (i, img_ids[i], scores[i]))
        return
    
    def show_results_grid(self, query_path, 
                          scores_dict, n=3, figsize=(4., 4)):
        
        def partitiongrid(n):
            def factorize(num):
                return [(n,round(num/n)) for n in 
                        range(1, num + 1) if num % n == 0]

            F = factorize(n)
            L = len(F)
            
            midind=int(L / 2) - (L+1) % 2
            
            return F[midind]
         

        def padding(img, expected_size):
            desired_size = expected_size
            delta_width = desired_size - img.size[0]
            delta_height = desired_size - img.size[1]
            pad_width = delta_width // 2
            pad_height = delta_height // 2
            padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
            return ImageOps.expand(img, padding)


        def resize_with_padding(img, expected_size):
            img.thumbnail((expected_size[0], expected_size[1]))
            # print(img.size)
            delta_width = expected_size[0] - img.size[0]
            delta_height = expected_size[1] - img.size[1]
            pad_width = delta_width // 2
            pad_height = delta_height // 2
            padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
            return ImageOps.expand(img, padding, fill='lightgrey')

        partition = partitiongrid(n+1)
        
        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols = partition,  # creates grid of axes
                 axes_pad = 1,  # pad between axes in inch.
                 )

        img_ids = list(scores_dict.keys())
        scores = list(scores_dict.values())
        
        qim = self.dataset.read_image(query_path)
        qim = Image.fromarray(qim)
        qimr = resize_with_padding(qim, (200,200))
        grid[0].imshow(qimr)
        grid[0].axis("off")
        grid[0].set_title("Query image", fontsize=24)
        
        for i, ax in enumerate(grid[1:]):
            nscore = np.exp(-scores[i+1])
            ax.axis("off")
            im = self.dataset.read_image(img_ids[i+1])
            im = Image.fromarray(im)
            imr = resize_with_padding(im, (200,200))
            ax.imshow(imr)
            ax.set_title("#%d. Score:%.2f" %
                        (i+1, nscore), 
                        fontsize=24)
        return
