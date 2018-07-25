#!/usr/bin/env python

import os
import cv2
import sys
#import shutil
import random
import numpy as np
from detect import detect_face

"""
Example Call:
    $> python2.7 eigenfaces.py att_faces celebrity_faces
"""

class Eigenfaces(object):                                                       # *** COMMENTS ***
    faces_count=""

    faces_dir = '.'                                                             # directory path to the AT&T faces

    train_faces_count = 10                                                       # number of faces used for training
    # test_faces_count = 3   
    tot= train_faces_count                                                         # number of faces used for testing
    names=[]

                                                                         # training images count
    m = 100                                                                 # number of columns of the image
    n = 100                                                              # number of rows of the image
    mn = m * n                                                                  # length of the column vector

    """
    Initializing the Eigenfaces model.
    """
    def __init__(self, faces_dir):
        # print ('> Initializing started')

        _energy = 0.85
        self.faces_dir = faces_dir
        self.energy = _energy
        self.training_ids = []                                                  # train image id's for every at&t face
        self.faces_count=len(os.listdir(faces_dir))
        self.l = self.train_faces_count * self.faces_count

        L = np.empty(shape=(self.mn, self.l), dtype='float64')                  # each row of L represents one train image

        cur_img = 0
        for name in os.listdir(faces_dir):
            self.names.append(name)
            training_ids = random.sample(range(1, self.tot+1), self.train_faces_count)  # the id's of the 6 random training images
            self.training_ids.append(training_ids)                              # remembering the training id's for later

            for training_id in training_ids:
                path_to_img = os.path.join(self.faces_dir,
                        name , str(training_id) + '.jpg')          # relative path
                #print '> reading file: ' + path_to_img

                img = cv2.imread(path_to_img, 0)                                # read a grayscale image
                img_col = np.array(img, dtype='float64').flatten()              # flatten the 2d image into 1d

                L[:, cur_img] = img_col[:]                                      # set the cur_img-th column to the current training image
                cur_img += 1

        self.mean_img_col = np.sum(L, axis=1) / self.l                          # get the mean of all images / over the rows of L

        for j in range(0, self.l):                                             # subtract from all training images
            L[:, j] -= self.mean_img_col[:]

        C = np.matrix(L.transpose()) * np.matrix(L)                             # instead of computing the covariance matrix as
        C /= self.l                                                             # L*L^T, we set C = L^T*L, and end up with way
                                                                                # smaller and computentionally inexpensive one
                                                                                # we also need to divide by the number of training
                                                                                # images


        self.evalues, self.evectors = np.linalg.eig(C)                          # eigenvectors/values of the covariance matrix
        sort_indices = self.evalues.argsort()[::-1]                             # getting their correct order - decreasing
        self.evalues = self.evalues[sort_indices]                               # puttin the evalues in that order
        self.evectors = self.evectors[sort_indices]                             # same for the evectors

        evalues_sum = sum(self.evalues[:])                                      # include only the first k evectors/values so
        evalues_count = 0                                                       # that they include approx. 85% of the energy
        evalues_energy = 0.0
        for evalue in self.evalues:
            evalues_count += 1
            evalues_energy += evalue / evalues_sum

            if evalues_energy >= self.energy:
                break

        self.evalues = self.evalues[0:evalues_count]                            # reduce the number of eigenvectors/values to consider
        self.evectors = self.evectors[0:evalues_count]

        self.evectors = self.evectors.transpose()                               # change eigenvectors from rows to columns
        self.evectors = L * self.evectors                                       # left multiply to get the correct evectors
        norms = np.linalg.norm(self.evectors, axis=0)                           # find the norm of each eigenvector
        self.evectors = self.evectors / norms                                   # normalize all eigenvectors

        self.W = self.evectors.transpose() * L                                  # computing the weights

        # print ('> Initializing ended')

    """
    Classify an image to one of the eigenfaces.
    """
    def classify(self, path_to_img):
        # img = cv2.imread(path_to_img,0)                                        # read as a grayscale image
        img = detect_face(path_to_img)
        if(img is None):
            return -2
        img_col = np.array(img, dtype='float64').flatten()                      # flatten the image
        img_col -= self.mean_img_col                                            # subract the mean column
        img_col = np.reshape(img_col, (self.mn, 1))                             # from row vector to col vector

        S = self.evectors.transpose() * img_col                                 # projecting the normalized probe onto the
                                                                                # Eigenspace, to find out the weights

        diff = self.W - S                                                       # finding the min ||W_j - S||
        norms = np.linalg.norm(diff, axis=0)
        print(np.min(norms))
        closest_face_id = np.argmin(norms)                                      # the id [0..240) of the minerror face to the sample
        #threshold for not detecting any face
        if np.min(norms)>=2700:
            return -1
        return self.names[(closest_face_id // self.train_faces_count)]                   # return the faceid (1..40)

# path to image and folder should start from ./ etc.
#name is -1 if no match and the name of the person otherwise
def eigenfaces_main(path_to_folder,path_to_img):
    efaces = Eigenfaces(path_to_folder)                                       
    name=efaces.classify(path_to_img)
    #name==-1 means that face did not match
    #name==-2 means that face was not detected
    print(name)                                   

eigenfaces_main("./faces","1.jpg")
