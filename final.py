import sys
import pickle
import os
import glob
import time
from GenerateEmbeddings import generate_embeddings

input_path= 'data/validation'
output_path = 'out' 

dataset = glob.glob(input_path+'/*.txt')

generate_embeddings(dataset, input_path, output_path)

