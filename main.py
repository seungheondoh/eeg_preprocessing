import os
import pandas as pd
from preprocessor.deap import DEAP_preprocssor
from preprocessor.constants import DATASET

def main():
    DEAP_preprocssor(path= os.path.join(DATASET, 'DEAP'))

if __name__ == '__main__':
    main()