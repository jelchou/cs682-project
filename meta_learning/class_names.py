import pandas as pd
from pathlib import Path
# Reading the file and storing the class names in a list
dataset_path = Path('/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/dataset/CUB_200_2011')
class_names = pd.read_csv(dataset_path/"classes.txt", header=None, sep=" ")
class_names.columns = ["id", "class_name"]

print(class_names[:5])