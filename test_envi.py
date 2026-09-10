import numpy
import pandas
import sklearn
import matplotlib
import seaborn
import joblib
import torch

print("================================")
print("     NETWORK IDS ENVIRONMENT")
print("================================")

print("NumPy       :", numpy.__version__)
print("Pandas      :", pandas.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Matplotlib  :", matplotlib.__version__)
print("Seaborn     :", seaborn.__version__)
print("Joblib      :", joblib.__version__)
print("PyTorch     :", torch.__version__)

print("--------------------------------")

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: Not available")
    print("Training device: CPU")

print("================================")