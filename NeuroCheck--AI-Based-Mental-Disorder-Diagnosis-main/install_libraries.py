import sys
import subprocess

packages = ['tensorflow' ,'pandas', 'numpy', 'scikit-fuzzy', 'scikit-learn', 'joblib']
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

