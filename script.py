# Create the complete COVID-19 project structure and files
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

# Create project directory structure
project_dirs = [
    'COVID19_Dashboard_Project',
    'COVID19_Dashboard_Project/data',
    'COVID19_Dashboard_Project/data/raw',
    'COVID19_Dashboard_Project/data/processed',
    'COVID19_Dashboard_Project/src',
    'COVID19_Dashboard_Project/notebooks',
    'COVID19_Dashboard_Project/dashboards',
    'COVID19_Dashboard_Project/reports',
    'COVID19_Dashboard_Project/requirements'
]

for dir_path in project_dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

# Generate sample data structure (since we can't access the actual data)
print("\n✅ Project structure created successfully!")