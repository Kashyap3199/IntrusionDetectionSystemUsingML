**MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles.**
Modern automobiles, particularly connected and autonomous vehicles, contain several electronic control units that are linked via intra-vehicle networks to implement various functionalities and conduct actions. In order to communicate with other vehicles, infrastructure, and smart gadgets, modern automobiles are also connected to external networks via vehicle-to-everything technology. Modern cars are becoming more useful and connected, but because to their broad attack surfaces, this also makes them more susceptible to cyberattacks that target both internal and external networks. A lot of effort has gone into creating intrusion detection systems (IDSs) that use machine learning techniques to identify hostile cyber-attacks in order to secure vehicle networks.

In order to identify both known and unidentified assaults on vehicular networks, a multi-tiered hybrid IDS that combines a signature-based IDS and an anomaly-based IDS is proposed in this research. The vulnerabilities of intra-vehicle and external networks are also examined. The CICIDS2017 dataset and the CAN-intrusion-dataset, which represent the data from inside and outside of vehicles, respectively, show that the suggested system can accurately detect a variety of known attacks.

The proposed MTH-IDS framework comprises of four tiers of learning models in addition to two conventional ML stages (data pre-processing and feature engineering): Four tree-based supervised learners - decision tree (DT), random forest (RF), extra trees (ET), and extreme gradient boosting (XGBoost) — used as multi-class classifiers for known attack detection.
A stacking ensemble model and a Bayesian optimization with tree Parzen estimator (BO-TPE) method for supervised learner optimization. 
A cluster labeling (CL)k-means used as an unsupervised learner for zero-day attack detection; 
Two biased classifiers and a Bayesian optimization with Gaussian process (BO-GP) method for unsupervised learner optimization.

**Software setup:**


In order to implement this paper, initially python 3 has been installed.
The below steps are followed to install python 3:

1. Check if the system is compatible for python 3.
2. Open your web browser and navigate to the Downloads for Windows section of the official     Python website.
3. Search for your desired version of Python.
4. Select a link to download the executable installer. The download is approximately 25MB.
5. Run Executable Installer
6. Verify python was installed.
7. Verify pip was installed.
Next, A data science platform that runs on Windows, Apple, or Linux – Anaconda has been installed. Where it allows environments to be created for previous versions of Python or R or even later versions. Applications, such as Jupyter Notebook and Spyder, may be installed for each of these environments. The Navigator allows updating packages, creating environments, and launching applications.
 
 
  **<p align="center">Figure 1: Anaconda Navigator.</p>**
<p align="center">
<img src="https://github.com/Shireesha21/MTH_IDS_IoV/blob/main/MicrosoftTeams-image%20(6).png" width="500" />
</p>
 
 
  **<p align="center">Figure 2: Package Plan.</p>**
<p align="center">
<img src="https://github.com/Shireesha21/MTH_IDS_IoV/blob/main/MicrosoftTeams-image.png" width="500" />
</p>

  **<p align="center">Figure 3: Support Center.</p>**
<p align="center">
<img src="https://github.com/Shireesha21/MTH_IDS_IoV/blob/main/MicrosoftTeams-image%20(1).png" width="500" />
</p>



**Libraries used:**

Libraries/packages used for data preprocessing and feature engineering:

**NumPy:**
“import numpy as np”

NumPy is the fundamental package needed for scientific computing with Python. This package
contains:
• A powerful N-dimensional array object
• sophisticated (broadcasting) functions
• basic linear algebra functions
• basic Fourier transforms
• sophisticated random number capabilities
• tools for integrating Fortran code
• tools for integrating C/C++ code
Besides its obvious scientific uses, NumPy can also be used as an efficient multi-dimensional
container of generic data. Arbitrary data types can be defined. This allows NumPy to seamlessly and
speedily integrate with a wide variety of databases.
NumPy is a successor for two earlier scientific Python libraries: Numeric and Numarray.

**Pandas:**

“import pandas as pd”
Python's Pandas package is used to manipulate data sets. It offers tools for data exploration, cleaning, analysis, and manipulation. Both "Panel Data" and "Python Data Analysis" are mentioned in the term "Pandas". With the aid of Pandas, we can examine large data sets and draw conclusions based on statistical principles.
**Seaborn:**
“import seaborn as sns”
A package called Seaborn uses Matplotlib as its foundation to plot graphs. In order to see random distributions, it will be used.

**matplotlib.pyplot:**

“import matplotlib.pyplot as plt”

A MATLAB-like interface is offered by the Matplotlib plugin Pyplot. With the ability to use Python and the benefit of being free and open-source, Matplotlib is made to be just as usable as MATLAB.

**From skyline.preprocessing import LabelEncoder:**

“from skyline.preprocessing import LabelEncoder”

Target labels with values between 0 and n classes-1 should be encoded.
Instead of encoding the input X, this transformer should be used to encode the target values, or y.

**From sklearn.model_selection import train_test_split:**

“from sklearn.model_selection import train_test_split”

Python's train test split method divides arrays or matrices into random subsets for the train and test sets of data, respectively.


**Data preprocessing and feature engineering execution flow:**

1. To preprocess data- CICIDS2017 we first imported python libraries like numpy, pandas, sklearn, matplotlib

Code: 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder



2. Importing data set:

Read CICIDS2017_sample.csv.

Code: 

dataSet = pd.read_csv('CICIDS2017_sample.csv')

3. Visualization of data:

Used data visualization for the clear picture of network attacks.

In order to visualize the data present in the datasets we have used piechart and barcharts.


**Code:**

 **<p align="center">Figure 4: Visualization Of Data Code Sample.</p>**
<p align="center">
<img src="https://github.com/Shireesha21/MTH_IDS_IoV/blob/main/MicrosoftTeams-image%20(2).png" width="500" />
</p>

 **<p align="center">Figure 5: Bar Chart</p>**
<p align="center">
<img src="https://github.com/Shireesha21/MTH_IDS_IoV/blob/main/MicrosoftTeams-image%20(3).png" width="500" />
</p>

  **<p align="center">Figure 6: PIE Chart</p>**
<p align="center">
<img src="https://github.com/Shireesha21/MTH_IDS_IoV/blob/main/MicrosoftTeams-image%20(4).png" width="500" />
</p>


 
4. Data Preprocessing:

Here we used Z-score normalization inorder to remove the outliers in the dataset and to normalize the features into similar scale.
LabelEncoder is used to convert categorical(non-numeric) data into numeric values.
The count of attacks are found and assigned to respective attacks in numeric form
use k-means to cluster the data samples and select a proportion of data from each cluster
5. Split Data for Train and test

6. Feature Engineering: In order to achieve this, first we calculate the sum of importance scores and then select the important features from top to bottom until the accumulated importance reaches 90%. 


**Execution flow**

 **<p align="center">Figure 7: Flow Chart</p>**
<p align="center">
<img src="https://github.com/Shireesha21/MTH_IDS_IoV/blob/main/MicrosoftTeams-image%20(5).png" width="500" />
</p>

 
**Issues experiencing and solutions:**

We Faced Few errors while installing the softwares such as python version not compatible for our code so we reinstalled the python 3 version and Few Errors in packages.

**What is going to be done during the next round?**
CICIDS2017 dataset is used in the existing system, for the zero-day attack detection on this dataset the F1-score is 0.800. In the proposed system we will be using CAN-intrusion-dataset to detect both known and unknown cyber- attacks on vehicular networks with optimal performance. In the existing system only four algorithms named decision tree (DT), random forest (RF), extra trees (ET) and extreme gradient boosting (XGBoost) are used to detect the attacks.where as we will be using neural network algorithms like Convolutional Neural Network (CNN), Long short-term memory (LSTM) etc to improve the accuracy of the existing system. The four main stages of the proposed system are data pre-processing, feature engineering, a signature-based IDS, and an anomaly-based IDS. In data pre-processing the raw data is prepared for another data processing procedure. Feature engineering is the process of reducing the number of input variables when developing a predictive model. Signature-based anomaly detection is a type of intrusion detection method that involves looking for known patterns of malicious activity, called signatures, in network traffic or system logs, whereas in Anomaly-based intrusion detection system (IDS) is a type of IDS that detects cyber-attacks by identifying deviations from normal or expected patterns of behavior in network traffic or system activity.****




References:

https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning
