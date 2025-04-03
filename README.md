```
              _   _   _    _   _ _  ___ _    
  ___ _ _ ___| |_| |_(_)__| |_| | || _ \ |   
 / _ \ '_/ _ \  _|  _| / _| / /_  _|   / |__ 
 \___/_| \___/\__|\__|_\__|_\_\ |_||_|_\____|
---------------------------------------------
Oregon Lottery - Pick 4 Predictor (w/ Randife)
---------------------------------------------

=============================================
                  LINKS
  ----------------------------------------

 + Kaggle: https://orottick4.randife.com/kaggle

 + GitHub: https://orottick4.randife.com/github

 + Lottery: https://orottick4.randife.com/lotte


=============================================
                  ABOUT
  ----------------------------------------

Oregon Lottery - Pick 4 draw prediction is a simple
sample of Randife's application. It contains only 1
property and it is repeated daily with more than
10,000 instances.

Orottick4RL is an implementation of Randife for
predicting Oregon Lottery - Pick 4 drawing.


=============================================
                HOW TO USE
  ----------------------------------------

#----------#

import json
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import os
import warnings 
warnings.filterwarnings('ignore')

OROTTICK4RL_DIR = "/kaggle/buffers/orottick4rl"

os.system(f'mkdir -p "{OROTTICK4RL_DIR}"')
os.system(f'cd "{OROTTICK4RL_DIR}" && git clone https://github.com/dinhtt-randrise/orottick4rl.git')
os.system(f'cd "{OROTTICK4RL_DIR}" && git clone https://github.com/dinhtt-randrise/randife.git')

import sys 
sys.path.append(os.path.abspath(OROTTICK4RL_DIR))
import orottick4rl.orottick4rl as vok4

#----------#

BUY_DATE = '2025.03.09'
BUFFER_DIR = '/kaggle/buffers/orottick4rl'
LOTTE_KIND = 'p4a'
DATA_DF = None
DATE_CNT = 56 * 5
O_DATE_CNT = 7
TCK_CNT = 56 * 5
F_TCK_CNT = 250
RUNTIME = 60 * 60 * 11.5
PRD_SORT_ORDER = 'B'
HAS_STEP_LOG = True
RANGE_CNT = 52
M4P_OBS = False
M4P_CNT = 3
M4P_VRY = False
M4P_ONE = True
RESULT_DIR = '/kaggle/working'
LOAD_CACHE_DIR = '/kaggle/woring'
SAVE_CACHE_DIR = '/kaggle/working'
CACHE_CNT = -1
CACHE_ONLY = False

METHOD = 'simulate'
#METHOD = 'observe'
#METHOD = 'observe_range'
#METHOD = 'download'

#----------#

options = {'BUY_DATE': BUY_DATE, 'BUFFER_DIR': BUFFER_DIR, 'LOTTE_KIND': LOTTE_KIND, 'DATA_DF': DATA_DF, 'DATE_CNT': DATE_CNT, 'O_DATE_CNT': O_DATE_CNT, 'TCK_CNT': TCK_CNT, 'F_TCK_CNT': F_TCK_CNT, 'RUNTIME': RUNTIME, 'PRD_SORT_ORDER': PRD_SORT_ORDER, 'HAS_STEP_LOG': HAS_STEP_LOG, 'RANGE_CNT': RANGE_CNT, 'M4P_OBS': M4P_OBS, 'M4P_CNT': M4P_CNT, 'M4P_VRY': M4P_VRY, 'M4P_ONE': M4P_ONE, 'RESULT_DIR': RESULT_DIR, 'LOAD_CACHE_DIR': LOAD_CACHE_DIR, 'SAVE_CACHE_DIR': SAVE_CACHE_DIR, 'CACHE_CNT': CACHE_CNT, 'USE_GITHUB': USE_GITHUB, 'METHOD': METHOD, 'CACHE_ONLY': CACHE_ONLY}

vok4.Orottick4RLSimulator.run(options, vok4, None)

#----------#


=============================================
                 EXAMPLES
  ----------------------------------------

+ Predict Notebook: https://www.kaggle.com/code/dinhttrandrise/orottick4rl-predict-rsp-a-b-2025-03-09


=============================================
                 CACHES
  ----------------------------------------

  ----------------------------------------
            BACKWARD CACHES
  ----------------------------------------

[ 2025.03.23 ]

+ Notebook: https://www.kaggle.com/code/dinhttrandrise/orottick4rl-cache-p4a-o7r52-3-2025-03-23

[ 2024.03.24 ]

+ Notebook: https://www.kaggle.com/code/dinhttrandrise/orottick4rl-cache-p4a-o7r52-2024-03-24

[ 2023.03.26 ]

+ Notebook: https://www.kaggle.com/code/dinhttrandrise/orottick4rl-cache-p4a-o7r52-2023-03-26


```
