
import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from data import get_dataset

import models
#from utils.log import setup_logging, ResultsLog, save_checkpoint
#from utils.meters import AverageMeter, accuracy
#from utils.optim import OptimRegime
#from utils.misc import torch_dtypes

from datetime import datetime
from ast import literal_eval

model_names = sorted(name for name in models.__dict__ if name.islower() and
                     not name.startswith('_') and callable(models.__dict__[name]))

print(model_names)