# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import os
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

### SET THESE PATHS MANUALLY #####################################################
# Full paths are required because otherwise the code will not know where to look
# when it is executed on one of the clusters.

use_gpu = True  # Are you running this code from the ETH Computer Vision Lab (Biwi)?

project_root = 'D:\dev_x\PHiSeg-code'
local_hostnames = ['Winston-Desktop']  # used to check if on cluster or not
log_root = 'D:\dev_x\phiseg_log'

##################################################################################

running_on_gpu_host = True if socket.gethostname() not in local_hostnames else False


def setup_GPU_environment():

    if use_gpu:

        hostname = socket.gethostname()
        print('Running on %s' % hostname)
        if not hostname in local_hostnames:
            logging.info('Setting CUDA_VISIBLE_DEVICES variable...')

            # This command is multi GPU compatible:
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(os.environ["SGE_GPU"].split('\n'))
            logging.info('SGE_GPU is %s' % os.environ['SGE_GPU'])
            logging.info('CUDA_VISIBLE_DEVICES is %s' % os.environ['CUDA_VISIBLE_DEVICES'])

    else:
        logging.warning('!! No GPU setup defined. Perhaps you need to set CUDA_VISIBLE_DEVICES etc...?')