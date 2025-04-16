import logging
import os
from datetime import datetime
# from from_root import from_root

# timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
os_path = os.getcwd()

# log path
log_path = os.path.join(os_path,'log',LOG_FILE)

# 

os.makedirs(log_path,exist_ok=True)

log_FILE_PATH = os.path.join(log_path,LOG_FILE)

# log format
logging.basicConfig(
    filename=log_FILE_PATH,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)



