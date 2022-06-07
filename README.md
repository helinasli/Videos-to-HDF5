# Videos-to-HDF5


## Installation
`pip2 install -r requirements.txt`

## How to Run:

1. Append the the right path for the folder networks and KTS (line 15 code), in order for them to be imported.
2. Run: `python2 generate_h5.py --model_name resnet --data path_to_data_folder --out h5_file_name.h5 > file_log.log & tail -f file_log.log`
