DEPTH_DATA_URL="https://www.dropbox.com/s/qtab28cauzalqi7/depth_data.tar.gz?dl=1"
DATA_EXTRACT_DIR="data/"

wget -c $DEPTH_DATA_URL -O - | tar -xz -C $DATA_EXTRACT_DIR