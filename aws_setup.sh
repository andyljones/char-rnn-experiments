luarocks install nngraph
luarocks install stdlib

mkdir rnn
mkdir rnn/checkpoints
mkdir rnn/results

sudo mkdir /mnt/saturation-records
sudo mount /dev/xvdf /mnt/saturation-records
sudo chown ubuntu /mnt/saturation-records
