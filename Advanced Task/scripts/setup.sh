# RUN FROM PROJECT ROOT

source /opt/flight/etc/setup.sh
flight env activate gridware
module add gnu
pyenv virtualenv 3.9.5 drl
echo drl > .python-version
which python
python --version
pip3 install --proxy http://hpc-proxy00.city.ac.uk:3128 -r requirements.txt
pip3 install --proxy http://hpc-proxy00.city.ac.uk:3128 torch torchvision --index-url https://download.pytorch.org/whl/cu118