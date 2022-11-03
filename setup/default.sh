# exit when any command fails
set -e

if true; then
    sudo apt update && sudo apt upgrade -y
fi

# install CUDA11.7 (taken from https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)
if false; then
    mkdir -p cuda_tmp && cd cuda_tmp
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.1-515.65.01-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.1-515.65.01-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda
    cd ../
    rm -rf cuda_tmp
fi

# install python 3.10
if true; then
    sudo apt install software-properties-common -y
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install python3.10 python3.10-dev python3.10-venv
    sudo apt-get install python3.10-tk
fi

# create virtual env
if true; then
    echo "Creating venv"
    python3.10 -m venv .venv
fi

# activate virtual env
if true; then
    source .venv/bin/activate
    which pip3
fi

# install requirements
if true; then
    pip install --upgrade luxai2022
fi

# download and install lux ai
if true; then
    git clone git@github.com:o-Oscar/Lux-Design-2022.git
    cd Lux-Design-2022
    pip install -e .
    cd ../
fi

# if true; then
#     pip install -e src