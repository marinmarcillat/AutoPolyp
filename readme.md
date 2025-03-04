To install the conda env, open anaconda prompt and run following command

    mamba create -n AutoPolyp  -c pytorch -c nvidia -c fastai -c anaconda -c conda-forge python=3.9 fastai jinja2 requests pandas tqdm opencv jupyterlab fiftyone
    pip install opencv-contrib-python

To add support for gpu, check your cuda version with nvidia-smi command (cuda >= 11.6 required) and run following command

    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

To launch, open anaconda prompt and run following commands

    conda activate AutoPolyp 

    cd /path/to/AutoPolyp  # Navigate to your Autopolyp directory, for example: cd D:\AutoPolyp
    jupyter lab # Launch the jupyterlab interface

