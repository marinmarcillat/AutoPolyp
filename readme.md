To install the conda env, open anaconda prompt and run following command

    conda create -n AutoPolyp  -c pytorch -c nvidia -c fastai -c anaconda -c conda-forge python=3.9 fastai jinja2 requests pandas tqdm opencv jupyterlab

To launch, open anaconda prompt and run following commands

    conda activate AutoPolyp 

    cd /path/to/AutoPolyp  # Navigate to your Autopolyp directory, for example: cd D:\Autopolip\AutoPolyp
    jupyter lab # Launch the jupyterlab interface

