VENV=~/venv310_nohf
echo "Creating virtualenv at $VENV"
python3.10 -m venv $VENV

source $VENV/bin/activate

pip install -U pip
pip install -U wheel

# jax
pip install -U "jax[tpu]==0.4.6" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

cd anticipation-kathli

pip install -e .