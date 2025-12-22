This is an RL project for my school. This project contains a training loop and a trained bot. you can train your own bot using the train_bot notebook or play against the already  made bot using the website

# How to install?
  ## requirements
  Cuda 13.0
  1. ````git clone https://github.com/yohlimem/Connect-four-AI.git````
  2. ````cd ./Connect-four-AI````
  3. ````pip install -r requirements.txt```` (for python 3.14)
  4. ```pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130````
  ## Actually Running
  - You can open the notebook ````train_AI.ipynb```` and run it to train your own AI
  - You can do ````python .\play_against_bot.py```` which will open a server at ````http://0.0.0.0:8000```` just go to this url in your browser and play against the ai
  
