#Techniques

1. I've used `keras` library to train the Unet model
2. Trained model has been fitted using `stage1_train` dataset
3. To parse all images the script was used to detect ids of pictures and then to add to the list
4. To parse and compose all masks for one picture the zero matrix has been used: parsed pixels overwrite zeros where the mask covered the current region
5. For images I used 3 channels (for color detection), for masks - only one (because of 2 classes it was black and white image)
6. To analyze images I've performed resizing to 128*128 pixels

#How to use

##Google Colab Notebooks

1. If you want to use full script that includes fitting and prediction parts without downloading, you can follow my notebook by [this link](https://colab.research.google.com/drive/1z1lXR1POU_dL4FEP9MRNeqRlYC8vJ9io?usp=sharing)
2. If you want just execute program without installation, you can use [this notebook](https://colab.research.google.com/drive/1Il0E2Ah_CLc3ccSeEhowogoSmlJ90Iiy?usp=sharing)
##Program execution

1. Clone github repository
2. Create virtual environment (it's better to use Jetbrains PyCharm)
3. Install all requirements from the file `requirements.txt`
4. Run script called `Train.py` to train model. There are 2 parameters you can enter to setting your program:
    1) `--path_to_train` - the folder with images and masks directories. By default it's `data/stage1_train`
    2) `--path_to_model` - the folder where the model will be saved after fitting. By default it's root of the project
5. Run script called `Predict_masks.py` to predict masks on the test images. There are 3 parameters you can enter to setting your program:
    1) `--path_to_test` - the folder with images directories. By default it's `data/stage1_test`
    2) `--path_to_model` - path to the file with fitted model. By default it's `model.h5`
    3) `--path_to_output` - the folder where the test pictures with their masks will be saved. By default it's `test_with_predictions`