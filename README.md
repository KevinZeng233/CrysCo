This repository includes CrysCo model to predict inorganic material properties via a hybrid graph-transformer based model

% Dataset
We trained our model mainly on MP21 datasets for 9 properties.
%Data preprocessing

You need to download cif file from materialproject database via pymatgen that we propose in in data_extraction.ipynb.
you need to save all cif files and 2 csv files including pretty formula and calculated properties. Please note that for some of the materials, there are same pretty formula. Thus, for one of the csv file we add one identifier number to end of each pretty formula to diffrentiate between those materials that have same formula but different structure.

% Graph generation and feature extraction:

You need to run data_geneeration.py:
python data_generation.py --path to cif structure  --first CSV file  --second CSV file  --output
please note that the first CVS file is related to the csv file with regular pretty formula and second CSV file is related to that csv file including pretty formula with an indentifier. 

Models:
CrysCo.py includes the model architecture. to train the CrysCo model, you need to run model_training.py

python model_training.py --data_dir "current path" --data "generated data obtained from running the data_generation.py"

you can easily change all parameters and models from parameter.py. 

please note that all generated data should be saved in processed directory. 

Example: for example the current directory is "C:/Users/mom19004/Downloads/sams/". for data generation and training the model you need to:

1- python C:/Users/mom19004/Downloads/sams/data_generation.py "C:/Users/mom19004/Downloads/sams/material" "Eh.csv" "Ehs.csv" "C:/Users/mom19004/Downloads/sams/processed/"M.pt"
2-python C:/Users/mom19004/Downloads/sams/model_training.py --data_dir "C:/Users/mom19004/Downloads/sams/" --data "M.pt"


