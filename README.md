This repository includes CrysCo model to predict inorganic material properties via a hybrid graph-transformer based model

% Dataset
We trained our model mainly on MP21 datasets for 9 properties.
%Data preprocessing

You need to download cif file from materialproject database via pymatgen that we propose in in data_extraction.ipynb.
you need to save all cif files and 2 csv files including pretty formula and calculated properties. Please note that for some of the materials, there are same pretty formula. Thus, for one of the csv file we add one identifier number to end of each pretty formula to diffrentiate between those materials that have same formula but different structure.
