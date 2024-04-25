# This repository includes CrysCo model to predict inorganic material properties via a hybrid graph-transformer based model.

We will updates this repository further after acceptance of our manuscript. 

# Dataset

We trained our model mainly on MP21 datasets for 9 properties.

# Data preprocessing

"First, download CIF files from the Material Project database using pymatgen as outlined in the data_extraction.ipynb notebook. Ensure to save all CIF files and generate two CSV files containing pretty formulas and calculated properties.

Please note that in cases where multiple materials have the same pretty formula but different structures, we append a unique identifier number to the end of each pretty formula in one of the CSV files. This differentiation allows for proper identification of materials with similar formulas but distinct structures."

# Graph generation and feature extraction:

"Before training the model, you must run data_generation.py with the following command:

python data_preparation.py --path_to_cif_structure <path_to_cif_structure> --first_csv_file <first_CSV_file> --second_csv_file <second_CSV_file> --output <output_directory>


Please note that the first_CSV_file should contain regular pretty formula data, while the second_CSV_file should include pretty formula with an identifier."

This revision provides a clear instruction for running the data_generation.py script, specifying the required arguments and clarifying the purpose of each CSV file.

# Models:
"To train the CrysCo model, you will need to use the model_training.py script. First, ensure that CrysCo.py includes the desired model architecture. Then, execute the following command in your terminal:

python main.py --data_dir "current path" --data "generated data obtained from running the data_preparation.py"

Remember, you can easily modify all parameters and models by editing parameter.py. Additionally, ensure that all generated data is saved in the processed directory."

This revised version clarifies the steps required to train the model, specifies the commands to execute, and emphasizes the importance of saving generated data in the correct directory.
# Example: 
"To perform data generation and model training using the directory "C:/Users/mom19004/Downloads/sams/", you would need to follow these steps:"

1- python C:/Users/mom19004/Downloads/sams/data_preparation.py "C:/Users/mom19004/Downloads/sams/temp_folder/material_cif" "ehs.csv" "eh.csv" "C:/Users/mom19004/Downloads/sams/processed/"M.pt"


2-python C:/Users/mom19004/Downloads/sams/main.py --data_dir "C:/Users/mom19004/Downloads/sams/" --data "M.pt"

# Prediction

for prediction, you can use prediction.ipynb file. you need to run this notebook in the difrectory that you saved your pretrained model.

# References 

If you use this repository, in addition to our manuscript, please cite the following papers:

1- Omee, Sadman Sadeed, et al. "Scalable deeper graph neural networks for high-performance materials property prediction." Patterns 3.5 (2022).
2- Wang, Anthony Yu-Tung, et al. "Compositionally restricted attention-based network for materials property predictions." Npj Computational Materials 7.1 (2021): 77.
3- 




