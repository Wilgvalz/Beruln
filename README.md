# Beruln_Fast_Api

- file "dataset_modef.py":
   the DiverseVul dataset (available on huggingface) was used in this work. It was stabilized by equalizing the number of vulnerable/unvulnerable examples. The amount of data has also been reduced

- file "api_fr.py":
   a user request in FastAPI is sent in the format of a piece of C-code in a variable field
  
- file "api_num2_fr.py":
  a piece of C-code for analysis can be atteched as a txt-file
  
- the model was saved in pickle-file format, but due to the large size the file was not downloaded
- the final dataset is also not loaded

- file "train_model_2.ipynb":
  contains code for training the model, also added optimizations and other things that were used and can be viewed in the code

  P.S. a detailed description of process of assembling the dataset, training the model (Bert based model was used) and other details will be added later/can be viewed in the code. You can reproduce the results by yourself. 
