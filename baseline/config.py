# In this file, you can change simulated data
# According to this file, simulated data will be created

# Number of subjects in simulated data 
N_SUBJECTS = 50

# Number of ROIs in source brain graph for simulated data 
N_SOURCE_NODES = 35

# Number of ROIs in target brain graph for simulated data
N_TARGET_NODES = 160

# Number of traning epochs
N_EPOCHS = 400


####** DO NOT MODIFY BELOW **####
N_SOURCE_NODES_F =int((N_SOURCE_NODES*(N_SOURCE_NODES-1))/2)
N_TARGET_NODES_F =int((N_TARGET_NODES*(N_TARGET_NODES-1))/2)
###**************************####