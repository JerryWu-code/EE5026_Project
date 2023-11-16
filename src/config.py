# set the train/test ratio
train_ratio = 0.7
# set the target number of objectives
num_object = 25
# define image format we may use
image_format = tuple(('.png', '.jpg', '.jpeg'))

##########

# PCA & LDA Settings:
PCA_dimension_list = [40, 80, 200]
LDA_dimension_list = [2, 3, 9]
# select which selfie you want to reconstruct
case_selfie_num = 7
# set the random seed
seed = 5026
# set the selfie label, could find in data/PCA_train_map.txt
selfie_label = 25
# set training set size
train_target_num = 500

##########

# set directory
data_dir = '../data'
train_dir = '../data/train'
test_dir = '../data/test'

PIE_dir = '../data/PIE'
raw_selfie_dir = '../data/raw_selfie/'
final_selfie_dir = '../data/final_selfie/'

output_fig_dir = '../figs/'

PCA_train_dir = '../data/PCA_train_map.txt'
PCA_test_dir = '../data/PCA_test_map.txt'