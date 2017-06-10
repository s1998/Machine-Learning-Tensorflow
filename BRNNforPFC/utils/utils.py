import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
# %matplotlib inline

data = pd.read_table('./../data/uniprot-all.tab', sep = '\t')
data = data.dropna(axis = 0, how = 'any')

# fig, ax = plt.subplots()
# data['Protein families'].value_counts().plot(ax=ax, kind='bar')
# # plt.show()
# fig.savefig('./../data/family_freq')

data_np = data.as_matrix()
print(data_np.shape)

def save_familywise_db(min_no_of_seq = 200):
	families = []
	for i in range(data_np.shape[0]):
		families.append(data_np[i, 3])
	families_count = Counter(families)
	
	no_of_families = 0
	families_included = []
	for k in families_count.keys():
		if(families_count[k] >= min_no_of_seq):
			no_of_families += 1
			families_included.append(k)
	# store the entire data family-wise
	# this would help to divide data 
	# into three parts with stratification

	db_200 = {}
	for fam in families_included:
		db_200[fam] = []

	for i in range(data_np.shape[0]):
		if(data_np[i, 3] in families_included):
			temp = [data_np[i, 0], data_np[i, 1], data_np[i, 3]]
			db_200[data_np[i, 3]].append(temp)

	if(not os.path.isfile('./../data/db_200.pickle')):
		

	print(no_of_families)

save_familywise_db()
save_familywise_db(100)
save_familywise_db(50)


