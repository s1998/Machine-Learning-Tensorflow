import pickle
from utils import *
import numpy as np

output = open('identifier_to_family_2.txt', 'rb')
family =pickle.load(output)	
li = []
for k in family.keys():
	# print(k, " : ", family[k])
	li.append(family[k])

from collections import Counter
counter = Counter(li)

count_200 = 0
counter_200 = 0
count_100 = 0
counter_100 = 0
count_050  = 0
counter_050  = 0
families_200 = []
families_100 = []
families_050 = []

for k in counter.keys():
	if(counter[k] >= 200):
		count_200 += 1
		counter_200 += counter[k]
		families_200.append(k)
	if(counter[k] >= 100):
		count_100 += 1
		counter_100 += counter[k]
		families_100.append(k)
	if(counter[k] >= 50):
		count_050 += 1
		counter_050 += counter[k]
		families_050.append(k)

print("200, 100, 50 : ", count_200, count_100, count_050) 
print("200, 100, 50 : ", counter_200, counter_100, counter_050)

seq_identifier = get_seq_identifier()
print("length of seq_iden, fam_200, fam_100, fam_050", 
	len(seq_identifier), len(families_200), len(families_100), len(families_050))

# Create a sequence to family mappings.
database_200 = []
database_100 = []
database_050 = []

for k in family.keys():
	if(family[k] in  families_200):
		database_200.append([seq_identifier[k], family[k]])
	if(family[k] in  families_100):
		database_100.append([seq_identifier[k], family[k]])
	if(family[k] in  families_050):
		database_050.append([seq_identifier[k], family[k]])
	
print(len(database_200))
print(len(database_100))
print(len(database_050))

output = open('dbase_200', 'wb')
pickle.dump(database_200, output)
output.close()
print("Written to dbase_200")

output = open('dbase_100', 'wb')
pickle.dump(database_100, output)
output.close()
print("Written to dbase_100")

output = open('dbase_050', 'wb')
pickle.dump(database_050, output)
output.close()	
print("Written to dbase_050")

"""
200, 100, 50 :  550 981 1787
200, 100, 50 :  273803 333626 389207
""" 

