from utils import *

"""
According to data_understanding.py, 
there are examples.
"""
no_of_examples = 554515

def test_get_seq_identifier():
	seq_identifier = get_seq_identifier()
	"""
	Q6GZX3 - 
	MSIIGATRLQNDKSDTYSAGPCYAGGCSAFTPRGTCGKDWDLGEQTCASGFCTSQPLCAR
	IKKTQVCGLRYSSKGKDPLVSAEWDSRGAPYVRCTYDADLIDTQAQVDQFVSMFGESPSL
	AERYCMRGVKNTAGELVSRVSSDADPAGGWCRKWYSAHRGPDQDAALGSFCIKNPGAADC
	KCINRASDPVYQKVKTLHAYPDQCWYVPCAADVGELKMGTQRDTPTNCPTQVCQIVFNML
	DDGSVTMDDVKNTINCDFSKYVPPPPPPKPTPPTPPTPPTPPTPPTPPTPPTPRPVHNRK
	VMFFVAGAVLVAILISTVRW
	
	Q948R8 - 
	MRSGGLEMMSSSAIVAFNLKEGKNWWWDVNESPVWQDRIFHVLAVLYGIVSVIAVIQLVR
	IQLRVPEYGWTTQKVFHFLNFMVNGVRALVFLFRRDAQNMQPEILQHILLDIPSLAFFTT
	YALLVLFWAEIYYQARAVSTDGLRPSFFTINAVVYVIQIALWLVLWWKPVHLMVIISKMF
	FAGVSLFAALGFLLYGGRLFLMLQRFPVESKGRRKKLQEVGYVTTICFTCFLIRCIMMCF
	DAFDDAADLDVLDHPILNFIYYLLVEILPSSLVLFILRKLPPKRGITQYHQIQ
	
	B2ZDY1 -
	MGLRYSKDVKDRYGDREPEGRIPITLNMPQSLYGRYNCKSCWFANKGLLKCSNHYLCLKC
	LTLMLRRSDYCGICGEVLPKKLVFENSPSAPPYEA

	"""
	Q6GZX3 = "\
	MSIIGATRLQNDKSDTYSAGPCYAGGCSAFTPRGTCGKDWDLGEQTCASGFCTSQPLCAR\
	IKKTQVCGLRYSSKGKDPLVSAEWDSRGAPYVRCTYDADLIDTQAQVDQFVSMFGESPSL\
	AERYCMRGVKNTAGELVSRVSSDADPAGGWCRKWYSAHRGPDQDAALGSFCIKNPGAADC\
	KCINRASDPVYQKVKTLHAYPDQCWYVPCAADVGELKMGTQRDTPTNCPTQVCQIVFNML\
	DDGSVTMDDVKNTINCDFSKYVPPPPPPKPTPPTPPTPPTPPTPPTPPTPPTPRPVHNRK\
	VMFFVAGAVLVAILISTVRW"
	Q6GZX3 = ''.join(Q6GZX3.split())

	Q948R8 = "\
	MRSGGLEMMSSSAIVAFNLKEGKNWWWDVNESPVWQDRIFHVLAVLYGIVSVIAVIQLVR\
	IQLRVPEYGWTTQKVFHFLNFMVNGVRALVFLFRRDAQNMQPEILQHILLDIPSLAFFTT\
	YALLVLFWAEIYYQARAVSTDGLRPSFFTINAVVYVIQIALWLVLWWKPVHLMVIISKMF\
	FAGVSLFAALGFLLYGGRLFLMLQRFPVESKGRRKKLQEVGYVTTICFTCFLIRCIMMCF\
	DAFDDAADLDVLDHPILNFIYYLLVEILPSSLVLFILRKLPPKRGITQYHQIQ"
	Q948R8 = ''.join(Q948R8.split())

	B2ZDY1 = "\
	MGLRYSKDVKDRYGDREPEGRIPITLNMPQSLYGRYNCKSCWFANKGLLKCSNHYLCLKC\
	LTLMLRRSDYCGICGEVLPKKLVFENSPSAPPYEA"
	B2ZDY1 = ''.join(B2ZDY1.split())

	ans = True
	ans = ans and (seq_identifier["Q6GZX3"] == Q6GZX3)
	ans = ans and (seq_identifier["Q948R8"] == Q948R8)
	ans = ans and (seq_identifier["B2ZDY1"] == B2ZDY1)
	ans = ans and (len(seq_identifier) == no_of_examples)

	if(ans):
		print("get_seq_identifier passed the tests")
	else:
		print("get_seq_identifier failed the tests")

def test_get_all_identifiers():
	identifiers = get_all_identifiers()
	check_keys = ["Q6GZX3", "Q948R8", "B2ZDY1"]
	count = {"Q6GZX3":0, "Q948R8":0, "B2ZDY1":0}
	for item in identifiers:
		for keys in check_keys:
			if(item == keys):
				count[item] += 1
	ans = True
	for keys in check_keys:
		ans = ans and (count[keys] == 1)
	ans = ans and (len(identifiers) == no_of_examples)

	if(ans):
		print("get_all_identifiers passed the tests")
	else:
		print("get_all_identifiers failed the tests")

def test_get_family_from_identifiers():
	output = open('identifier_to_family.txt', 'rb')
	family =pickle.load(output)
	ans = True
	ans = ans and (family["Q6GZX3"] == "Pox_G9-A16")
	ans = ans and (family["Q948R8"] == "DUF1084")
	ans = ans and (family["B2ZDY1"] == "zf-P11")
	
	if(ans):
		print("get_all_identifiers passed the tests")
	else:
		print("get_all_identifiers failed the tests")

def dbase_tester():
	output = open('identifier_to_family_2.txt', 'rb')
	family =pickle.load(output)	
	li = []
	for k in family.keys():
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

	ans = True
	ans = ans and (count_200 == 550)
	ans = ans and (count_100 == 981)
	ans = ans and (count_050 == 1787)



test_get_seq_identifier()
test_get_all_identifiers()
test_get_family_from_identifiers()
