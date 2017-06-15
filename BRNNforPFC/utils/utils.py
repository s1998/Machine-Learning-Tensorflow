import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pickle
from nltk import ngrams

data = pd.read_table('./../data/uniprot-all.tab', sep = '\t')
data = data.dropna(axis = 0, how = 'any')

# fig, ax = plt.subplots()
# data['Protein families'].value_counts().plot(ax=ax, kind='bar')
# # plt.show()
# fig.savefig('./../data/family_freq')

data_np = data.as_matrix()
print("Data loaded and NaN values dropped, shape : ", data_np.shape)

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

	db_ = {}
	for fam in families_included:
		db_[fam] = []

	for i in range(data_np.shape[0]):
		if(data_np[i, 3] in families_included):
			temp = [data_np[i, 0], data_np[i, 2], data_np[i, 3]]
			db_[data_np[i, 3]].append(temp)

	file_path = './../data/db_' + str(min_no_of_seq) +'_pickle'
	if(not os.path.isfile(file_path)):
		output = open(file_path, 'ab')
		pickle.dump(db_, output)
		output.close()

	no_length_seq_gt_200 = {}
	counter = 0
	total_counter = 0
	for fam in db_.keys():
		fam_seq = db_[fam]
		count_fam = 0
		for seq_no in range(len(fam_seq)):
			data = fam_seq[seq_no]
			seq = data[1]
			len_seq = len(seq)
			if(len(seq) > 1000):
				count_fam += 1
		counter += count_fam
		total_counter += len(fam_seq)				
		print(count_fam, len(fam_seq), count_fam*100/len(fam_seq),fam)

	print(min_no_of_seq, " : ", counter, total_counter, counter*100/total_counter)
	print(no_of_families)
	debug = input()

def map_creator():
	amino_acid_map = {}
	amino_acid_map['A'] = 1
	amino_acid_map['C'] = 2
	amino_acid_map['D'] = 3 # aspartic acid
	amino_acid_map['E'] = 4
	amino_acid_map['F'] = 5
	amino_acid_map['G'] = 6
	amino_acid_map['H'] = 7
	amino_acid_map['I'] = 8
	amino_acid_map['K'] = 9
	amino_acid_map['L'] = 10
	amino_acid_map['M'] = 11
	amino_acid_map['N'] = 12
	amino_acid_map['P'] = 13
	amino_acid_map['Q'] = 14
	amino_acid_map['R'] = 15
	amino_acid_map['S'] = 16
	amino_acid_map['T'] = 17
	amino_acid_map['U'] = 18 # Q9Z0J5 - confused with v ?
	amino_acid_map['V'] = 18
	amino_acid_map['W'] = 19
	amino_acid_map['Y'] = 20
	amino_acid_map['X'] = 21 # Q9MVL6 - undetermined
	amino_acid_map['B'] = 22 # asparagine/aspartic acid
	amino_acid_map['Z'] = 23 # glutamine/glutamic acid P01340

	families = []
	for i in range(data_np.shape[0]):
		families.append(data_np[i, 3])
	families_count = Counter(families)

	families_map = {}
	counter = 0
	
	for k, v in families_count.most_common():
		counter += 1
		families_map[k] = counter
	
	"""
	Class-II aminoacyl-tRNA synthetase family 3729
	3729 1
	RRF family 764
	764 87
	TGF-beta family 213
	213 510
	"""

	file_path = './../data/amino_acid_map' +'_pickle'
	if(not os.path.isfile(file_path)):
		output = open(file_path, 'ab')
		pickle.dump(amino_acid_map, output)
		output.close()

	file_path = './../data/families_map' +'_pickle'
	if(not os.path.isfile(file_path)):
		output = open(file_path, 'ab')
		pickle.dump(families_map, output)
		output.close()

def seq_merger_one_file():
	tri_seq_tot_0 = ""
	tri_seq_tot_1 = ""
	tri_seq_tot_2 = ""

	for i in range(data_np.shape[0]):
	# for i in range(2):
		if(i%1000 == 0):
			print("Itertion number : ", i)
		seq = str(data_np[i, 2])
		spaced_seq = " "
		for x in seq:
			spaced_seq += x + " "
		trigrams = ngrams(spaced_seq.split(), 3)
		tri_seq_list_0 = []
		tri_seq_list_1 = []
		tri_seq_list_2 = []

		counter_gram = 0
		for gram in trigrams:
			if(counter_gram%3 == 0):
				tri_seq_list_0.append(gram)
			if(counter_gram%3 == 1):
				tri_seq_list_1.append(gram)
			if(counter_gram%3 == 2):
				tri_seq_list_2.append(gram)
			counter_gram += 1

		tri_seq_str_0 = ""
		tri_seq_str_1 = ""
		tri_seq_str_2 = ""

		for gram in tri_seq_list_0:
			tri_seq_str_0 += gram[0] + gram[1] + gram[2] + " "
		for gram in tri_seq_list_1:
			tri_seq_str_1 += gram[0] + gram[1] + gram[2] + " "
		for gram in tri_seq_list_2:
			tri_seq_str_2 += gram[0] + gram[1] + gram[2] + " "
		
		tri_seq_tot_0 += tri_seq_str_0 + "dummy dummy dummy dummy dummy "		
		tri_seq_tot_1 += tri_seq_str_1 + "dummy dummy dummy dummy dummy "		
		tri_seq_tot_2 += tri_seq_str_2 + "dummy dummy dummy dummy dummy "		

	print(tri_seq_tot_0, "\n\n", tri_seq_tot_1, "\n\n", tri_seq_tot_2, "\n\n")	

	tri_seq_tot = tri_seq_tot_0
	tri_seq_tot += tri_seq_tot_1
	tri_seq_tot += tri_seq_tot_2

	file_path = "./../data/all_seq.txt"
	if(not os.path.isfile(file_path)):
		with open(file_path, "w") as otput_file:
			otput_file.write(tri_seq_tot)

# Ran these once, so files are saved 
seq_merger_one_file()
save_familywise_db()
save_familywise_db(100)
save_familywise_db(50)

map_creator()


"""

		3 440 0.6818181818181818 IspH family
		0 685 0.0 Complex I 20 kDa subunit family
		0 688 0.0 Glycosyltransferase 28 family, MurG subfamily
		4 650 0.6153846153846154 Ketol-acid reductoisomerase family
		5 231 2.1645021645021645 Cytidine and deoxycytidylate deaminase family
		76 1935 3.9276485788113695 G-protein coupled receptor 1 family
		0 967 0.0 Bacterial ribosomal protein bL33 family
	255 523 48.75717017208413 FGGY kinase family
		55 1164 4.725085910652921 MurCDEF family
		91 617 14.748784440842787 Methylthiotransferase family, MiaB subfamily
		0 257 0.0 Periviscerokinin family
		0 380 0.0 ABC transporter superfamily, Phosphate importer (TC 3.A.1.7) family
		4 222 1.8018018018018018 AB hydrolase superfamily, MetX family
	118 298 39.59731543624161 Major facilitator superfamily
		0 219 0.0 Isocitrate and isopropylmalate dehydrogenases family, LeuB type 1 subfamily
		0 395 0.0 Acetokinase family
		0 491 0.0 RecO family
		0 305 0.0 Bacterial ribosomal protein bL31 family, Type B subfamily
		5 802 0.6234413965087282 RNA polymerase alpha chain family
		1 275 0.36363636363636365 HutI family
		0 269 0.0 Radical SAM superfamily, MoaA family
		0 905 0.0 Bacterial ribosomal protein bS18 family
		3 336 0.8928571428571429 Thiamine-phosphate synthase family
		1 654 0.1529051987767584 DnaJ family
		0 205 0.0 Methylglyoxal synthase family
		0 385 0.0 Peptidase M20A family, DapE subfamily
		0 761 0.0 ATPase gamma chain family
		27 259 10.424710424710424 Glycosyltransferase 2 family
		0 691 0.0 RNase HII family
		0 247 0.0 Iron/manganese superoxide dismutase family
		0 280 0.0 Urease gamma subunit family
		32 819 3.907203907203907 TRAFAC class OBG-HflX-like GTPase superfamily, OBG GTPase family
		11 301 3.654485049833887 Ubiquitin-conjugating enzyme family
		0 204 0.0 GCKR-like family, MurNAc-6-P etherase subfamily
	272 277 98.19494584837545 Urease family
	411 411 100.0 Class-I aminoacyl-tRNA synthetase family, MetG type 1 subfamily
		0 546 0.0 RuvC family
	230 230 100.0 PEPCase type 1 family
		0 206 0.0 Pyruvate, phosphate/water dikinase regulatory protein family, PSRP subfamily
		2 205 0.975609756097561 Class-V pyridoxal-phosphate-dependent aminotransferase family, NifS/IscS subfamily
		0 501 0.0 RecF family
	281 285 98.59649122807018 Phosphoenolpyruvate carboxykinase (ATP) family
		0 749 0.0 ATPase epsilon chain family
		16 255 6.2745098039215685 DNA polymerase type-Y family
		0 206 0.0 Transaldolase family, Type 3B subfamily
	57 259 22.00772200772201 PP2C family
		10 622 1.607717041800643 TRAFAC class TrmE-Era-EngA-EngB-Septin-like GTPase superfamily, EngA (Der) GTPase family
		0 566 0.0 Gamma-glutamyl phosphate reductase family
		0 382 0.0 Casparian strip membrane proteins (CASP) family
		8 650 1.2307692307692308 TRNA pseudouridine synthase TruA family
		0 205 0.0 CobT family
		0 911 0.0 ATPase B chain family
		0 206 0.0 RbsD / FucU family, RbsD subfamily
	251 253 99.2094861660079 CarB family
		0 492 0.0 Imidazoleglycerol-phosphate dehydratase family
		0 469 0.0 MurB family
		0 235 0.0 Prokaryotic pantothenate kinase family
		1 393 0.2544529262086514 Dihydroorotate dehydrogenase family, Type 2 subfamily
		0 731 0.0 Bacterial ribosomal protein bS6 family
		46 236 19.491525423728813 Methyltransferase superfamily
		5 299 1.6722408026755853 Thiolase family
		0 433 0.0 XseB family
		0 253 0.0 Dethiobiotin synthetase family
		58 608 9.539473684210526 Amidase family, GatA subfamily
		0 565 0.0 Acetylglutamate kinase family, ArgB subfamily
		0 458 0.0 Purine/pyrimidine phosphoribosyltransferase family, PyrE subfamily
		0 363 0.0 Pyridoxamine 5'-phosphate oxidase family
	272 272 100.0 Phenylalanyl-tRNA synthetase beta subunit family, Type 1 subfamily
		1 1017 0.09832841691248771 TRAFAC class translation factor GTPase superfamily, Classic translation factor GTPase family, EF-Tu/EF-1A subfamily
		0 215 0.0 Thymidine kinase family
		0 368 0.0 SecB family
		800 800 100.0 TRAFAC class translation factor GTPase superfamily, Classic translation factor GTPase family, LepA subfamily
		0 466 0.0 Phosphoglycerate mutase family, BPG-dependent PGAM subfamily
		8 200 4.0 Pseudouridine synthase TruD family
		2 299 0.6688963210702341 Class-II pyridoxal-phosphate-dependent aminotransferase family, BioF subfamily
		6 254 2.3622047244094486 Adenosylhomocysteinase family
		0 831 0.0 Bacterial ribosomal protein bL32 family
		0 237 0.0 Beta-defensin family
		0 212 0.0 UPP synthase family
		11 501 2.1956087824351296 Small GTPase superfamily, Rab family
		1 900 0.1111111111111111 Universal ribosomal protein uL1 family
		0 340 0.0 Guanylate kinase family
		0 690 0.0 Complex I subunit 1 family
		0 286 0.0 Azoreductase type 1 family
		0 684 0.0 Universal ribosomal protein uS14 family
		0 1090 0.0 Universal ribosomal protein uS7 family
		0 733 0.0 AdoMet synthase family
		0 251 0.0 ATP phosphoribosyltransferase family, Long subfamily
		0 402 0.0 Thymidylate synthase family, Bacterial-type ThyA subfamily
		0 882 0.0 Universal ribosomal protein uS10 family
		4 677 0.5908419497784343 RecA family
		60 367 16.348773841961854 Glycosyltransferase 1 family, Bacterial/plant glycogen synthase subfamily
		0 548 0.0 TRAFAC class TrmE-Era-EngA-EngB-Septin-like GTPase superfamily, EngB GTPase family
		0 433 0.0 IspF family
		0 359 0.0 DEFL family
		6 748 0.8021390374331551 Adenylate kinase family
		0 210 0.0 UreE family
	676 1191 56.75902602854744 Cytochrome P450 family
		0 362 0.0 UPF0234 family
		0 211 0.0 G-protein coupled receptor T2R family
		0 207 0.0 Cytochrome c oxidase subunit 3 family
		0 207 0.0 GTP cyclohydrolase I family, QueF type 1 subfamily
	202 505 40.0 Peptidase M17 family
		0 354 0.0 NAGSA dehydrogenase family, Type 1 subfamily
	138 218 63.30275229357798 Peptidase S10 family
		0 529 0.0 Dehydroquinate synthase family
		0 212 0.0 DNA glycosylase MPG family
		0 659 0.0 Bacterial ribosomal protein bS21 family
		12 582 2.0618556701030926 Glutamyl-tRNA reductase family
		2 767 0.2607561929595828 Peptidase S14 family
		0 571 0.0 PlsX family
		0 200 0.0 CarA family
	522 522 100.0 Transketolase family, DXPS subfamily
	804 804 100.0 NAD-dependent DNA ligase family, LigA subfamily
		73 821 8.891595615103533 Phosphohexose mutase family
		0 384 0.0 MscL family
		14 351 3.988603988603989 GcvT family
		0 391 0.0 NadD family
		0 260 0.0 UPF0145 family
		0 571 0.0 DapA family
		0 874 0.0 Universal ribosomal protein uL18 family
		0 215 0.0 Eukaryotic ribosomal protein eS1 family
		0 764 0.0 RRF family
		0 484 0.0 UPF0102 family
		0 895 0.0 Universal ribosomal protein uL24 family
		0 610 0.0 NusB family
		0 633 0.0 TACO1 family
	223 225 99.11111111111111 KdpA family
		0 412 0.0 PanD family
		0 371 0.0 Alanine racemase family
	67 262 25.572519083969464 AB hydrolase superfamily, Lipase family
		0 760 0.0 RuvB family
		0 206 0.0 UPF0235 family
		0 443 0.0 LipB family
	962 968 99.3801652892562 RNA polymerase beta chain family
	170 214 79.4392523364486 Influenza viruses hemagglutinin family
		0 282 0.0 RecX family
		0 217 0.0 UPF0060 family
	563 583 96.56946826758147 ATP-dependent AMP-binding enzyme family
		0 252 0.0 DeoC/FbaB aldolase family, DeoC type 1 subfamily
		0 1039 0.0 Universal ribosomal protein uL14 family
		4 1055 0.3791469194312796 Universal ribosomal protein uL2 family
		0 642 0.0 RNA methyltransferase TrmD family
		144 522 27.586206896551722 ABC transporter superfamily
		0 555 0.0 UMP kinase family
	573 573 100.0 Class-II aminoacyl-tRNA synthetase family, ProS type 1 subfamily
		0 804 0.0 ATPase delta chain family
	141 417 33.81294964028777 Aldehyde dehydrogenase family
		0 251 0.0 Complex I subunit 6 family
		0 815 0.0 Tetrahydrofolate dehydrogenase/cyclohydrolase family
	83 287 28.9198606271777 CobB/CobQ family, CobQ subfamily
		0 451 0.0 TrpB family
		0 1117 0.0 ATCase/OTCase family
		0 642 0.0 ATPase C chain family
		0 302 0.0 HSP33 family
		0 420 0.0 ClpX chaperone family, HslU subfamily
	210 220 95.45454545454545 RNase Y family
		0 587 0.0 GHMP kinase family, IspE subfamily
		0 467 0.0 GcvH family
		0 384 0.0 Hfq family
		0 248 0.0 Conotoxin O1 superfamily
		18 424 4.245283018867925 Mitochondrial carrier (TC 2.A.29) family
		0 477 0.0 Argininosuccinate synthase family, Type 1 subfamily
		0 307 0.0 EIF-2B alpha/beta/delta subunits family, MtnA subfamily
		0 367 0.0 Type III pantothenate kinase family
		0 216 0.0 Transaldolase family, Type 1 subfamily
		0 724 0.0 RbfA family
		0 200 0.0 Frataxin family
		0 588 0.0 D-alanine--D-alanine ligase family
		0 796 0.0 Universal ribosomal protein uL29 family
		0 366 0.0 MinE family
		0 581 0.0 P-Pant transferase superfamily, AcpS family
	49 250 19.6 Class-I pyridine nucleotide-disulfide oxidoreductase family
	377 377 100.0 UvrB family
		0 260 0.0 ABC transporter superfamily, Methionine importer (TC 3.A.1.24) family
		0 257 0.0 SIMIBI class G3E GTPase family, UreG subfamily
		0 585 0.0 Complex I 23 kDa subunit family
		0 318 0.0 UPF0246 family
		0 659 0.0 RNA polymerase subunit omega family
		262 262 100.0 TPP enzyme family, MenD subfamily
		0 260 0.0 Fe(2+)-trafficking protein family
		33 274 12.043795620437956 Peptidase A1 family
		13 732 1.7759562841530054 FKBP-type PPIase family, Tig subfamily
		5 294 1.7006802721088434 UbiA prenyltransferase family
		0 465 0.0 Ribose 5-phosphate isomerase family
		0 208 0.0 SIS family, GmhA subfamily
		0 319 0.0 LpxC family
		0 342 0.0 GHMP kinase family, Homoserine kinase subfamily
		0 268 0.0 Transferase hexapeptide repeat family, LpxA subfamily
		0 311 0.0 LDH/MDH superfamily, LDH family
		0 212 0.0 PdxS/SNZ family
		2 646 0.30959752321981426 Radical SAM superfamily, Lipoyl synthase family
		0 259 0.0 Histone H2A family
		2 808 0.24752475247524752 Bacterial ribosomal protein bL9 family
	440 471 93.41825902335457 PurH family
		0 421 0.0 Peptidase A8 family
	591 593 99.6627318718381 IlvD/Edd family
		0 343 0.0 UPF0301 (AlgH) family
		0 619 0.0 RNA methyltransferase RlmH family
		3 573 0.5235602094240838 Anthranilate phosphoribosyltransferase family
		0 568 0.0 DapB family
	1830 2708 67.57754800590843 Class-I aminoacyl-tRNA synthetase family
		10 320 3.125 TRNA(Ile)-lysidine synthase family
		1 658 0.1519756838905775 RuBisCO large chain family, Type I subfamily
	1234 1280 96.40625 Heat shock protein 70 family
		102 822 12.408759124087592 Class-I aminoacyl-tRNA synthetase family, Glutamate--tRNA ligase type 1 subfamily
	256 276 92.7536231884058 Complex I subunit 5 family
		3 282 1.0638297872340425 TRAFAC class YlqF/YawG GTPase family, RsgA subfamily
		0 723 0.0 RimP family
		0 282 0.0 PRA-PH family
		0 368 0.0 Type-II 3-dehydroquinase family
		8 503 1.5904572564612327 Ribonuclease III family
		4 223 1.7937219730941705 ArgJ family
		4 392 1.0204081632653061 Small heat shock protein (HSP20) family
	73 279 26.164874551971327 Protein kinase superfamily, CMGC Ser/Thr protein kinase family, CDC2/CDKX subfamily
		0 520 0.0 QueC family
		0 222 0.0 UPF0391 family
		0 288 0.0 LuxS family
		0 655 0.0 Class I-like SAM-binding methyltransferase superfamily, rRNA adenine N(6)-methyltransferase family, RsmA subfamily
		0 472 0.0 Peptidase S24 family
		0 345 0.0 Aspartate/glutamate racemases family
		0 1076 0.0 Globin family
		0 476 0.0 AIR synthase family
		4 1262 0.31695721077654515 HisA/HisF family
		2 505 0.39603960396039606 Complex I 49 kDa subunit family
		0 848 0.0 Universal ribosomal protein uL4 family
		0 690 0.0 Fmt family
	1185 2365 50.10570824524313 ATPase alpha/beta chains family
		0 827 0.0 Elongation factor P family
		0 846 0.0 Methyltransferase superfamily, RsmH family
		0 738 0.0 Universal ribosomal protein uS9 family
		0 270 0.0 CoaE family
		0 263 0.0 Anhydro-N-acetylmuramic acid kinase family
		0 477 0.0 HMBS family
		0 385 0.0 NqrDE/RnfAE family
		0 234 0.0 YciB family
	292 292 100.0 Glycosyl hydrolase 13 family, GlgB subfamily
		0 807 0.0 Bacterial ribosomal protein bL27 family
		0 475 0.0 Queuine tRNA-ribosyltransferase family
		15 807 1.858736059479554 Short-chain dehydrogenases/reductases (SDR) family
		13 293 4.436860068259386 RNase Z family
		0 769 0.0 Universal ribosomal protein uL30 family
		23 788 2.918781725888325 Purine/pyrimidine phosphoribosyltransferase family
	293 443 66.13995485327314 ThiC family
		0 201 0.0 Peptidase M20B family
		0 482 0.0 Cytidylate kinase family, Type 1 subfamily
		300 300 100.0 Formate--tetrahydrofolate ligase family
		0 720 0.0 GroES chaperonin family
		0 239 0.0 Peptidase T1A family
		0 514 0.0 CrcB (TC 9.B.71) family
	251 260 96.53846153846153 PAL/histidase family
		5 589 0.8488964346349746 EPSP synthase family
		0 533 0.0 Succinate/malate CoA ligase beta subunit family
		0 423 0.0 Diaminopimelate epimerase family
		3 211 1.4218009478672986 Serpin family
		0 256 0.0 MnmG family, TrmFO subfamily
		0 305 0.0 TtcA family
		0 732 0.0 MnmA/TRMU family
		0 294 0.0 Frog skin active peptide (FSAP) family, Brevinin subfamily
		0 989 0.0 Universal ribosomal protein uS15 family
		18 317 5.678233438485805 Bacterial/plant glucose-1-phosphate adenylyltransferase family
		0 283 0.0 UPF0178 family
		9 412 2.1844660194174756 Methylthiotransferase family, RimO subfamily
		0 367 0.0 IspD/TarI cytidylyltransferase family, IspD subfamily
		0 240 0.0 Aerobic coproporphyrinogen-III oxidase family
	976 977 99.89764585465711 TRAFAC class translation factor GTPase superfamily, Classic translation factor GTPase family, EF-G/EF-2 subfamily
		0 516 0.0 Uroporphyrinogen decarboxylase family
		0 512 0.0 Shikimate dehydrogenase family
		4 596 0.6711409395973155 TRAFAC class TrmE-Era-EngA-EngB-Septin-like GTPase superfamily, TrmE GTPase family
		0 717 0.0 Glycosyltransferase 4 family, MraY subfamily
		0 218 0.0 Ferredoxin--NADP reductase type 2 family
		0 216 0.0 Histone H2B family
		0 220 0.0 MsrB Met sulfoxide reductase family
	205 210 97.61904761904762 Nitrite and sulfite reductase 4Fe-4S domain family
		0 279 0.0 ClpS family
		0 1066 0.0 Universal ribosomal protein uL16 family
		0 448 0.0 LeuD family, LeuD type 1 subfamily
		0 284 0.0 CcmE/CycJ family
		0 827 0.0 Universal ribosomal protein uL15 family
		14 294 4.761904761904762 NhaA Na(+)/H(+) (TC 2.A.33) antiporter family
		30 531 5.649717514124294 AccD/PCCB family
		0 994 0.0 Universal ribosomal protein uS11 family
		0 852 0.0 Bacterial ribosomal protein bS16 family
		0 881 0.0 Universal ribosomal protein uL11 family
		0 373 0.0 LpxK family
		9 621 1.4492753623188406 Lyase 1 family, Argininosuccinate lyase subfamily
		0 203 0.0 ABC transporter superfamily, Energy-coupling factor EcfA family
		0 749 0.0 KAE1 / TsaD family
		0 790 0.0 Bacterial ribosomal protein bL12 family
		0 536 0.0 QueA family
		0 399 0.0 KdsB family
	293 293 100.0 PsaA/PsaB family
	2126 3729 57.01260391525878 Class-II aminoacyl-tRNA synthetase family
		0 322 0.0 Nudix hydrolase family, RppH subfamily
	344 378 91.005291005291 OXA1/ALB3/YidC family, Type 1 subfamily
		0 652 0.0 RecR family
		11 1796 0.6124721603563474 Cytochrome b family
		0 595 0.0 GrpE family
		0 411 0.0 Class I-like SAM-binding methyltransferase superfamily, MenG/UbiE family
		0 419 0.0 FabH family
		4 429 0.9324009324009324 Ferrochelatase family
		0 264 0.0 MinC family
	198 242 81.81818181818181 Heme-copper respiratory oxidase family
		0 305 0.0 L/F-transferase family
		1 437 0.2288329519450801 Tubulin family
		0 807 0.0 Universal ribosomal protein uL10 family
		17 834 2.038369304556355 SHMT family
	608 948 64.13502109704642 Krueppel C2H2-type zinc-finger protein family
		0 615 0.0 Thymidylate kinase family
		0 801 0.0 Universal ribosomal protein uS17 family
		0 463 0.0 Pseudouridine synthase TruB family, Type 1 subfamily
		0 252 0.0 Glutaminase PdxT/SNO family
		0 826 0.0 EF-Ts family
		0 659 0.0 Triosephosphate isomerase family
	1066 1110 96.03603603603604 Chaperonin (HSP60) family
	192 281 68.32740213523131 Glycosyl hydrolase 13 family
	136 303 44.884488448844884 Peptidase S8 family
	227 618 36.73139158576052 Complex I subunit 2 family
	125 203 61.576354679802954 Major facilitator superfamily, Sugar transporter (TC 2.A.1.1) family
		0 230 0.0 MetA family
		0 538 0.0 DTD family
		0 536 0.0 Lgt family
		0 205 0.0 MntP (TC 9.B.29) family
	782 782 100.0 Polyribonucleotide nucleotidyltransferase family
		0 326 0.0 Universal ribosomal protein uS14 family, Zinc-binding uS14 subfamily
		0 670 0.0 Bacterial histone-like protein family
		6 224 2.6785714285714284 G-protein coupled receptor 1 family, Opsin subfamily
		0 664 0.0 RimM family
		4 351 1.1396011396011396 Dicarboxylate/amino acid:cation symporter (DAACS) (TC 2.A.23) family
		0 324 0.0 Metallo-beta-lactamase superfamily, Glyoxalase II family
		9 215 4.186046511627907 Chalcone/stilbene synthases family
		0 317 0.0 HAM1 NTPase family
		0 445 0.0 Transferase hexapeptide repeat family
		0 342 0.0 UPF0758 family
		0 374 0.0 HrcA family
		0 873 0.0 Universal ribosomal protein uL5 family
		0 248 0.0 Phosphofructokinase type A (PFKA) family, ATP-dependent PFK group I subfamily, Prokaryotic clade "B1" sub-subfamily
	43 420 10.238095238095237 XseA family
		0 632 0.0 NrdR family
		8 657 1.21765601217656 ClpX chaperone family
		0 284 0.0 TolB family
		0 509 0.0 SurE nucleotidase family
		0 356 0.0 Complex I 30 kDa subunit family
		0 244 0.0 Profilin family
	691 693 99.71139971139971 CTP synthase family
		0 495 0.0 Bacterial ribosomal protein bL31 family, Type A subfamily
		0 610 0.0 YqgF nuclease family
		0 335 0.0 Glyceraldehyde-3-phosphate dehydrogenase family
		0 1027 0.0 Universal ribosomal protein uS12 family
		0 254 0.0 PurK/PurT family
		0 323 0.0 KdsA family
		0 218 0.0 Somatotropin/prolactin family
		0 444 0.0 DXR family
		0 442 0.0 MraZ family
		1 267 0.37453183520599254 Glucosamine/galactosamine-6-phosphate isomerase family, NagB subfamily
		11 991 1.1099899091826437 Universal ribosomal protein uS3 family
	265 266 99.62406015037594 Class-I aminoacyl-tRNA synthetase family, ValS type 1 subfamily
	444 445 99.7752808988764 Class-I aminoacyl-tRNA synthetase family, IleS type 1 subfamily
		0 442 0.0 Glutamate 5-kinase family
		0 998 0.0 Universal ribosomal protein uL22 family
		2 208 0.9615384615384616 LDH/MDH superfamily, MDH type 2 family
		0 512 0.0 UPF0161 family
		0 285 0.0 ArgR family
	274 277 98.91696750902527 Vitamin-B12 independent methionine synthase family
		0 374 0.0 TrpC family
		0 720 0.0 PTH family
	817 818 99.87775061124694 TRAFAC class translation factor GTPase superfamily, Classic translation factor GTPase family, IF-2 subfamily
		0 381 0.0 Shikimate kinase family
		0 277 0.0 Phosphopentomutase family
		0 600 0.0 Class-II aminoacyl-tRNA synthetase family, Phe-tRNA synthetase alpha subunit type 1 subfamily
		0 287 0.0 ThiI family
	466 474 98.31223628691983 DNA mismatch repair MutL/HexB family
		0 275 0.0 UPF0434 family
		0 301 0.0 Peptidase T1B family
		0 218 0.0 PNP/UDP phosphorylase family
		0 331 0.0 Transferase hexapeptide repeat family, LpxD subfamily
		0 851 0.0 Universal ribosomal protein uS13 family
		0 251 0.0 Class-II aminoacyl-tRNA synthetase family, HisZ subfamily
		0 496 0.0 Complex I subunit 3 family
		2 315 0.6349206349206349 IF-3 family
		0 221 0.0 PAPS reductase family, CysD subfamily
		16 828 1.932367149758454 Adenylosuccinate synthetase family
	207 211 98.10426540284361 AAA ATPase family; Peptidase M41 family
		0 273 0.0 Class-V pyridoxal-phosphate-dependent aminotransferase family, SerC subfamily
	593 593 100.0 UvrC family
		0 362 0.0 Cytochrome c oxidase subunit 2 family
	149 364 40.934065934065934 Intermediate filament family
		0 461 0.0 RNase PH family
		0 505 0.0 Maf family
		13 215 6.046511627906977 Class-I pyridoxal-phosphate-dependent aminotransferase family
		53 259 20.463320463320464 UPF0061 (SELO) family
		0 736 0.0 Bacterial ribosomal protein bL34 family
	380 402 94.5273631840796 Heat shock protein 90 family
	341 342 99.70760233918129 Peroxidase family, Peroxidase/catalase subfamily
		0 422 0.0 Uracil-DNA glycosylase (UDG) superfamily, UNG family
	323 334 96.7065868263473 Alpha-IPM synthase/homocitrate synthase family, LeuA type 1 subfamily
		7 200 3.5 R-transferase family
		0 286 0.0 CinA family
		0 225 0.0 Phosphatidylserine decarboxylase family, PSD-B subfamily, Prokaryotic type I sub-subfamily
		1 277 0.36101083032490977 AP endonuclease 2 family
		0 363 0.0 MoaC family
	212 212 100.0 Enoyl-CoA hydratase/isomerase family; 3-hydroxyacyl-CoA dehydrogenase family
		0 712 0.0 Universal ribosomal protein uL13 family
		0 202 0.0 Phosphatidylserine decarboxylase family, PSD-A subfamily
		24 441 5.442176870748299 IspG family
		0 416 0.0 Reaction center PufL/M/PsbA/D family
		0 634 0.0 IF-1 family
		0 674 0.0 Methyltransferase superfamily, RNA methyltransferase RsmG family
		5 601 0.831946755407654 N-acetylglucosamine-1-phosphate uridyltransferase family; Transferase hexapeptide repeat family
		0 503 0.0 PlsY family
		0 251 0.0 Urease beta subunit family
		13 213 6.103286384976526 TGF-beta family
		0 658 0.0 Radical SAM superfamily, RlmN family
	729 766 95.16971279373368 RNA polymerase beta' chain family
	261 279 93.54838709677419 BPG-independent phosphoglycerate mutase family
		0 775 0.0 Bacterial ribosomal protein bL35 family
		0 256 0.0 Pterin-4-alpha-carbinolamine dehydratase family
		0 426 0.0 Bacterial ribosomal protein bL25 family, CTC subfamily
		0 370 0.0 UPF0176 family
		0 551 0.0 Class I-like SAM-binding methyltransferase superfamily, TrmB family
	697 697 100.0 Class-II aminoacyl-tRNA synthetase family, Type 1 subfamily
		0 317 0.0 Thz kinase family
		5 915 0.546448087431694 Enolase family
		0 260 0.0 Insulin family
		0 780 0.0 Bacterial ribosomal protein bS20 family
		0 342 0.0 RNase H family
		0 710 0.0 Endoribonuclease YbeY family
		0 435 0.0 Zinc-containing alcohol dehydrogenase family
	192 254 75.59055118110236 PPR family, P subfamily
		0 759 0.0 ATPase A chain family
	253 253 100.0 GcvP family
		2 462 0.4329004329004329 AccA family
		0 587 0.0 Radical SAM superfamily, Biotin synthase family
		0 233 0.0 PsbL family
		3 230 1.3043478260869565 Uridine kinase family
	673 923 72.91440953412784 Protein kinase superfamily, Ser/Thr protein kinase family
		0 472 0.0 Methyltransferase superfamily, PrmA family
		1 214 0.4672897196261682 HPrK/P family
		0 233 0.0 NifH/BchL/ChlL family
		0 485 0.0 DUTPase family
		0 429 0.0 DCTP deaminase family
		0 501 0.0 Class-II pyridoxal-phosphate-dependent aminotransferase family, Histidinol-phosphate aminotransferase subfamily
		0 318 0.0 TrpF family
		0 640 0.0 UppP family
		0 434 0.0 Peptidase T1B family, HslV subfamily
		0 533 0.0 SAICAR synthetase family
		0 730 0.0 RuvA family
		0 224 0.0 Pancreatic ribonuclease family
		0 214 0.0 Ribose-phosphate pyrophosphokinase family
		0 238 0.0 UPF0271 (lamB) family
		0 396 0.0 ThiG family
		6 230 2.608695652173913 Histidinol dehydrogenase family
		0 417 0.0 OMP decarboxylase family, Type 1 subfamily
		0 210 0.0 MobA family
		1 568 0.176056338028169 EPSP synthase family, MurA subfamily
		0 734 0.0 Universal ribosomal protein uS5 family
		24 315 7.619047619047619 NAD synthetase family
		1 414 0.24154589371980675 Class I-like SAM-binding methyltransferase superfamily, RNA methyltransferase RlmE family
		3 521 0.5758157389635317 UbiA prenyltransferase family, Protoheme IX farnesyltransferase subfamily
		0 511 0.0 UPRTase family
		0 956 0.0 Prokaryotic/mitochondrial release factor family
		5 723 0.6915629322268326 Phosphoglycerate kinase family
	304 305 99.67213114754098 HAK/KUP transporter (TC 2.A.72) family
		0 546 0.0 TrpA family
		7 258 2.7131782945736433 LpxB family
		5 749 0.6675567423230975 NDK family
		0 939 0.0 Bacterial ribosomal protein bL20 family
		0 773 0.0 Bacterial ribosomal protein bL21 family
		0 236 0.0 SepF family
	631 632 99.84177215189874 MnmG family
		0 832 0.0 Universal ribosomal protein uL6 family
		0 341 0.0 Methyltransferase superfamily, L-isoaspartyl/D-aspartyl protein methyltransferase family
		0 621 0.0 DMRL synthase family
		0 785 0.0 Bacterial ribosomal protein bL19 family
	117 489 23.926380368098158 Peptidase S1 family
		0 323 0.0 PsbE/PsbF family
		0 374 0.0 GTP cyclohydrolase I family
		12 238 5.042016806722689 Spermidine/spermine synthase family
	627 669 93.72197309417041 Intron maturase 2 family, MatK subfamily
		0 1052 0.0 Universal ribosomal protein uS19 family
		0 507 0.0 RnpA family
		0 277 0.0 CsrA family
		0 207 0.0 WhiA transcriptional regulatory family
		0 302 0.0 SfsA family
		0 580 0.0 YbaB/EbfC family
		82 584 14.04109589041096 DnaA family
		0 659 0.0 Bacterial CoaD family
		0 855 0.0 Complex I subunit 4L family
		0 210 0.0 GTP cyclohydrolase IV family
		0 357 0.0 TRAFAC class TrmE-Era-EngA-EngB-Septin-like GTPase superfamily, Era GTPase family
		0 259 0.0 PNP synthase family
		0 211 0.0 Cu-Zn superoxide dismutase family
		0 206 0.0 Cytochrome c family
		0 799 0.0 Bacterial ribosomal protein bL17 family
		0 273 0.0 ATP phosphoribosyltransferase family, Short subfamily
		0 294 0.0 PRA-CH family
		0 567 0.0 Acyl carrier protein (ACP) family
		0 228 0.0 PsbN family
		0 570 0.0 Thioester dehydratase family, FabZ subfamily
		10 807 1.2391573729863692 Class-II aminoacyl-tRNA synthetase family, Type-1 seryl-tRNA synthetase subfamily
		6 545 1.1009174311926606 NAD kinase family
		0 1014 0.0 Universal ribosomal protein uS8 family
		0 210 0.0 GTP cyclohydrolase I family, QueF type 2 subfamily
		4 343 1.1661807580174928 GreA/GreB family
		0 271 0.0 Acylphosphatase family
		1 218 0.45871559633027525 Histone H3 family
		2 1094 0.18281535648994515 Universal ribosomal protein uS2 family
		0 757 0.0 Bacterial ribosomal protein bL28 family
		0 867 0.0 Universal ribosomal protein uL3 family
		0 373 0.0 TatA/E family
		0 425 0.0 Peptidase M48B family
		3 519 0.5780346820809249 FPG family
		10 264 3.787878787878788 Peptidase C1 family
		179 202 88.61386138613861 Peptidase C19 family
		0 800 0.0 Universal ribosomal protein uL23 family
		2 1148 0.17421602787456447 Universal ribosomal protein uS4 family
	817 818 99.87775061124694 SecA family
	197 352 55.96590909090909 AAA ATPase family
		0 575 0.0 PanB family
		0 800 0.0 IPP transferase family
	118 638 18.495297805642632 GatB/GatE family, GatB subfamily
		0 877 0.0 Bacterial ribosomal protein bL36 family
		0 244 0.0 Methyltransferase superfamily, UbiG/COQ3 family
		48 208 23.076923076923077 MurCDEF family, MurE subfamily
		0 465 0.0 RapZ-like family
		304 304 100.0 TRAFAC class translation factor GTPase superfamily, Classic translation factor GTPase family, PrfC subfamily
	631 632 99.84177215189874 DNA mismatch repair MutS family
		0 368 0.0 Class-I aminoacyl-tRNA synthetase family, TyrS type 1 subfamily
		0 485 0.0 Polypeptide deformylase family
		0 674 0.0 Class-III pyridoxal-phosphate-dependent aminotransferase family, HemL subfamily
	458 623 73.51524879614767 GPI family
		0 306 0.0 MsrA Met sulfoxide reductase family
		0 646 0.0 NAD-dependent glycerol-3-phosphate dehydrogenase family
	215 216 99.53703703703704 TRAFAC class myosin-kinesin ATPase superfamily, Myosin family
		0 524 0.0 GatC family
		0 370 0.0 HesB/IscA family
		0 212 0.0 KdpC family
		0 472 0.0 Aconitase/IPM isomerase family, LeuC type 1 subfamily
		1 714 0.1400560224089636 Chorismate synthase family
		5 404 1.2376237623762376 Actin family
		0 716 0.0 SmpB family
		0 222 0.0 UreF family
		0 247 0.0 CobS family
		81 303 26.73267326732673 Complex I subunit 4 family
		0 263 0.0 UreD family
		0 537 0.0 Pantothenate synthetase family
		84 304 27.63157894736842 UDP-glycosyltransferase family
		0 404 0.0 FBPase class 1 family
		344 344 100.0 FGAMS family
		500  :  37735 272595 13.842880463691557

"""