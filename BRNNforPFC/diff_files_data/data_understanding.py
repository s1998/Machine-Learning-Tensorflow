from Bio import SeqIO
import re

counter = 0

data = SeqIO.parse("./data/uniprot_sprot.fasta", "fasta")
print(data)

maxLen = 0
minLen = 10000000
for seq_record in data:
    unique_identifier = seq_record.id.split('|')[1]
    print(unique_identifier)
    url = "http://pfam.xfam.org/protein/" + unique_identifier # Q6GZX4
    print(seq_record.id)
    print(type(seq_record.id))
    print(repr(seq_record.seq))
    print(len(seq_record))
    maxLen = max(maxLen, len(seq_record))
    minLen = min(minLen, len(seq_record))
    counter += 1
    # print(counter)
    # debug = input()

print("Total number of sequences  : ", counter)
print("Maximum length and minimmum length of sequences : ", maxLen, minLen)



