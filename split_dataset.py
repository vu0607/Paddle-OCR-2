with open('train.txt', 'r') as firstfile:
	lines = firstfile.readlines()

with open('train.txt', 'w') as firstfile, open('val.txt', 'a') as secondfile:
    for num, line in enumerate(lines,start=1):
	if  (0 < num < 733) or (2000 < num < 2733) or (5000 < num < 5733) or (7000 < num < 7733) or (12000 < num < 12733) or (14000 < num < 14733):
	    secondfile.write(line)
	else:
	    firstfile.write(line)


