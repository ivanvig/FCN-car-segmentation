import os

data_path = "ig02-cars/"
os.chdir(data_path)
filename_in = "cars_info.txt"
filename_out_eval = "cars_eval.txt"
filename_out_train = "cars_train.txt"

with open(filename_in, 'r') as fp:
	with open(filename_out_eval, 'w') as fp_oe:
		with open(filename_out_train, 'w') as fp_ot:
	
			lines = fp.readlines()
			for ln in lines:
				img = ln.split('.')[0]
				if ln[0] == ' ':
					continue
				if "notraining" in ln and "notest" in ln:
					fp_oe.write(img+'\n')
				else:
					fp_ot.write(img+'\n')
