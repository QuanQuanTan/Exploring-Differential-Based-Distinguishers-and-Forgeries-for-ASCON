step 1: run main0, which calls ASCON_P.mzn to get ASCON_P_2.txt,ASCON_P_3.txt,ASCON_P_4.txt
step 2: run main, which recursively go into the txt files in step 1 and run all possible trails

DANGER: Note that step 2 runs the program with up to 80 parallel instances at one go