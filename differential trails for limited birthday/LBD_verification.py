# to be changed: overlapping equations should be removed
import os
import random
import numpy as np
import copy
import math
import itertools
from pprint import pprint
import time
import re

class ASCON():
	# linear layer forward
	Z0 = [0,19,28]
	Z1 = [0,61,39]
	Z2 = [0,1,6]
	Z3 = [0,10,17] 
	Z4 = [0,7,41]
	# linear layer inverse
	INV_Z0 = [0,3,6,9,11,12,14,15,17,18,19,21,22,24,25,27,30,33,36,38,39,41,42,44,45,47,50,53,57,60,63]   
	INV_Z1 = [0,1,2,3,4,8,11,13,14,16,19,21,23,24,25,27,28,29,30,35,39,43,44,45,47,48,51,53,54,55,57,60,61]
	INV_Z2 = [0,2,4,6,7,10,11,13,14,15,17,18,20,23,26,27,28,32,34,35,36,37,40,42,46,47,52,58,59,60,61,62,63]
	INV_Z3 = [1,2,4,6,7,9,12,17,18,21,22,23,24,26,27,28,29,31,32,33,35,36,37,40,42,44,47,48,49,53,58,61,63]
	INV_Z4 = [0,1,2,3,4,5,9,10,11,13,16,20,21,22,24,25,28,29,30,31,35,36,40,41,44,45,46,47,48,50,53,55,60,61,63]
	
	S_BOX = [0x4,0xb,0x1f,0x14,0x1a,0x15,0x9,0x2,0x1b,0x5,0x8,0x12,0x1d,0x3,0x6,0x1c,0x1e,0x13,0x7,0xe,0x0,0xd,0x11,0x18,0x10,0xc,0x1,0x19,0x16,0xa,0xf,0x17]
	INV_S_BOX = [20,26,7,13,0,9,14,18,10,6,29,1,25,21,19,30,24,22,11,17,3,5,28,31,23,27,4,8,15,12,16,2]
	# CONSTANT = [0xf0,0xe1,0xd2,0xc3,0xb4,0xa5,0x96,0x87,0x78,0x69,0x5a,0x4b]
	CONSTANT = [0xf0,0xe1,0xd2,0xc3,0xb4,0xa5,0x96,0x87,0x78,0x69,0x5a,0x4b]

	def add_constants(self,state,nr):
		# return state
		for i in range(8):
			state[2][-8+i] = state[2][-8+i] ^ ((self.CONSTANT[nr] >> (7-i)) & 0b1)
		return state
	def linear_layer(self,s):
		s_out = [[0]*64,[0]*64,[0]*64,[0]*64,[0]*64]
			
		for i in range(64):
			for term in self.Z0: s_out[0][i] = s_out[0][i] ^ s[0][(i-term)%64]
			for term in self.Z1: s_out[1][i] = s_out[1][i] ^ s[1][(i-term)%64]
			for term in self.Z2: s_out[2][i] = s_out[2][i] ^ s[2][(i-term)%64]
			for term in self.Z3: s_out[3][i] = s_out[3][i] ^ s[3][(i-term)%64]
			for term in self.Z4: s_out[4][i] = s_out[4][i] ^ s[4][(i-term)%64]
		
		s = s_out.copy()
		return s   
	def linear_layer_inverse(self,s):
		s_out = [[0 for i in range(64)] for j in range(5)]
		
		for i in range(64):
			for term in self.INV_Z0: s_out[0][i] = s_out[0][i] ^ s[0][(i-term)%64]
			for term in self.INV_Z1: s_out[1][i] = s_out[1][i] ^ s[1][(i-term)%64]
			for term in self.INV_Z2: s_out[2][i] = s_out[2][i] ^ s[2][(i-term)%64]
			for term in self.INV_Z3: s_out[3][i] = s_out[3][i] ^ s[3][(i-term)%64]
			for term in self.INV_Z4: s_out[4][i] = s_out[4][i] ^ s[4][(i-term)%64]
			
		s = s_out.copy()
		return s
	def S_box_sub(self,vals):
		v = (vals[0] << 4) ^ (vals[1] << 3) ^\
			(vals[2] << 2) ^ (vals[3] << 1) ^\
			(vals[4])
		s = self.S_BOX[v]
		sx = [(s >> 4),(s>>3) & 0b1, (s>>2) & 0b1, (s>>1) & 0b1, s & 0b1]
		return sx
	def S_box_main(self,state):
		for i in range(64):
			state[0][i],state[1][i],state[2][i],state[3][i],state[4][i] = \
			self.S_box_sub([state[0][i],state[1][i],state[2][i],state[3][i],state[4][i]])
		return state
	def S_box_sub_inverse(self,vals):
		v = (vals[0] << 4) ^ (vals[1] << 3) ^\
			(vals[2] << 2) ^ (vals[3] << 1) ^\
			(vals[4])
		s = self.INV_S_BOX[v]
		sx = [(s >> 4),(s>>3) & 0b1, (s>>2) & 0b1, (s>>1) & 0b1, s & 0b1]
		return sx
	def S_box_main_inverse(self,state):
		for i in range(64):
			state[0][i],state[1][i],state[2][i],state[3][i],state[4][i] = \
			self.S_box_sub_inverse([state[0][i],state[1][i],state[2][i],state[3][i],state[4][i]])
		return state

	def ascon_one_round(self,state,r):
		state = self.add_constants(state,r)
		state = self.S_box_main(state)
		state = self.linear_layer(state)
		return state

	def ascon_round_function(self,state,nr):
		for i in range(nr):
			state = self.ascon_one_round(state,i)
		return state

	def inv_ascon_one_round(self,state,r):
		state = self.linear_layer_inverse(state)
		state = self.S_box_main_inverse(state)
		state = self.add_constants(state,r)
		return state

	def inv_ascon_round_function(self,state,nr):
		for i in range(nr-1,-1,-1):
			state = self.inv_ascon_one_round(state,i)
		return state

	def computeDDT(self):
		DDT = np.zeros((32,32))
		for i in range(32):
			for j in range(32):
				DDT[i^j,self.S_BOX[i]^self.S_BOX[j]] += 1
		return DDT

	def computeSets(self,difference):
		# computing the size of D_in and D_out
		DDT = self.computeDDT()
		invDDT = DDT.T
		state = difference[-1]
		B = 0
		for i in range(64):
			S_in = (state[0][i] << 4) + (state[1][i] << 3) + (state[2][i] << 2) + (state[3][i] << 1) + (state[4][i] << 0) 
			B += np.log2(np.sum(DDT[S_in] > 0))
		A = 0
		state = difference[0]
		state = self.linear_layer_inverse(state)
		for i in range(64):
			S_out = (state[0][i] << 4) + (state[1][i] << 3) + (state[2][i] << 2) + (state[3][i] << 1) + (state[4][i] << 0) 	
			A += np.log2(np.sum(invDDT[S_out] > 0))
		# print('A:',A, 'B:',B)
		return A, B

	def generic_prob(self,difference):
		Din,Dout = self.computeSets(difference)
		cost = np.log2(max(min(np.sqrt(2**320/2**Din),np.sqrt(2**320/2**Dout)),2**321/((2**Din)*(2**Dout))))
		return cost
	def generic_prob_blackbox(self,difference):
		Din,Dout = self.computeSets(difference)
		front = np.log2(np.sqrt(2**320/2**Din))
		back = np.log2(np.sqrt(2**320/2**Dout))
		# print('cost:',np.log2(np.sqrt(2**320/2**Din)))
		return front,back


class ASCON_equation_computation(ASCON):
	# this records what inputs are to the AND0, AND1,..., AND4
	# by default, input1 has an NOT gate attached to it
	AND_GATE_IN = [[[0,4],[1]], 
				   [[1],[1,2]],
				   [[1,2],[3]],
				   [[3],[3,4]],
				   [[3,4],[0,4]]]
	# this records which AND gates affect y0, y1,...,y4
	AND_GATE_OUT_affect_y = [[0,1],
							 [1,2],
							 [3],
							 [3,4],
							 [0]]
	# this records what linear bits affect y0,y1,...,y4
	SBox_linear = [[0,3],
				   [0,1,4],
				   [1,2],
				   [1,2,3],
				   [3,4]]

	def compute_gate_activeness(self,Sbox_in,Sbox_out):
		# given the Sbox input and Sbox output, return a list of 5 tuples with each indicating if (input_1,input_2,output) are active or not
		y_linear_active = [0] * 5 # this will indicate if the linear part of the Sbox is active
		for i in range(len(self.SBox_linear)):
			y_linear_active[i] = [Sbox_in[bit] for bit in self.SBox_linear[i]]
			y_linear_active[i] = sum(y_linear_active[i]) % 2
		y_nonlinear_active = [y_linear_active[i] ^ Sbox_out[i] for i in range(5)]  # records if y is active because of nonlinear part

		AND_output = [0] * 5
		if y_nonlinear_active[2] == 1:
			AND_output[3] = 1
		if (AND_output[3] == 1 and y_nonlinear_active[3] == 0) or (AND_output[3] == 0 and y_nonlinear_active[3] == 1):
			AND_output[4] = 1
		if y_nonlinear_active[4] == 1:
			AND_output[0] = 1
		if (AND_output[0] == 1 and y_nonlinear_active[0] == 0) or (AND_output[0] == 0 and y_nonlinear_active[0] == 1):
			AND_output[1] = 1
		if (AND_output[1] == 0 and y_nonlinear_active[1] == 1) or (AND_output[1] == 1 and y_nonlinear_active[1] == 0):
			AND_output[2] = 1

		AND_input_1 = [0] * 5
		AND_input_2 = [0] * 5
		for i in range(5):
			for j in range(len(self.AND_GATE_IN[i][0])):
				AND_input_1[i] = AND_input_1[i] ^ Sbox_in[self.AND_GATE_IN[i][0][j]]
			for j in range(len(self.AND_GATE_IN[i][1])):
				AND_input_2[i] = AND_input_2[i] ^ Sbox_in[self.AND_GATE_IN[i][1][j]]

		return [(AND_input_1[i],AND_input_2[i],AND_output[i]) for i in range(5)]


	def find_equations(self,difference):
		# find difference immediately before Sbox and immediately after Sbox
		# returning equations in the form of LHS and RHS.
		# LHS contains exactly 1 tuple
		# RHS contains the remaining tuples/parity (RHS = 0 if there are no remaining tuples/parity)
		cipher = ASCON()
		difference_before = [cipher.linear_layer_inverse(difference[i]) for i in range(len(difference))]
		difference_after = [difference[i] for i in range(len(difference))]

		# builiding the equations
		LHS = []; RHS = [];
		for rd in range(1,len(difference)):
			for col in range(64):
				# get the input and output difference
				Sbox_in = [difference_after[rd-1][i][col] for i in range(5)]
				Sbox_out = [difference_before[rd][i][col] for i in range(5)]
				# from these input and output difference, find which gates are active
				AND_gate_activeness = self.compute_gate_activeness(Sbox_in,Sbox_out)
				for i,gate in enumerate(AND_gate_activeness):
					input1 = self.AND_GATE_IN[i][0]
					input2 = self.AND_GATE_IN[i][1]
					temp_L = []; temp_R = [];
					if gate == (0,0,0): continue # non active AND gates, no equations!
					elif gate == (0,1,0): # input1 must be 1 (before NOT gate)
						temp_L = [(rd-1,row,col) for row in input1]
						temp_R = [1]
					elif gate == (1,0,0): # input2 must be 0
						temp_L = [(rd-1,row,col) for row in input2]
						temp_R.append(0)
					elif gate == (1,1,0): # inputs must be the same (before NOT gate)
						temp_L = [(rd-1,row,col) for row in input1]
						temp_R = [(rd-1,row,col) for row in input2]
						temp_R.append(0)
					elif gate == (0,1,1): # input1 must be 0 (before NOT gate)
						temp_L = [(rd-1,row,col) for row in input1]
						temp_R.append(0)
					elif gate == (1,0,1): # input2 must be 1
						temp_L = [(rd-1,row,col) for row in input2]
						temp_R = [1]
					elif gate == (1,1,1): # inputs must be different (before NOT gate)
						temp_L = [(rd-1,row,col) for row in input1]
						temp_R = [(rd-1,row,col) for row in input2]
						temp_R.append(1)
					else:
						print('error in find_equations function')
						assert False

					# post-processing to get nice print outs
					# criteria:
					# 1. tempL should have exactly 1 tuple
					# 2. tempL and tempR should not have any tuples in common
					# 3. get rid of 0 if RHS is not 0, add in when there is no RHS
					# handing 2.
					L_index = 0
					while L_index < len(temp_L):
						for R_index in range(len(temp_R)):
							if temp_L[L_index] == temp_R[R_index]:
								del temp_L[L_index]
								del temp_R[R_index]
								L_index -= 1
								break
						L_index += 1
					# handing 1.
					if len(temp_L) == 0:
						for R_index in range(len(temp_R)):
							if type(temp_R[R_index]) == tuple:
								temp_L.append(temp_R[R_index])
								del temp_R[R_index]
								break
					while len(temp_L) > 1:
						temp_R.insert(0,temp_L[0])
						del temp_L[0]

					# handling 3.
					if len(temp_R) == 0: temp_R.append(0)
					if len(temp_R) > 1:
						if temp_R[-1] == 0: del temp_R[-1]

					# ready to add in LHS, RHS!
					flag = True
					for i in range(len(LHS)):
						if temp_L == LHS[i] and temp_R == RHS[i]:
							flag = False
					if flag == True:
						LHS.append(temp_L)
						RHS.append(temp_R)
		return LHS,RHS

	def make_string_equations(self,LHS,RHS):
		# convert LHS,RHS to a list with equations string
		equations = []
		for i in range(len(LHS)):
			string_temp = ''
			for tup in LHS[i]:
				string_temp += 'state['+str(tup[0])+']['+str(tup[1])+']['+str(tup[2])+'] ^ '
			string_temp = string_temp[:-2] + ' = '
			for stuff in RHS[i]:
				if type(stuff) == tuple:
					string_temp += 'state['+str(stuff[0])+']['+str(stuff[1])+']['+str(stuff[2])+'] ^ '
				else:
					string_temp += str(stuff)  + ' ^ '
			string_temp = string_temp[:-2]
			equations.append(string_temp)
		return equations
	def make_evaluating_equations(self,LHS,RHS):
		# convert LHS,RHS to a list with equations string
		equations = []
		for i in range(len(LHS)):
			string_temp = ''
			for tup in LHS[i]:
				string_temp += 'state['+str(tup[1])+']['+str(tup[2])+'] ^ '
			string_temp = string_temp[:-2] + ' = '
			for stuff in RHS[i]:
				if type(stuff) == tuple:
					string_temp += 'state['+str(stuff[1])+']['+str(stuff[2])+'] ^ '
				else:
					string_temp += str(stuff)  + ' ^ '
			string_temp = string_temp[:-2]
			equations.append(string_temp)
		return equations

	def print_equations(self,equations):
		for eqn in equations:
			print(eqn)

	def print_equations_to_file(self,equations,independent_list,infile,outfile,state_num):
		f = open(outfile,'w')
		for i in range(len(independent_list)):
			for j in range(len(independent_list[i][0])):
				f.write(equations[independent_list[i][0][j]])
				f.write('\n')
		f.close()

	def sort_constants(self,LHS,RHS):
		# this function helps to change the equations to satisfy the constants
		for i in range(len(LHS)):
			if 56 <= LHS[i][0][2] <= 63: # check column
				flag = False # check row
				if LHS[i][0][1] == 2: flag = True
				for j in range(len(RHS[i])): 
					if type(RHS[i][j]) == tuple and RHS[i][j][1] == 2: flag = True
				if flag == False: continue
				if (self.CONSTANT[LHS[i][0][0]+1] >> (63-LHS[i][0][2])) % 2 == 1: # add one to the constant for the additional round we add for backward
					if type(RHS[i][-1]) == tuple: RHS[i].append(1) # x1 = x2 + x3 --> append 1
					elif len(RHS[i]) == 1: RHS[i][0] ^= 1 # x1 = 1 --> x1 = 0 or the other way round
					else: del RHS[i][-1] # x1 = x2 + 1 --> x1 = x2
		return LHS,RHS


class ASCON_dependency_computation(ASCON):
	def __init__(self, nr, state_num):
		self.nr = nr
		self.state_num = state_num

	# the dependency of Sboxes
	forward_Sbox_dependency = [[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4]]
	reverse_Sbox_dependency = [[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4]]

	def compute_dependency_array(self):
		# the dependency array has length n * 2 + 1. Containing the bits involved at state_num at each round (half round included)
		# e.g. state[n][r][c] contains all the bits at state_num that influences the r row, c column at n/2 th round

		# compute the dependency chart based on the state_num state
		state = [[[[] for _ in range(64)] for _ in range(5)] for _ in range(self.nr*2+1)]
		# initialise the state_num with default
		for row in range(5): 
			for col in range(64): 
				state[int(self.state_num*2)][row][col].append([(row,col)])
		# forward direction
		for i in range(int(self.state_num*2+1),self.nr*2+1):
			if i % 2 == 0: # normal one round
				for row in range(5):
					for col in range(64):
						affected_cols = [(col-k) % 64 for k in eval('self.Z'+str(row))]
						for c in affected_cols:
							tmp = []
							for s in state[i-1][row][c]: tmp += s
							state[i][row][col].append(tmp)
			else: # .5 rounds
				for row in range(5):
					affected_rows = self.forward_Sbox_dependency[row]
					for col in range(64):
						tmp = []
						for r in affected_rows:
							for s in state[i-1][r][col]: tmp += s
						state[i][row][col].append(tmp)
		# reverse direction
		for i in range(int(self.state_num*2-1),-1,-1):
			if i % 2 == 0: # normal one round
				for row in range(5):
					affected_rows = self.reverse_Sbox_dependency[row]
					for col in range(64):
						tmp = []
						for r in affected_rows:
							for s in state[i+1][r][col]: tmp += s
						state[i][row][col].append(tmp)
			else: # .5 round
				for row in range(5):
					for col in range(64):
						affected_cols = [(col-k) % 64 for k in eval('self.INV_Z'+str(row))]
						for c in affected_cols:
							tmp = []
							for s in state[i+1][row][c]: tmp += s
							state[i][row][col].append(tmp)

		return state

	def sort_constraints(self,LHS,RHS,constraint_free_bits,constraint_dependency):
		# sort them based on round number. 
		# within each round, sort by column
		constraints = [[] for _ in range(self.nr)]
		temp_constraint_free_bits = [[] for _ in range(self.nr)]
		temp_constraint_dependency = [[] for _ in range(self.nr)]
		for i in range(len(LHS)):
			constraints[LHS[i][0][0]].append([LHS[i],RHS[i]])
			temp_constraint_free_bits[LHS[i][0][0]].append(constraint_free_bits[i])
			temp_constraint_dependency[LHS[i][0][0]].append(constraint_dependency[i])
		return constraints,temp_constraint_free_bits,temp_constraint_dependency

	def convert_after_sbox(self,LHS,RHS):
		# y[0] = (1-x[0])*(x[2]) ^ x[0] ^ (1-x[0])*x[1] ^ (1-x[4])*x[1] ^ x[3]
		# y[1] = x[3]*(1-x[1]) ^ x[3]*(1-x[2]) ^ x[1] ^ x[2]*(1-x[1]) ^ x[0] ^ x[4]
		# y[2] = (1-(1-x[3])*x[4]) ^ x[1] ^ x[2]
		# y[3] = x[0]*((1-x[3])+(1-x[4])) ^ x[1] ^ x[2]
		# y[4] = (1-x[0])*x[1] ^ (1-x[4])*x[1] ^ x[3] ^ x[4]
		
		counter = 0
		while counter < len(LHS):
			new_LHS = []
			new_RHS = []
			i = 0
			rd = LHS[counter][0][0]
			col = LHS[counter][0][2]
			y = np.zeros((5,5,5),dtype=int)
			y_c = np.zeros((5,),dtype=int)

			y[0,3,3] = 1; y[0,2,2] = 1; y[0,0,0] = 1; y[0,1,1] = 1;
			y[0,1,4] = 1; y[0,4,1] = 1; y[0,1,2] = 1; y[0,2,1] = 1; y[0,0,1] = 1; y[0,1,0] = 1;
			

			y[1,0,0] = 1; y[1,1,1] = 1; y[1,2,2] = 1; y[1,3,3] = 1; y[1,4,4] = 1;
			y[1,2,3] = 1; y[1,3,2] = 1; y[1,1,3] = 1; y[1,3,1] = 1; y[1,1,2] = 1; y[1,2,1] = 1;

			y[2,1,1] = 1; y[2,2,2] = 1; y[2,4,4] = 1;
			y[2,3,4] = 1; y[2,4,3] = 1; y_c[2] = 1;
			y_c[2] = 1;

			y[3,0,0] = 1; y[3,1,1] = 1; y[3,2,2] = 1; y[3,3,3] = 1; y[3,4,4] = 1;
			y[3,0,3] = 1; y[3,3,0] = 1; y[3,0,4] = 1; y[3,4,0] = 1;

			y[4,1,1] = 1; y[4,3,3] = 1; y[4,4,4] = 1;
			y[4,0,1] = 1; y[4,1,0] = 1; y[4,1,4] = 1; y[4,4,1] = 1;

			while counter < len(LHS) and LHS[counter][0][0] == rd and LHS[counter][0][2] == col:
				new_LHS.append(LHS[counter])
				new_RHS.append(RHS[counter])
				counter += 1
			for i in range(len(new_LHS)):
				# case 1: state[0][0][0] = 0
				if type(new_RHS[i][0]) != tuple:
					for j in range(5):
						for k in range(5):
							if y[j,k,new_LHS[i][0][1]] == 1:
								# x_lhs and x_lhs = 0/1
								if k == new_LHS[i][0][1]:
									y[j,new_LHS[i][0][1],new_LHS[i][0][1]] = 0
									if new_RHS[i][0] == 1:
										y_c[j] ^= 1
								else:
									if new_RHS[i][0] == 1:
										y[j,k,k] ^= 1
									y[j,k,new_LHS[i][0][1]] = 0
									y[j,new_LHS[i][0][1],k] = 0

				# case 2: state[0][0][0] = state[0][0][0]
				elif len(new_RHS[i]) == 1 and type(new_RHS[i][0]) == tuple:
					for j in range(5):
						for k in range(5):
							# x_lhs * x_k  and x_lhs = x1
							if y[j,k,new_LHS[i][0][1]] == 1:
								y[j,k,new_RHS[i][0][1]] ^= 1
								if k != new_RHS[i][0][1]:
									y[j,new_RHS[i][0][1],k] ^= 1
								y[j,k,new_LHS[i][0][1]] = 0
								y[j,new_LHS[i][0][1],k] = 0

				# case 3: state[0][0][0] = state[0][0][0] ^ 1
				elif len(new_RHS[i]) == 2 and type(new_RHS[i][1]) != tuple:
					for j in range(5):
						for k in range(5):
							# x_lhs and x_lhs = x1 + 1
							if k == new_LHS[i][0][1] and y[j,new_LHS[i][0][1],new_LHS[i][0][1]] == 1:
								y[j,new_LHS[i][0][1],new_LHS[i][0][1]] = 0
								y[j,new_RHS[i][0][1],new_RHS[i][0][1]] ^= 1
								y_c[j] ^= 1
							# x_lhs * x_k and x_lhs = x1 + 1
							elif y[j,k,new_LHS[i][0][1]] == 1:
								y[j,k,new_RHS[i][0][1]] ^= 1
								if k != new_RHS[i][0][1]:
									y[j,new_RHS[i][0][1],k] ^= 1
								y[j,k,k] ^= 1
								y[j,new_LHS[i][0][1],k] = 0
								y[j,k,new_LHS[i][0][1]] = 0

				# case 4: state[0][0][0] = state[0][0][0] ^ state[0][0][0]
				elif len(new_RHS[i]) == 2 and type(new_RHS[i][1]) == tuple:
					for j in range(5):
						for k in range(5):
							# if we have x_lhs  and x_lhs = x1 + x2
							if k == new_LHS[i][0][1] and y[j,new_LHS[i][0][1],new_LHS[i][0][1]] == 1:
								y[j,k,new_RHS[i][0][1]] ^= 1
								y[j,k,new_RHS[i][0][2]] ^= 1
								
							# if we have x_lhs * x_l and x_lhs = x1 + x2
							elif y[j,k,new_LHS[i][0][1]] == 1:
								y[j,k,new_RHS[i][0]] ^= 1
								if k != new_RHS[i][0]:
									y[j,new_RHS[i][0],k] ^= 1
								y[j,k,new_RHS[i][1]] ^= 1
								if k != new_RHS[i][1]:
									y[j,new_RHS[i][1],k] ^= 1
							y[j,k,new_LHS[i][0][1]] = 0
							y[j,new_LHS[i][0][1],k] = 0

				# case 5: state[0][0][0] = state[0][0][0] ^ state[0][0][0] ^ 1
				elif len(new_RHS[i]) == 3 and type(new_RHS[i][2]) == int:
					for j in range(5):
						for k in range(5):
							# if we have x_lhs  and x_lhs = x1 + x2 + 1 
							if k == new_LHS[i][0][1] and y[j,new_LHS[i][0][1],new_LHS[i][0][1]] == 1:
								y[j,new_RHS[i][0][1],new_RHS[i][0][1]] ^= 1
								y[j,new_RHS[i][1][1],new_RHS[i][1][1]] ^= 1
								y_c[j] ^= 1

							# if we have x_lhs * x_l and x_lhs = x1 + x2 + 1 
							elif y[j,k,new_LHS[i][0][1]] == 1:
								y[j,k,new_RHS[i][0]] ^= 1
								if k != new_RHS[i][0]:
									y[j,new_RHS[i][0],k] ^= 1
								y[j,k,new_RHS[i][1]] ^= 1
								if k != new_RHS[i][1]:
									y[j,new_RHS[i][1],k] ^= 1
								y[j,new_RHS[i][2],new_RHS[i][2]] ^= 1
							# deletion
							y[j,k,new_LHS[i][0][1]] = 0
							y[j,new_LHS[i][0][1],k] = 0

		# 	for i in range(len(new_RHS)):
		# 		print(new_LHS[i],new_RHS[i])
		# 	for i in range(5):
		# 		print('y'+str(i),end=' ')
		# 		for j in range(5):
		# 			for k in range(j,5):
		# 				if j == k:
		# 					if y[i,j,j] == 1:
		# 						print('x'+str(j),end=' ')
		# 				else:
		# 					if y[i,j,k] == 1:
		# 						print('x'+str(j)+'x'+str(k),end=' ')
		# 		print(y_c[i])
		# 	print()
		# 	print()
		# assert False
		# convert to cartesian equation
		# how?		



		return y,y_c

	def get_linear(self,y,y_c,round_num,col_num):
		y = np.array(y)
		lhs = []
		rhs = []
		for num in range(1,32):
			tmp = np.zeros((5,5),dtype=int)
			tmp_c = 0
			if (num >> 4) % 2 == 1: 
				tmp ^= y[0]; tmp_c ^= y_c[0]
			if (num >> 3) % 2 == 1: 
				tmp ^= y[1]; tmp_c ^= y_c[1]
			if (num >> 2) % 2 == 1: 
				tmp ^= y[2]; tmp_c ^= y_c[2]
			if (num >> 1) % 2 == 1: 
				tmp ^= y[3]; tmp_c ^= y_c[3]
			if (num >> 0) % 2 == 1: 
				tmp ^= y[4]; tmp_c ^= y_c[4]
			linear = True
			for i in range(5):
				for j in range(i,5):
					if tmp[i,j] == 1: linear = False
			if linear == True:
				lhs.append([]); rhs.append([])
				indices = [(num >> (4-i))%2 for i in range(5)]
				indices = [idx for idx, v in enumerate(indices) if v]
				lhs[-1].append((round_num,indices[0],col_num))
				for i in range(1,len(indices)):
					rhs[-1].append((round_num,indices[i],col_num))
				if len(rhs[-1]) == 0 or tmp_c == 1:
					rhs[-1].append(int(tmp_c))
		return lhs,rhs




	def update_free_bits(self,constraint_free_bits,bits_used):
		temp_constraint_free_bits = []
		for i in range(len(constraint_free_bits)):
			temp = []
			for j in range(len(constraint_free_bits[i])):
				if set(bits_used) & set(constraint_free_bits[i][j]) == set():
					temp.append(constraint_free_bits[i][j])
			temp_constraint_free_bits.append(temp)
		return temp_constraint_free_bits

	def find_max_independent_constraint_blackbox(self,LHS,RHS):
		Sboxes_activeness = [0 for _ in range(64)]
		# we can just count Sboxes
		for i in range(len(LHS)):
			if LHS[i][0][0] == 0:
				Sboxes_activeness[LHS[i][0][2]] = 1
		LHS_1 = [LHS[i] for i in range(len(LHS)) if LHS[i][0][0] == 1]
		RHS_1 = [RHS[i] for i in range(len(RHS)) if LHS[i][0][0] == 1]
		Sbox_inactive = [i for i in range(64) if Sboxes_activeness[i] == 0]

		indices = []
		for i in range(len(LHS_1)):
			temp = []
			exec('temp += [(('+str(LHS_1[i][0][2])+'-term) % 64) for term in self.Z' + str(LHS_1[i][0][1])+']')
			for j in range(len(RHS_1[i])):
				if type(RHS_1[i][j]) == int: break
				exec('temp += [(('+str(RHS_1[i][0][2])+'-term) % 64) for term in self.Z' + str(RHS_1[i][0][1])+']')
			indices.append(list(set(temp)))
		for i in range(len(indices)):
			index = []
			for j in range(len(indices[i])):
				if indices[i][j] in Sbox_inactive:
					index.append(indices[i][j])
			indices[i] = index
		min_index = -1
		LHS_fix = [LHS[i] for i in range(len(LHS)) if LHS[i][0][0] == 0]
		RHS_fix = [RHS[i] for i in range(len(RHS)) if LHS[i][0][0] == 0]
		while True:
			min_constraint = 10000
			for i in range(len(indices)):
				if len(indices[i]) > 0 and len(indices[i]) < min_constraint:
					min_index = i
					min_constraint = len(indices[i])
			if min_constraint == 10000: break
			else:
				LHS_fix.append(LHS_1[min_index])
				RHS_fix.append(RHS_1[min_index])
				Sbox_inactive.remove(indices[min_index][0])
				del LHS_1[min_index]
				del RHS_1[min_index]
				del indices[min_index]
				for i in range(len(indices)):
					index = []
					for j in range(len(indices[i])):
						if indices[i][j] in Sbox_inactive:
							index.append(indices[i][j])
					indices[i] = index
		return LHS_fix,RHS_fix

	def find_max_independent_constraints(self,LHS,RHS):
		# first, from each equation, retrieve the tuples
		tuples_bits = [[[LHS[i][0]] for i in range(len(LHS))][k] + [[RHS[i][j] for j in range(len(RHS[i])) if type(RHS[i][j]) == tuple] for i in range(len(RHS))][k] for k in range(len(LHS))]
		# find out the dependency arrays
		dependency_array = self.compute_dependency_array()
		# find dependent bits for each constraint
		constraint_dependency = [[] for i in range(len(LHS))]
		for i in range(len(tuples_bits)):
			for j in range(len(tuples_bits[i])):
				for k in dependency_array[int(tuples_bits[i][j][0]*2)][tuples_bits[i][j][1]][tuples_bits[i][j][2]]:
					if k not in constraint_dependency[i]:
						constraint_dependency[i].append(k)


		#####
		# Initialization: we keep track of the bits we have used up and what are the bits we can still control for each constraint
		#####
		constraint_free_bits = copy.deepcopy(constraint_dependency)
		bits_used = []
		constraints_wanted = []
		constraints_bits = [] # this contains the bits that affect the constraint
		constraints_fixers = [] # this contains the bits that we can alter
		constraints,constraint_free_bits,constraint_dependency = self.sort_constraints(LHS,RHS,constraint_free_bits,constraint_dependency)

		## METHOD 2
		# fix the Sboxes at state_num-0.5 first
		# then state_num+0.5 next
		# remainders
		#####
		# by default, we select all the Sboxes at round state_num-0.5
		for i in range(len(constraints[int(self.state_num-0.5)])):
			for bits in constraint_dependency[int(self.state_num-0.5)][i]: bits_used += bits
			constraints_wanted.append(copy.deepcopy(constraints[int(self.state_num-0.5)][i]))
			constraints_bits.append(copy.deepcopy(constraint_dependency[int(self.state_num-0.5)][i]))
			constraints_fixers.append(copy.deepcopy(constraint_dependency[int(self.state_num-0.5)][i]))
		bits_used = list(set(bits_used))

		if self.state_num-1.5 > 0:
			# replace the constraints at further away by linear constraints
			i = 0
			temp_constraints = []
			temp_constraint_dependency = []
			temp_constraint_free_bits = []
			while i < len(constraints[int(self.state_num-1.5)]):
				# gathering all the constraints from the same Sbox
				Sbox_num = constraints[int(self.state_num-1.5)][i][0][0][2] # [i][0][0][2] --> i^th constraint, LHS, first of LHS, column number
				lhs = [constraints[int(self.state_num-1.5)][i][0]]
				rhs = [constraints[int(self.state_num-1.5)][i][1]]
				j = i+1
				while j < len(constraints[int(self.state_num-1.5)]) and constraints[int(self.state_num-1.5)][j][0][0][2] == Sbox_num:
					lhs.append(constraints[int(self.state_num-1.5)][j][0])
					rhs.append(constraints[int(self.state_num-1.5)][j][1])
					j += 1
				i = j
				output,output_constants = self.convert_after_sbox(lhs,rhs)
				lhs,rhs = self.get_linear(output,output_constants,self.state_num-1,Sbox_num)
				temp = []
				for k in range(len(lhs)): temp += dependency_array[int(lhs[k][0][0]*2)][lhs[k][0][1]][lhs[k][0][2]]
				for k in range(len(rhs)): 
					for l in range(len(rhs[k])):
						if type(rhs[k][l]) == int: continue
						temp += dependency_array[int(rhs[k][l][0]*2)][rhs[k][l][1]][rhs[k][l][2]]
				temp_constraints += [lhs,rhs]
				temp_constraint_dependency += copy.deepcopy(temp)
				temp_constraint_free_bits += copy.deepcopy(temp)
			# combine all the constraints together with the dependencies
			constraint_free_bits = [constraint_free_bits[i][j] for i in range(1,len(constraint_free_bits)) for j in range(len(constraint_free_bits[i]))\
			 if constraints[i][j][0][0][0] != self.state_num-0.5 and self.state_num-1.5<=constraints[i][j][0][0][0]<=self.state_num+0.5] + [temp_constraint_free_bits]
			
			constraint_dependency = [constraint_dependency[i][j] for i in range(1,len(constraint_dependency)) for j in range(len(constraint_dependency[i]))\
			 if constraints[i][j][0][0][0] != self.state_num-0.5 and self.state_num-1.5<=constraints[i][j][0][0][0]<=self.state_num+0.5] + [temp_constraint_dependency]
			constraints = [constraints[i][j] for i in range(1,len(constraints)) for j in range(len(constraints[i]))\
			 if constraints[i][j][0][0][0] != self.state_num-0.5 and self.state_num-1.5<=constraints[i][j][0][0][0]<=self.state_num+0.5] + \
			 [[temp_constraints[i][j] for i in range(len(temp_constraints)) for j in range(len(temp_constraints[i]))]]
			constraint_free_bits = self.update_free_bits(constraint_free_bits,bits_used)
		else:
			# combine all the constraints together with the dependencies
			constraint_free_bits = [constraint_free_bits[i][j] for i in range(1,len(constraint_free_bits)) for j in range(len(constraint_free_bits[i]))\
			 if constraints[i][j][0][0][0] != self.state_num-0.5 and self.state_num-1.5<=constraints[i][j][0][0][0]<=self.state_num+0.5]
			
			constraint_dependency = [constraint_dependency[i][j] for i in range(1,len(constraint_dependency)) for j in range(len(constraint_dependency[i]))\
			 if constraints[i][j][0][0][0] != self.state_num-0.5 and self.state_num-1.5<=constraints[i][j][0][0][0]<=self.state_num+0.5]
			constraints = [constraints[i][j] for i in range(1,len(constraints)) for j in range(len(constraints[i]))\
			 if constraints[i][j][0][0][0] != self.state_num-0.5 and self.state_num-1.5<=constraints[i][j][0][0][0]<=self.state_num+0.5]
			constraint_free_bits = self.update_free_bits(constraint_free_bits,bits_used)

		# now, we settle based on the number of free Sboxes/bits
		available_constraints = True
		while available_constraints:
			available_constraints = False
			min_value = 1000
			min_value2 = 1000
			min_index = -1 # this keeps track of the best min_index for state_num+0.5 (high priority)
			min_index2 = -1 # this keeps track of the best min_index for state_num-1 (less priority)
			for i in range(len(constraints)):
				sbox = constraints[i][0][0][0]
				if sbox == self.state_num+0.5:
					if len(constraint_free_bits[i]) < min_value and len(constraint_free_bits[i]) > 0:
						min_index = i
						min_value = len(constraint_free_bits[i])
				elif sbox == self.state_num-1:
					if len(constraint_free_bits[i]) < min_value and len(constraint_free_bits[i]) > 0:
						min_index2 = i
						min_value2 = len(constraint_free_bits[i])
			if min_index == -1: min_index = min_index2 # use min_index 2 when we exhaust all state_num+0.5
			constraints_wanted.append(copy.deepcopy(constraints[min_index]))
			constraints_bits.append(copy.deepcopy(constraint_dependency[min_index]))
			constraints_fixers.append(copy.deepcopy(constraint_free_bits[min_index][0]))
			for bits in constraint_dependency[min_index]: bits_used += bits
			bits_used = list(set(bits_used))
			constraint_free_bits = self.update_free_bits(constraint_free_bits,bits_used)
			for i in range(len(constraint_free_bits)):
				if len(constraint_free_bits[i]) > 0: available_constraints = True
		return constraints_wanted,constraints_bits,bits_used,constraints_fixers

	def fix_constraints(self,constraints,constraints_bits,constraints_fixers):
		cypher = ASCON_equation_computation()
		ascon = ASCON()
		equations = cypher.make_evaluating_equations([constraints[i][0] for i in range(len(constraints))],[constraints[i][1] for i in range(len(constraints))])
		state = [[0 for _ in range(64)] for _ in range(5)]

		for r in range(5):
			rand_num = random.randint(0,2**64-1)
			for c in range(64):
				state[r][c] = (rand_num >> c) % 2
		for i in range(len(constraints)):
			if constraints[i][0][0][0] == self.state_num-0.5: # settle these Sboxes first
				exec(equations[i],locals())
			else:
				break
		state = ascon.S_box_main(state)
		for j in range(i,len(constraints)):
			sbox_num = constraints[j][0][0][0]
			if sbox_num == self.state_num+0.5:
				# find out the parity of the bits
				# check if lhs and rhs has states affected by constants
				const_flag = False
				if 56 <= constraints[j][0][0][2] <= 63:
					if constraints[j][0][0][1] == 2: const_flag = True
					for k in range(len(constraints[j][1])): 
						if type(constraints[j][1][k]) == tuple and constraints[j][1][k][1] ==2: const_flag = True
					if const_flag and ((ascon.CONSTANT[int(state_num+1.5)] >> (63-constraints[j][0][0][2])) % 2) == 1: 
						if type(constraints[j][1][-1]) == tuple: constraints[j][1].append(1)
						elif len(constraints[j][1]) == 1: constraints[j][1][0] ^= 1


				if type(constraints[j][1][-1]) == tuple: parity = 0
				else: parity = constraints[j][1][-1]
				parity_tuple = 0
				for k in range(len(constraints_bits[j])): 
					for l in range(len(constraints_bits[j][k])):
						parity_tuple ^= state[constraints_bits[j][k][l][0]][constraints_bits[j][k][l][1]]
				if parity_tuple != parity:
					state[constraints_fixers[j][0][0]][constraints_fixers[j][0][1]] ^= 1
			if sbox_num == self.state_num-1:
				if type(constraints[j][1][-1]) == tuple: parity = 0
				else: parity = constraints[j][1][-1]
				parity_sbox = 0
				for sbox in constraints_bits[j]:
					if 56 <= sbox[0][1] <= 63: 
						parity_sbox ^= ascon.INV_S_BOX[(state[sbox[0][0]][sbox[0][1]]<<4) ^ (state[sbox[1][0]][sbox[1][1]]<<3) ^ (state[sbox[2][0]][sbox[2][1]]<<2) 
						^ (state[sbox[3][0]][sbox[3][1]]<<1) ^ (state[sbox[4][0]][sbox[4][1]]<<0) ^ ((ascon.CONSTANT[int(state_num+1.5)] >> (63-sbox[0][1])) % 2)]
					else:
						parity_sbox ^= ascon.INV_S_BOX[(state[sbox[0][0]][sbox[0][1]]<<4) ^ (state[sbox[1][0]][sbox[1][1]]<<3) ^ (state[sbox[2][0]][sbox[2][1]]<<2) 
						^ (state[sbox[3][0]][sbox[3][1]]<<1) ^ (state[sbox[4][0]][sbox[4][1]]<<0)]
				# if the parity is wrong,
				# we inv the fixer Sbox, alter the bit we want, then Sbox back
				if (parity_sbox >> (4-constraints[j][0][0][1])) % 2 != parity:
					s = ascon.INV_S_BOX[ \
					(state[constraints_fixers[j][0][0]][constraints_fixers[j][0][1]] << 4) ^ (state[constraints_fixers[j][1][0]][constraints_fixers[j][1][1]] << 3) ^ \
					(state[constraints_fixers[j][2][0]][constraints_fixers[j][2][1]] << 2) ^ (state[constraints_fixers[j][3][0]][constraints_fixers[j][3][1]] << 1) ^ \
					(state[constraints_fixers[j][4][0]][constraints_fixers[j][4][1]] << 0)]
					s ^= (1 << (4-constraints[j][0][0][1]))
					s = ascon.S_BOX[s]
					state[constraints_fixers[j][0][0]][constraints_fixers[j][0][1]] = (s >> 4) % 2
					state[constraints_fixers[j][1][0]][constraints_fixers[j][1][1]] = (s >> 3) % 2
					state[constraints_fixers[j][2][0]][constraints_fixers[j][2][1]] = (s >> 2) % 2
					state[constraints_fixers[j][3][0]][constraints_fixers[j][3][1]] = (s >> 1) % 2
					state[constraints_fixers[j][4][0]][constraints_fixers[j][4][1]] = (s >> 0) % 2
		return state

	def check_satisfiability_forward(self,state,difference):
		state0 = copy.deepcopy(state)
		state1 = copy.deepcopy(state)
		expected_diff = difference 
		if int(self.state_num) == self.state_num:
			state0 = [XOR(state0[i],expected_diff[self.state_num][i]) for i in range(5)]
			for i in range(self.state_num+1,self.nr+1):
				state0 = self.ascon_one_round(state0,i)
				state1 = self.ascon_one_round(state1,i)
				actual_diff = [XOR(state0[j],state1[j]) for j in range(5)]
				if actual_diff != expected_diff[i]:
					# for q in actual_diff:
					# 	print(q)
					# print()
					# for q in expected_diff[i]:
					# 	print(q)
					# print()
					# print('1forward: not the same at round',i)
					return False
		else:
			state0 = [XOR(state0[i],self.linear_layer_inverse(expected_diff[int(self.state_num+0.5)])[i]) for i in range(5)]
			state0 = self.linear_layer(state0)
			state1 = self.linear_layer(state1)
			for i in range(int(self.state_num+1.5),self.nr+1):
				state0 = self.ascon_one_round(state0,i)
				state1 = self.ascon_one_round(state1,i)
				actual_diff = [XOR(state0[j],state1[j]) for j in range(5)]
				if actual_diff != expected_diff[i]:
					actual_diff = self.linear_layer_inverse(actual_diff)
					expected_diff[i] = self.linear_layer_inverse(expected_diff[i])
					# for q in actual_diff:
					# 	print(q)
					# print()
					# for q in expected_diff[i]:
					# 	print(q)
					# print()
					# print('2forward: not the same at round',i)
					return False
		return True

	def check_satisfiability_backward(self,state,difference):
		state0 = copy.deepcopy(state)
		state1 = copy.deepcopy(state)
		expected_diff = difference

		if int(self.state_num) == self.state_num:
			state0 = [XOR(state0[i],expected_diff[self.state_num][i]) for i in range(5)]
			for i in range(self.state_num,0,-1):
				state0 = self.inv_ascon_one_round(state0,i)
				state1 = self.inv_ascon_one_round(state1,i)
				actual_diff = [XOR(state0[j],state1[j]) for j in range(5)]
				if actual_diff != expected_diff[i]:
					# for q in actual_diff:
					# 	print(q)
					# print()
					# for q in expected_diff[i]:
					# 	print(q)
					# print()
					# print('1backward: not the same at round',i)
					return False
		else:
			state0 = self.linear_layer(state0);
			state1 = copy.deepcopy(state0);
			state0 = [XOR(state0[i],expected_diff[int(self.state_num+0.5)][i]) for i in range(5)]
			for i in range(int(self.state_num+0.5),0,-1):
				state0 = self.inv_ascon_one_round(state0,i)
				state1 = self.inv_ascon_one_round(state1,i)
				actual_diff = [XOR(state0[j],state1[j]) for j in range(5)]
				if actual_diff != expected_diff[i-1]:
					# for q in actual_diff:
					# 	print(q)
					# print()
					# for q in expected_diff[i-1]:
					# 	print(q)
					# print()
					# print('2backward: not the same at round',i)
					return False

		return True

	def verification(self,state,difference):
		if int(self.state_num) == self.state_num:
			state = self.inv_ascon_round_function(state,self.state_num)
		else:
			state = self.linear_layer(state)
			for i in range(int(self.state_num+0.5),0,-1):
				state = self.inv_ascon_one_round(state,i)
		state0 = copy.deepcopy(state)
		state1 = copy.deepcopy(state)
		state0 = [XOR(state0[i] , difference[0][i]) for i in range(5)]
		temp_state0 = copy.deepcopy(state0)
		temp_state1 = copy.deepcopy(state1)
		temp_state0 = self.inv_ascon_one_round(temp_state0,0)
		temp_state1 = self.inv_ascon_one_round(temp_state1,0)
		print()
		print('temp_state0',bitToString(temp_state0))
		print('temp_state1',bitToString(temp_state1))
		for i in range(len(difference)-1):
			state0 = self.ascon_one_round(state0,i+1)
			state1 = self.ascon_one_round(state1,i+1)
			diff = [XOR(state0[i],state1[i]) for i in range(5)]
			if diff != difference[i+1]:	
				return False
		return True


def stringToBin(difference_hex_string):
	difference = []
	for rd in difference_hex_string:
		round_state = []
		for row in rd:
			row_state = []
			for nibble in row:
				b = bin(int(nibble, 16))[2:].zfill(4)
				b = [int(b[i]) for i in range(len(b))]
				row_state.extend(b)
			round_state.append(row_state)
		difference.append(round_state)
	return difference

def bitToInt(state):
	state0 = [0] * 5
	for i in range(5):
		for j in range(64):
			state0[i] = state0[i] ^ (state[i][j] * (2**(63-j)))
	return state0

def IntTobit(state):
	state0 = [0] * 5
	for i in range(5):
		state0[i] = [int(bin(state[i])[2:].zfill(64)[j]) for j in range(64)] # converting from dec to binary array
	return state0

def XOR(s1,s2):
	s = [0] * 64
	for i in range(64):
		s[i] = s1[i] ^ s2[i]
	return s



def gauss_jordan(M):
	row = 0
	leading_row = 0
	leading_col = 0
	while row < M.shape[0]:
		flag = False
		for c in range(leading_col,M.shape[1]):
			for r in range(row,M.shape[0]):
				if M[r,c] == 1:
					leading_row = r
					leading_col = c
					flag = True
					break
			if flag: break
		if not flag: break
		M[[row,leading_row]] = M[[leading_row,row]]
		for r in range(row+1,M.shape[0]):
			if M[r,leading_col] == 1:
				for p in range(M.shape[1]):
					M[r,p] = M[r,p] ^ M[row,p]
		row += 1


	
	for i in reversed(range(M.shape[0])):
		if sum(M[i]) == 0: continue
		for j in range(M.shape[1]):
			if M[i,j] == 1:
				leading_col = j
				break
		for k in range(i):
			if M[k,leading_col] == 1:
				for p in range(M.shape[1]):
					M[k,p] = M[k,p] ^ M[i,p]
	return M

def simplify_equations(LHS,RHS):
	counter = 0
	new_LHS = []
	new_RHS = []
	while counter < len(LHS):
		i = 0
		rd = LHS[counter][0][0]
		col = LHS[counter][0][2]
		M = np.zeros((5,6),dtype=int)
		while counter < len(LHS) and LHS[counter][0][0] == rd and LHS[counter][0][2] == col:
			#print(LHS[counter],RHS[counter])
			M[i,LHS[counter][0][1]] = 1
			for j in range(len(RHS[counter])):
				if type(RHS[counter][j]) == tuple:
					M[i,RHS[counter][j][1]] = 1
				else:
					M[i,-1] = RHS[counter][j]
			i += 1
			counter += 1
		M = gauss_jordan(M)
		# sort M based on HW except last row
		for i in range(M.shape[0]):
			lowest_index = i
			for j in range(i+1,M.shape[0]):
				if sum(M[j,:-1]) < sum(M[lowest_index,:-1]) and sum(M[j,:-1]) > 0:
					lowest_index = j
			M[[i,lowest_index]] = M[[lowest_index,i]]
		# form back equations from the matrix M
		for r in reversed(range(M.shape[0])):
			if sum(M[r]) == 0: continue
			LHS_trigger = False
			new_LHS.append([])
			new_RHS.append([])
			for c in range(M.shape[1]-1):
				if M[r,c] == 1 and LHS_trigger == False:
					LHS_trigger = True
					new_LHS[-1].append((rd,c,col))
				elif M[r,c] == 1 and LHS_trigger == True:
					new_RHS[-1].append((rd,c,col))
			if M[r,-1] == 1:
				new_RHS[-1].append(1)
			elif M[r,-1] == 0 and new_RHS[-1] == []:
				new_RHS[-1].append(0)
	return new_LHS,new_RHS

def Spread2Rounds(difference,blackbox=False):
	def substitution_OR(s):
		ascon = ASCON()
		DDT = ascon.computeDDT().astype(int)
		s_out = [[0 for _ in range(65)] for _ in range(5)]
		# subsitution
		for c in range(64):
			s_in = (s[0][c] << 4) ^ (s[1][c] << 3) ^ (s[2][c] << 2) ^ (s[3][c] << 1) ^ (s[4][c] << 0)
			for j in range(32):
				if DDT[s_in,j] > 0:
					s_out[0][c] = s_out[0][c] | ((j >> (4-0)) % 2)
					s_out[1][c] = s_out[1][c] | ((j >> (4-1)) % 2)
					s_out[2][c] = s_out[2][c] | ((j >> (4-2)) % 2)
					s_out[3][c] = s_out[3][c] | ((j >> (4-3)) % 2)
					s_out[4][c] = s_out[4][c] | ((j >> (4-4)) % 2)
		return s_out
	def linear_layer_OR(s):
		ascon = ASCON()
		s_out = [[0 for _ in range(65)] for _ in range(5)]
		for i in range(64):
			for term in ascon.Z0: s_out[0][i] = s_out[0][i] | s[0][(i-term)%64]
			for term in ascon.Z1: s_out[1][i] = s_out[1][i] | s[1][(i-term)%64]
			for term in ascon.Z2: s_out[2][i] = s_out[2][i] | s[2][(i-term)%64]
			for term in ascon.Z3: s_out[3][i] = s_out[3][i] | s[3][(i-term)%64]
			for term in ascon.Z4: s_out[4][i] = s_out[4][i] | s[4][(i-term)%64]
		s = s_out.copy()
		return s
	def substitution_space(s):
		count = 1
		ascon = ASCON()
		DDT = ascon.computeDDT().astype(int)
		for c in range(64):
			active_bits = (s[0][c] << 4) ^ (s[1][c] << 3) ^ (s[2][c] << 2) ^ (s[3][c] << 1) ^ (s[4][c] << 0)
			tmp = np.zeros((32,),dtype=int)
			for i in range(1,32):
				for j in range(0,32):
					if i&active_bits == 0: continue
					tmp += DDT[i&active_bits,:]
			# print(active_bits,tmp.tolist(),tmp.tolist().count(0))
			count = count * tmp.tolist().count(0)
		# print('count:',count, end=' ')
		return math.log2(count)
	ascon = ASCON()
	Din,_ = ascon.computeSets(difference)
	s = copy.deepcopy(difference[-1])
	s = substitution_OR(s)
	s = linear_layer_OR(s)
	Dout_p = substitution_space(s)
	# print(Din,320-Dout_p)
	if blackbox == True:
		# print(320-Dout_p)
		cost = np.log2(np.sqrt(2**320/(2**(320-Dout_p))))
	else:
		cost = np.log2(max(min(np.sqrt(2**320/2**Din),np.sqrt(2**320/(2**(320-Dout_p)))),2**321/((2**Din)*(2**(320-Dout_p)))))
	return cost


def bitToString(state):
	state = bitToInt(state)
	for i in range(len(state)):
		state[i] = hex(state[i])[2:].zfill(16)
	return state
	


def main_nonblackbox(difference,state_num,filename=None):
	# initialization
	ascon = ASCON()
	print('probability of generic permutation',ascon.generic_prob(difference))
	equation_computation = ASCON_equation_computation()
	dependency_computation = ASCON_dependency_computation(len(difference)-1,state_num)

	LHS,RHS = equation_computation.find_equations(difference.copy())
	LHS,RHS = simplify_equations(LHS,RHS)
	# for printing of equations
	equations = equation_computation.make_string_equations(LHS,RHS)
	# equation_computation.print_equations(equations) # this prints out all the possible equations
	
	# this computes independent equations 
	constraints_wanted,constraints_bits,bits_used,constraints_fixers = dependency_computation.find_max_independent_constraints(LHS,RHS)
	# for i in range(len(constraints_wanted)):
	# 	print(constraints_wanted[i],constraints_bits[i],constraints_fixers[i])
	print('total number of possible constraints',len(equations))
	print('total number of independent constraints:',len(constraints_wanted))
	print('probability of limited birthday distinguisher:',len(equations)-len(constraints_wanted))
	print('probability of generic permutation (1,1)',ascon.generic_prob(difference))
	print('probability of generic permutation (1,2)',Spread2Rounds(difference))
	# print('probability of generic permutation (2,1)',Spread2RoundsBackwards(difference))
	max_tries = 1
	counter = 0
	tries = 0
	while tries < max_tries:
		diff_satisfiability = inverse_diff_satisfiability = False
		while diff_satisfiability == False or inverse_diff_satisfiability == False:
			counter += 1 
			print('\rNumber of tries so far:',math.log2(counter),end='',flush=True)
			state = dependency_computation.fix_constraints(constraints_wanted,constraints_bits,constraints_fixers)
			diff_satisfiability = dependency_computation.check_satisfiability_forward(state,difference.copy())
			if not diff_satisfiability: continue
			inverse_diff_satisfiability = dependency_computation.check_satisfiability_backward(state,difference.copy())
		if dependency_computation.verification(state,difference):
			# print(bitToString(state))
			pass
		else:	
			print('verification failed!')
			assert False
		tries += 1
	print()
	print('average number of tries:',math.log2(counter/max_tries))


def main_blackbox(difference,state_num):
	if state_num != 0:
		print('state_num must be 0!')
		return
	ascon = ASCON()
	equation_computation = ASCON_equation_computation()
	dependency_computation = ASCON_dependency_computation(len(difference)-1,state_num)
	LHS,RHS = equation_computation.find_equations(difference.copy())
	LHS,RHS = simplify_equations(LHS,RHS)
	print('probability of blackbox limited birthday distinguisher:',len(LHS))
	front_cost, back_cost = ascon.generic_prob_blackbox(difference) # front_cost: adding a free round in the front
	print('probability of generic permutation start from the front:',back_cost)
	print('probability of generic permutation start from the back:',front_cost)
	print('probability of generic permutation (1,2)',Spread2Rounds(difference,blackbox=True))
		

if __name__ == '__main__':
	# LB2
	# difference_hex_string = \
	# [['0000000000000000','0000000000000001','0000000000000001','0000000000000000','0000000000000000'],
	#  ['0000201000000001','0000000000000000','0000000000000000','0000000000000000','0000000000000000'],
	#  ['0000000004000101','2001209002000049','0000000000000000','0000000000000000','0000000000000000']
	# ]
	# state_num = 0.5
	# LB3
	# difference_hex_string = \
	# [['0000000000000000','0000000000000001','0000000000000001','0000000000000000','0000000000000000'],
	#  ['0000201000000001','0000000000000000','0000000000000000','0000000000000000','0000000000000000'],
	#  ['0000000004000101','2001209002000049','0000000000000000','0000000000000000','0000000000000000'],
	#  ['4020100004000180','0008000224000900','9481b45a4308006c','322d30d8b6488148','0000000000000000']
	# ]
	# state_num = 1.5
	# LB4
	# difference_hex_string = \
	# [['0000000000000000','0000000000000001','0000000000000001','0000000000000000','0000000000000000'],
	# ['0000201000000001','0000000000000000','0000000000000000','0000000000000000','0000000000000000'],
	# ['0000000004000101','2001209002000049','0000000000000000','0000000000000000','0000000000000000'],
	# ['0020100000000100','2009241226000948','9481b45a4308006c','322d30d8b6488148','1002000000080008'],
	# ['162e14c670b19a21','0012000210000d48','645f5698151c0c77','99b7ea6001186aa2','6648288901610300']]
	# state_num = 2.5
 	# LB4.2
	# difference_hex_string = \
	# [['fb8e401124ca8085','04318d0c40007a10','04318d0c40007a10','fb8c400120408005','fb8c400120408005'],
	#  ['4020001000000100','0020300000000181','0020300000000001','4000001004010000','0000000004010181'],
	#  ['2000000000000001','0000000000000000','0000000000000000','0000000000000000','0000000000000000'],
	#  ['2000241200000001','2000000002400008','0000000000000000','0000000000000000','0000000000000000'],
	#  ['0000040204800121','2401048202000041','108000000369000c','0204000002409128','0000000000000000']]
	# state_num = 0.5

	# LB5
	# difference_hex_string = \
	# [['0000000000020081','0000000000000000','0000000000000000','0000000000020081','0000000000020081'],
	# ['0000000000000000','0000000000000000','0000000000000000','2000800000020000','0000000000000000'],
	# ['60208402100a0000','0000000000000000','0000000000000000','0000000000000000','2040800000120440'],
	# ['61e8c00a141a0442','4740141a1058e64a','0000000000184000','0000000000000000','61e8c00a141a0442'],
	# ['644998a100440322','0000100800482400','44241d484669b184','e42585812e40b044','e5619ca12420a2a4'],
	# ['83d466293fa88565','0948c84107473492','579146a2e5018394','c3220c515630a665','5041813b7a143040']]
	# state_num = 3.5




	# for black-box
	# LB2
	# difference_hex_string = \
	# [['0000000000000000','0000000000000001','0000000000000001','0000000000000000','0000000000000000'],
	#  ['0000201000000001','0000000000000000','0000000000000000','0000000000000000','0000000000000000'],
	#  ['0000000004000101','2001209002000049','0000000000000000','0000000000000000','0000000000000000']
	# ]
	# state_num = 0
	# LB3
	# difference_hex_string = \
	# [['0000000000000000','0000000000000001','0000000000000001','0000000000000000','0000000000000000'],
	#  ['0000201000000001','0000000000000000','0000000000000000','0000000000000000','0000000000000000'],
	#  ['0000000004000101','2001209002000049','0000000000000000','0000000000000000','0000000000000000'],
	#  ['4020100004000180','0008000224000900','9481b45a4308006c','322d30d8b6488148','0000000000000000']
	# ]
	# state_num = 0
	# LB3.1
	# difference_hex_string = \
	# [['32a11104c9b008db','0000000000000001','0000000000000001','32a11104c9b008da','32a11104c9b008da'],
	#  ['0000201000000001','0000000000000000','0000000000000000','0000201000000001','0000000000000000'],
	#  ['0000000004000101','0000000000000000','0000000000000000','0000000000000000','0000000000000000'],
	#  ['4020301004000181','0008000226000909','0000000000000000','0000000000000000','0000000000000000']]
	# state_num = 0
	# LB4.1
	# difference_hex_string = \
	# [['0000000400000000','63b6c53b00766181','63b6c53b00766181','0000000400000000','0000000400000000'],
	# ['0000000000000000','0000000401020000','0000000000000000','0000000401020000','0000000000000000'],
	# ['0000000000000000','0000000000000000','0000000000000000','0000000400004001','0000000000000000'],
	# ['080420140000c041','0000000000000000','0000000000000000','0000000000000000','0000002408804081'],
	# ['98100d240cc44291','283420b1cc948e80','0000000000000000','0000000000000000','0000002408804081']]
	# state_num = 0
	
	# difference = stringToBin(difference_hex_string)

	main_blackbox(difference,state_num)
	# main_nonblackbox(difference,state_num)




