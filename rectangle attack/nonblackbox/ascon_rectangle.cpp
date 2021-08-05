#include <iostream>
#include <bitset>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <memory.h>
#include <sstream>
#include <iomanip>

using namespace std;

#define WORD_SIZE 32
mt19937 rand_generator;

uint8_t states[1000][4][4][5][8] = {0}; // state_index, then round_index
uint8_t states_sbox[1000][4][4][5][8] = {0}; // state_index, then round_index
int states_length = 0;
int states_num[1000] = {0};

class ASCON;
void ROR(uint8_t ROW[8], uint32_t x);

class ASCON {
	public:
		uint8_t S_BOX[WORD_SIZE] = {0x4,0xb,0x1f,0x14,0x1a,0x15,0x9,0x2,0x1b,0x5,0x8,0x12,0x1d,0x3,0x6,0x1c, \
         								  0x1e,0x13,0x7,0xe,0x0,0xd,0x11,0x18,0x10,0xc,0x1,0x19,0x16,0xa,0xf,0x17};
        uint8_t INV_S_BOX[WORD_SIZE] = {20,26,7,13,0,9,14,18,10,6,29,1,25,21,19,30,24,22,11,17,3,5,28,31,23,27,4,8,15,12,16,2};

		uint32_t SHIFT_VALUES[5][3] = {{0,19,28},{0,61,39},{0,1,6},{0,10,17},{0,7,41}};
		uint32_t INV_SHIFT_VALUES[5][35] ={{0,3,6,9,11,12,14,15,17,18,19,21,22,24,25,27,30,33,36,38,39,41,42,44,45,47,50,53,57,60,63,0,0,0,0},
												{0,1,2,3,4,8,11,13,14,16,19,21,23,24,25,27,28,29,30,35,39,43,44,45,47,48,51,53,54,55,57,60,61,0,0},
												{0,2,4,6,7,10,11,13,14,15,17,18,20,23,26,27,28,32,34,35,36,37,40,42,46,47,52,58,59,60,61,62,63,0,0},
												{1,2,4,6,7,9,12,17,18,21,22,23,24,26,27,28,29,31,32,33,35,36,37,40,42,44,47,48,49,53,58,61,63,0,0},
												{0,1,2,3,4,5,9,10,11,13,16,20,21,22,24,25,28,29,30,31,35,36,40,41,44,45,46,47,48,50,53,55,60,61,63}};
		uint32_t INV_SHIFT_VALUES_LENGTH[5] = {31,33,33,33,35};

		uint32_t CONSTANT[12] = {0xf0,0xe1,0xd2,0xc3,0xb4,0xa5,0x96,0x87,0x78,0x69,0x5a,0x4b};

		void addConstants(uint8_t state[5][8], uint32_t round)
		{
			state[2][7] ^= CONSTANT[round];
		}

		void linearLayer(uint8_t state[5][8])
		{
			uint8_t tempState[5][8] = {0};
			uint8_t tempROW[8];
			for (int row_index = 0; row_index < 5; row_index++)
			{
				for (int shift_index = 0; shift_index < 3; shift_index++)
				{
					for (int col_index = 0; col_index < 8; col_index++) tempROW[col_index] = state[row_index][col_index];
					ROR(tempROW,SHIFT_VALUES[row_index][shift_index]);
					for (int col_index = 0; col_index < 8; col_index++) tempState[row_index][col_index] ^= tempROW[col_index];
				}

				for (int col_index = 0; col_index < 8; col_index++) state[row_index][col_index] = tempState[row_index][col_index];
			}
		}

		void linearLayerInverse(uint8_t state[5][8])
		{
			uint8_t tempState[5][8] = {0};
			uint8_t tempROW[8];
			for (int row_index = 0; row_index < 5; row_index++)
			{
				for (int shift_index = 0; shift_index < INV_SHIFT_VALUES_LENGTH[row_index]; shift_index++)
				{
					for (int col_index = 0; col_index < 8; col_index++) tempROW[col_index] = state[row_index][col_index];
					ROR(tempROW,INV_SHIFT_VALUES[row_index][shift_index]);
					for (int col_index = 0; col_index < 8; col_index++) tempState[row_index][col_index] ^= tempROW[col_index];
				}
				
				for (int col_index = 0; col_index < 8; col_index++) state[row_index][col_index] = tempState[row_index][col_index];
			}
		}

		void substitution(uint8_t state[5][8])
		{
			uint8_t tempVar;
			for (int i = 0; i < 64; i++)
			{
				tempVar = (((state[0][i/8] >> (7-i%8)) % 2) << 4) ^ 
						  (((state[1][i/8] >> (7-i%8)) % 2) << 3) ^ 
						  (((state[2][i/8] >> (7-i%8)) % 2) << 2) ^ 
						  (((state[3][i/8] >> (7-i%8)) % 2) << 1) ^ 
						   ((state[4][i/8] >> (7-i%8)) % 2);

				tempVar = S_BOX[tempVar];

				state[0][i/8] = state[0][i/8] & (0xff - (1 << (7-i%8)));
				state[1][i/8] = state[1][i/8] & (0xff - (1 << (7-i%8)));
				state[2][i/8] = state[2][i/8] & (0xff - (1 << (7-i%8)));
				state[3][i/8] = state[3][i/8] & (0xff - (1 << (7-i%8)));
				state[4][i/8] = state[4][i/8] & (0xff - (1 << (7-i%8)));


				state[0][i/8] ^= (((tempVar >> 4) % 2) << (7-i%8));
				state[1][i/8] ^= (((tempVar >> 3) % 2) << (7-i%8));
				state[2][i/8] ^= (((tempVar >> 2) % 2) << (7-i%8));
				state[3][i/8] ^= (((tempVar >> 1) % 2) << (7-i%8));
				state[4][i/8] ^= ((tempVar % 2) << (7-i%8));
			}
		}

		void substitutionInverse(uint8_t state[5][8])
		{
			uint8_t tempVar;
			for (int i = 0; i < 64; i++)
			{
				tempVar = (((state[0][i/8] >> (7-i%8)) % 2) << 4) ^ 
						  (((state[1][i/8] >> (7-i%8)) % 2) << 3) ^ 
						  (((state[2][i/8] >> (7-i%8)) % 2) << 2) ^ 
						  (((state[3][i/8] >> (7-i%8)) % 2) << 1) ^ 
						   ((state[4][i/8] >> (7-i%8)) % 2);

				tempVar = INV_S_BOX[tempVar];

				state[0][i/8] = state[0][i/8] & (0xff - (1 << (7-i%8)));
				state[1][i/8] = state[1][i/8] & (0xff - (1 << (7-i%8)));
				state[2][i/8] = state[2][i/8] & (0xff - (1 << (7-i%8)));
				state[3][i/8] = state[3][i/8] & (0xff - (1 << (7-i%8)));
				state[4][i/8] = state[4][i/8] & (0xff - (1 << (7-i%8)));


				state[0][i/8] ^= (((tempVar >> 4) % 2) << (7-i%8));
				state[1][i/8] ^= (((tempVar >> 3) % 2) << (7-i%8));
				state[2][i/8] ^= (((tempVar >> 2) % 2) << (7-i%8));
				state[3][i/8] ^= (((tempVar >> 1) % 2) << (7-i%8));
				state[4][i/8] ^= ((tempVar % 2) << (7-i%8));
			}
		}

		void roundFunction(uint8_t state[5][8], uint32_t round_num)
		{
			addConstants(state,round_num); 
			substitution(state);
			linearLayer(state);
		}

		void encrypt(uint8_t state[5][8], uint32_t nr)
		{
			for (uint32_t i = 0; i < nr; i++) roundFunction(state,i);
		}

		void invRoundFunction(uint8_t state[5][8], uint32_t round_num)
		{
			linearLayerInverse(state);
			substitutionInverse(state);
			addConstants(state,round_num);
		}

		void decrypt(uint8_t state[5][8], uint32_t nr)
		{
			for (uint32_t i = nr-1; i >= 0; i--) invRoundFunction(state,i);
		}

};


bool containTrail(uint8_t tempState[4][4][5][8], uint8_t tempState_sbox[4][4][5][8])
{
	for (int i = 0; i < states_length; i++)
	{
		bool flag = true;
		// upper round difference
		for (int p = 0; p < 2; p++)
		{
			for (int nr = 0; nr < 2; nr++)
			{
				for (int j = 0; j < 5; j++)
				{
					for (int k = 0; k < 8; k++)
					{
						if (tempState[p][nr][j][k] != states[i][p][nr][j][k]) flag = false;
						if (tempState_sbox[p][nr][j][k] != states_sbox[i][p][nr][j][k]) flag = false;
					}
				}
			}
		}
		// lower round difference
		for (int p = 2; p < 4; p++)
		{
			for (int nr = 2; nr < 4; nr++)
			{
				for (int j = 0; j < 5; j++)
				{
					for (int k = 0; k < 8; k++)
					{
						if (tempState[p][nr][j][k] != states[i][p][nr][j][k]) flag = false;
						if (tempState_sbox[p][nr][j][k] != states_sbox[i][p][nr][j][k]) flag = false;
					}
				}
			}
		}
		if (flag)
		{
			states_num[i]++; 
			return true;
		}
	}
	return false;
}

void inputState(uint8_t tempState[4][4][5][8], uint8_t p0[5][8], uint8_t p1[5][8], int state_index, int round_index)
{
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			tempState[state_index][round_index][i][j] =  (p0[i][j] ^ p1[i][j]);
		}
	}
}
void saveTrail(uint8_t p0[5][8], uint8_t upper_difference[5][8], uint8_t lower_difference[5][8])
{
	uint8_t tempState[4][4][5][8], tempState_sbox[4][4][5][8];
	ASCON cypher = ASCON();
	uint8_t p1[5][8], p2[5][8], p3[5][8];
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 8; j++) p1[i][j] = p0[i][j] ^ upper_difference[i][j];
	}
	for (int i = 0; i < 4; i++)
	{
		cypher.roundFunction(p0,i);
		cypher.roundFunction(p1,i);
	}
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 8; j++) 
		{
			p2[i][j] = p0[i][j] ^ lower_difference[i][j];
			p3[i][j] = p1[i][j] ^ lower_difference[i][j];
		}
	}
	
	for (int i = 3; i >= 0; i--)
	{
		cypher.linearLayerInverse(p0);
		cypher.linearLayerInverse(p1);
		cypher.linearLayerInverse(p2);
		cypher.linearLayerInverse(p3);
		inputState(tempState_sbox,p0,p1,0,i);
		inputState(tempState_sbox,p2,p3,1,i);
		inputState(tempState_sbox,p0,p2,2,i);
		inputState(tempState_sbox,p1,p3,3,i);
		cypher.substitutionInverse(p0);
		cypher.substitutionInverse(p1);
		cypher.substitutionInverse(p2);
		cypher.substitutionInverse(p3);
		inputState(tempState,p0,p1,0,i);
		inputState(tempState,p2,p3,1,i);
		inputState(tempState,p0,p2,2,i);
		inputState(tempState,p1,p3,3,i);
		cypher.addConstants(p0,i);
		cypher.addConstants(p1,i);
		cypher.addConstants(p2,i);
		cypher.addConstants(p3,i);
	}
	if (containTrail(tempState,tempState_sbox)) return;
	for (int m = 0; m < 4; m++)
	{
		for (int n = 0; n < 4; n++)
		{
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 8; j++)
				{
					states[states_length][m][n][i][j] = tempState[m][n][i][j];
					states_sbox[states_length][m][n][i][j] = tempState_sbox[m][n][i][j];
				}
			}
		}
	}
	states_num[states_length]++;
	states_length++;
}

void printBoomerang(int index)
{
	cout << "============================================================================" << endl;
	cout << "upper difference" << endl;
	for (int nr = 0; nr < 4; nr++)
	{
		// for row 0
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) cout << hex << ((int(states[index][0][nr][0][j]) >> (7-k)) % 2);
			cout << "|";
		}
		cout << " ";
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) cout << hex << ((int(states[index][1][nr][0][j]) >> (7-k)) % 2);
			cout << "|";
		}
		cout << endl;
		// for row 1-4
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) 
			{
				uint8_t b = (((int(states[index][0][nr][1][j]) >> (7-k)) % 2) << 3) ^ \
							(((int(states[index][0][nr][2][j]) >> (7-k)) % 2) << 2) ^ \
							(((int(states[index][0][nr][3][j]) >> (7-k)) % 2) << 1) ^ \
							(((int(states[index][0][nr][4][j]) >> (7-k)) % 2) << 0);
				cout << hex << int(b);
			}
			cout << "|";
		}
		cout << " ";
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) 
			{
				uint8_t b = (((int(states[index][1][nr][1][j]) >> (7-k)) % 2) << 3) ^ \
							(((int(states[index][1][nr][2][j]) >> (7-k)) % 2) << 2) ^ \
							(((int(states[index][1][nr][3][j]) >> (7-k)) % 2) << 1) ^ \
							(((int(states[index][1][nr][4][j]) >> (7-k)) % 2) << 0);
				cout << hex << int(b);
			}
			cout << "|";
		}
		cout << endl;
		cout << "--------------------------------SUBSTITUTION-------------------------------------------------------SUBSTITUTION----------------------------------" << endl;
		// for row 0
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) cout << hex << ((int(states_sbox[index][0][nr][0][j]) >> (7-k)) % 2);
			cout << "|";
		}
		cout << " ";
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) cout << hex << ((int(states_sbox[index][1][nr][0][j]) >> (7-k)) % 2);
			cout << "|";
		}
		cout << endl;
		// for row 1-4
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) 
			{
				uint8_t b = (((int(states_sbox[index][0][nr][1][j]) >> (7-k)) % 2) << 3) ^ \
							(((int(states_sbox[index][0][nr][2][j]) >> (7-k)) % 2) << 2) ^ \
							(((int(states_sbox[index][0][nr][3][j]) >> (7-k)) % 2) << 1) ^ \
							(((int(states_sbox[index][0][nr][4][j]) >> (7-k)) % 2) << 0);
				cout << hex << int(b);
			}
			cout << "|";
		}
		cout << " ";
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) 
			{
				uint8_t b = (((int(states_sbox[index][1][nr][1][j]) >> (7-k)) % 2) << 3) ^ \
							(((int(states_sbox[index][1][nr][2][j]) >> (7-k)) % 2) << 2) ^ \
							(((int(states_sbox[index][1][nr][3][j]) >> (7-k)) % 2) << 1) ^ \
							(((int(states_sbox[index][1][nr][4][j]) >> (7-k)) % 2) << 0);
				cout << hex << int(b);
			}
			cout << "|";
		}
		cout << endl;
		cout << endl;
		cout << "--------------------------------NEXT ROUND----------------------------------------------------------NEXT ROUND-----------------------------------" << endl;
		cout << endl;
	}


	cout << "lower difference" << endl;
	for (int nr = 0; nr < 4; nr++)
	{
		// for row 0
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) cout << hex << ((int(states[index][2][nr][0][j]) >> (7-k)) % 2);
			cout << "|";
		}
		cout << " ";
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) cout << hex << ((int(states[index][3][nr][0][j]) >> (7-k)) % 2);
			cout << "|";
		}
		cout << endl;
		// for row 1-4
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) 
			{
				uint8_t b = (((int(states[index][2][nr][1][j]) >> (7-k)) % 2) << 3) ^ \
							(((int(states[index][2][nr][2][j]) >> (7-k)) % 2) << 2) ^ \
							(((int(states[index][2][nr][3][j]) >> (7-k)) % 2) << 1) ^ \
							(((int(states[index][2][nr][4][j]) >> (7-k)) % 2) << 0);
				cout << hex << int(b);
			}
			cout << "|";
		}
		cout << " ";
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) 
			{
				uint8_t b = (((int(states[index][3][nr][1][j]) >> (7-k)) % 2) << 3) ^ \
							(((int(states[index][3][nr][2][j]) >> (7-k)) % 2) << 2) ^ \
							(((int(states[index][3][nr][3][j]) >> (7-k)) % 2) << 1) ^ \
							(((int(states[index][3][nr][4][j]) >> (7-k)) % 2) << 0);
				cout << hex << int(b);
			}
			cout << "|";
		}
		cout << endl;
		cout << "--------------------------------SUBSTITUTION-------------------------------------------------------SUBSTITUTION----------------------------------" << endl;
		// for row 0
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) cout << hex << ((int(states_sbox[index][2][nr][0][j]) >> (7-k)) % 2);
			cout << "|";
		}
		cout << " ";
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) cout << hex << ((int(states_sbox[index][3][nr][0][j]) >> (7-k)) % 2);
			cout << "|";
		}
		cout << endl;
		// for row 1-4
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) 
			{
				uint8_t b = (((int(states_sbox[index][2][nr][1][j]) >> (7-k)) % 2) << 3) ^ \
							(((int(states_sbox[index][2][nr][2][j]) >> (7-k)) % 2) << 2) ^ \
							(((int(states_sbox[index][2][nr][3][j]) >> (7-k)) % 2) << 1) ^ \
							(((int(states_sbox[index][2][nr][4][j]) >> (7-k)) % 2) << 0);
				cout << hex << int(b);
			}
			cout << "|";
		}
		cout << " ";
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 8; k++) 
			{
				uint8_t b = (((int(states_sbox[index][3][nr][1][j]) >> (7-k)) % 2) << 3) ^ \
							(((int(states_sbox[index][3][nr][2][j]) >> (7-k)) % 2) << 2) ^ \
							(((int(states_sbox[index][3][nr][3][j]) >> (7-k)) % 2) << 1) ^ \
							(((int(states_sbox[index][3][nr][4][j]) >> (7-k)) % 2) << 0);
				cout << hex << int(b);
			}
			cout << "|";
		}
		cout << endl;
		cout << endl;
		cout << "--------------------------------NEXT ROUND----------------------------------------------------------NEXT ROUND-----------------------------------" << endl;
		cout << endl;
	}
	cout << "============================================================================" << endl;
}

void ROR(uint8_t ROW[8], uint32_t x)
{
	uint8_t tempROW[8];
	if (x % 8 == 0)
	{
		for (int i = 0; i < 8; i++) tempROW[i] = ROW[(i-(x/8))%8];
	}
	else
	{
		for (int i = 0; i < 8; i++)
		{
			tempROW[i] = (ROW[(i-x/8-1)%8] << (8-x%8)) ^ (ROW[(i-x/8)%8] >> (x%8));
		}
	}
	for (int i = 0; i < 8; i++) ROW[i] = tempROW[i];
	
}




void getPlainText(uint8_t plaintext[5][8])
{
	uniform_int_distribution<uint8_t> rand_distribution(0,255);
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			plaintext[i][j] = rand_distribution(rand_generator);
		}
	}
}

void fixBits(uint8_t p[5][8], int nr)
{
	ASCON cypher = ASCON();
	if (nr == 4)
	{
		p[1][2] = (p[1][2] & 0b11011111);
		p[3][2] = (p[3][2] & 0b11011111) ^ (p[4][2] & 0b00100000) ^ (0b00100000);
		p[1][3] = (p[1][3] & 0b11101111);
		p[3][3] = (p[3][3] & 0b11101111) ^ (p[4][3] & 0b00010000) ^ (0b00010000);
		p[1][7] = (p[1][7] & 0b11111110);
		p[3][7] = (p[3][7] & 0b11111110) ^ (p[4][7] & 0b00000001) ^ (0b00000001);
	}
	if (nr == 5)
	{
		p[2][0] = p[2][0] & 0b11011111; // 2, 2 =  0
		p[2][1] = p[2][1] & 0b11111110; // 2, 15 = 0
		p[2][2] = p[2][2] & 0b11011111; // 2, 18 = 0
		p[2][3] = p[2][3] & 0b01111111; // 2, 24 = 0
		p[2][3] = p[2][3] & 0b11101111; // 2, 27 = 0
		p[1][4] = p[1][4] & 0b11111011; // 1, 37 = 0
		p[2][4] = p[2][4] & 0b11111101; // 2, 38 = 0
		p[1][6] = p[1][6] & 0b11111110; // 1, 55 = 0
		p[2][7] = p[2][7] & 0b10111111; // 2, 57 = 0
		p[2][7] = p[2][7] & 0b11110111; // 2, 60 = 0
		
		p[3][0] = p[3][0] | 0b00100000; // 3, 2 =  1
		p[3][1] = p[3][1] | 0b00000001; // 3, 15 = 1
		p[3][2] = p[3][2] | 0b00100000; // 3, 18 = 1
		p[3][3] = p[3][3] | 0b10000000; // 3, 24 = 1
		p[3][3] = p[3][3] | 0b00010000; // 3, 27 = 1
		p[3][4] = p[3][4] | 0b00000010; // 3, 38 = 1
		p[3][7] = p[3][7] | 0b01000000; // 3, 57 = 1
		p[3][7] = p[3][7] | 0b00001000; // 3, 60 = 1
		p[2][7] = p[2][7] | 0b00000001; // 2, 63 = 1
		p[3][7] = p[3][7] | 0b00000001; // 3, 63 = 1
		p[4][7] = p[4][7] | 0b00000001; // 4, 63 = 1

		p[4][0] = (p[4][0] & 0b11011111) ^ (p[0][0] & 0b00100000) ^ (0b00100000); // 4, 2 =  0, 2 + 1
		p[4][1] = (p[4][1] & 0b11111110) ^ (p[0][1] & 0b00000001) ^ (0b00000001); // 4, 15 = 0, 15 + 1
		p[4][2] = (p[4][2] & 0b11011111) ^ (p[0][2] & 0b00100000) ^ (0b00100000); // 4, 18 = 0, 18 + 1
		p[4][3] = (p[4][3] & 0b01111111) ^ (p[0][3] & 0b10000000) ^ (0b10000000); // 4, 24 = 0, 24 + 1
		p[4][3] = (p[4][3] & 0b11101111) ^ (p[0][3] & 0b00010000) ^ (0b00010000); // 4, 27 = 0, 27 + 1
		p[4][4] = (p[4][4] & 0b11111101) ^ (p[0][4] & 0b00000010) ^ (0b00000010); // 4, 38 = 0, 38 + 1
		p[4][7] = (p[4][7] & 0b10111111) ^ (p[0][7] & 0b01000000) ^ (0b01000000); // 4, 57 = 0, 57 + 1
		p[4][7] = (p[4][7] & 0b11110111) ^ (p[0][7] & 0b00001000) ^ (0b00001000); // 4, 60 = 0, 60 + 1
		p[4][6] = (p[4][6] & 0b11111110) ^ (p[3][6] & 0b00000001) ^ (0b00000001); // 4, 55 = 3, 55 + 1
		p[4][4] = (p[4][4] & 0b11111011) ^ (p[3][4] & 0b00000100) ^ (0b00000100); // 4, 37 = 3, 37 + 1
		p[0][7] = (p[0][7] & 0b11111110) ^ (p[1][7] & 0b00000001) ^ (0b00000001); // 0, 63 = 1, 63 + 1


		cypher.addConstants(p,4);
		// backwards 18 control 5
		// backwards 27 control 3
		// backwards 63 control 10
		// I need all these to XOR to 1
		int index_18[34] = {18,17,16,15,14,13,9,8,7,2,62,61,60,58,57,54,53,52,51,47,46,42,41,38,37,36,35,34,32,29,27,22,21,19};
		int index_27[34] = {27,26,25,24,23,22,18,17,16,14,11,7,6,5,2,63,62,61,60,56,55,51,50,47,46,45,44,43,41,38,36,31,30,28};
		int index_63[34] = {63,62,61,60,59,58,54,53,52,50,47,43,42,41,39,38,35,34,33,32,28,27,23,22,19,18,17,16,15,13,8,3,2,0};
		int bit_18 = 1;
		int bit_27 = 1;
		int bit_63 = 1;

		for (int i = 0; i < 34; i++) bit_18 ^= (p[4][index_18[i]/8] >> (7-(index_18[i]%8))) % 2;
		p[4][0] = (p[4][0] & 0b11111011) ^ (bit_18 << 2);
		for (int i = 0; i < 34; i++) bit_27 ^= (p[4][index_27[i]/8] >> (7-(index_27[i]%8))) % 2;
		p[4][0] = (p[4][0] & 0b11101111) ^ (bit_27 << 4);
		for (int i = 0; i < 34; i++) bit_63 ^= (p[4][index_63[i]/8] >> (7-(index_63[i]%8))) % 2;
		p[4][1] = (p[4][1] & 0b11011111) ^ (bit_63 << 5);
	}
}

void testBoomerangTrailNonBlackBox_4rounds()
{
	// this boomerangTrail is for 5 rounds trail starting at the and of 4 rounds

	ASCON cypher = ASCON();
	int rot_value = 31;
	// 2 --> 3
	// (0-1 -- 0-1)-2
	string upper_difference_string[5] = {"0000000000000000","0000000000000001","0000000000000001","0000000000000000","0000000000000000"};
	string lower_difference_string[5] = {"0000000004000101","2001209002000049","0000000000000000","0000000000000000","0000000000000000"};
	string four_round_difference_string[5] = {"0000201000000001","0000000000000000","0000000000000000","0000000000000000","0000000000000000"};
	uint8_t upper_difference[5][8];
	uint8_t lower_difference[5][8];
	uint8_t final_difference[5][8];
	uint8_t four_round_difference[5][8];
	for (uint32_t j = 0; j < 5; j++)
	{
		for (uint32_t k = 0; k < 8; k++)
		{
			upper_difference[j][k] = std::stoul(upper_difference_string[j].substr(2*k,2), nullptr, 16);
			lower_difference[j][k] = std::stoul(lower_difference_string[j].substr(2*k,2), nullptr, 16);
			four_round_difference[j][k] = std::stoul(four_round_difference_string[j].substr(2*k,2), nullptr, 16);
		}
		ROR(upper_difference[j],64-rot_value);
	}
	uint8_t p0[5][8], p1[5][8], p2[5][8], p3[5][8];
	uint64_t count = 0;
	int correct = 0;
	int max_correct = 1000;
	while (correct < max_correct)
	{
		count++;
		// generate plaintexts
		getPlainText(p0);
		fixBits(p0,4);
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				p1[i][j] = p0[i][j] ^ four_round_difference[i][j]; 
				p2[i][j] = p0[i][j];
				p3[i][j] = p1[i][j];
			}
		}
		bool flag = true;
		// forward direction
		// for (int i = 0; i < 5; i++)
		// {
		// 	for (int j = 0; j < 8; j++)
		// 	{
		// 		for (int k = 0; k < 8; k++)
		// 		{
		// 			cout << ((int(p0[i][j]) >> (7-k)) % 2);
		// 		}
		// 		cout << "|";
		// 	}
		// 	cout << endl;
		// }
		// cout << endl;
		cypher.substitution(p0);
		cypher.substitution(p1);
		cypher.linearLayer(p0);
		cypher.linearLayer(p1);
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				if ((p0[i][j] ^ p1[i][j]) != lower_difference[i][j]) flag = false; 
			}
		}
		if (flag == false) exit(0);
		// reverse direction
		cypher.addConstants(p2,3);
		cypher.addConstants(p3,3);
		for (int i = 2; i >= 0; i--)
		{
			cypher.invRoundFunction(p2,i);
			cypher.invRoundFunction(p3,i);
		}
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				p2[i][j] ^= upper_difference[i][j];
				p3[i][j] ^= upper_difference[i][j];
			}
		}
		for (int i = 0; i < 4; i++)
		{
			cypher.roundFunction(p2,i);
			cypher.roundFunction(p3,i);
		}
		
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				if ((p2[i][j] ^ p3[i][j]) != lower_difference[i][j]) flag = false; 
			}
		}
		if (flag) 
		{
			correct++;
			cout << "average so far: " << (count+0.0)/correct << endl;
		}
		if (count > pow(2,20)) 
		{
			max_correct = -1;
			break;
		}
	}
	cout << "average: " << (count+0.0)/max_correct << endl;
}


void testBoomerangTrailNonBlackBox_5rounds()
{
	// this boomerangTrail is for 5 rounds trail starting at the and of 4 rounds

	ASCON cypher = ASCON();
	int rot_value = 31;
	// 2 --> 3
	// (0-1 -- 0-1)-2
	string upper_difference_string[5] = {"0000000000000000","0000000000000001","0000000000000001","0000000000000000","0000000000000000"};
	string lower_difference_string[5] = {"0000000004000101","2001209002000049","0000000000000000","0000000000000000","0000000000000000"};
	string final_difference_string[5] = {"4020100004000180","0008000224000900","9481b45a4308006c","322d30d8b6488148","0000000000000000"};

	uint8_t upper_difference[5][8];
	uint8_t lower_difference[5][8];
	uint8_t final_difference[5][8];
	for (uint32_t j = 0; j < 5; j++)
	{
		for (uint32_t k = 0; k < 8; k++)
		{
			upper_difference[j][k] = std::stoul(upper_difference_string[j].substr(2*k,2), nullptr, 16);
			lower_difference[j][k] = std::stoul(lower_difference_string[j].substr(2*k,2), nullptr, 16);
			final_difference[j][k] = std::stoul(final_difference_string[j].substr(2*k,2), nullptr, 16);
		}
		ROR(upper_difference[j],64-rot_value);
	}
	uint8_t p0[5][8], p1[5][8], p2[5][8], p3[5][8];
	uint64_t count = 0;
	int correct = 0;
	int max_correct = 1000;
	while (correct < max_correct)
	{
		count++;
		// generate plaintexts
		getPlainText(p0);
		fixBits(p0,5);
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				p1[i][j] = p0[i][j] ^ lower_difference[i][j]; 
				p2[i][j] = p0[i][j];
				p3[i][j] = p1[i][j];
			}
		}
		cypher.addConstants(p0,4); cypher.addConstants(p1,4);
		cypher.substitution(p0); cypher.linearLayer(p0);
		cypher.substitution(p1); cypher.linearLayer(p1);
		bool sanity_flag = true;
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				if ((p0[i][j]^p1[i][j]) != final_difference[i][j]) sanity_flag = false;
			}
		}
		if (sanity_flag == false) 
		{
			cout << "Error!" << endl; 
			exit(0);
		}			
		// encryption
		for (int i = 3; i >= 0; i--)
		{
			cypher.invRoundFunction(p2,i);
			cypher.invRoundFunction(p3,i);
		}
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				p2[i][j] = p2[i][j] ^ upper_difference[i][j];
				p3[i][j] = p3[i][j] ^ upper_difference[i][j];
			}
		}
		for (int i = 0; i < 4; i++)
		{
			cypher.roundFunction(p2,i);
			cypher.roundFunction(p3,i);
		}
		bool flag = true;
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				if ((p2[i][j] ^ p3[i][j]) != lower_difference[i][j]) flag = false; 
			}
		}
		if (flag) 
		{
			correct++;
			cout << "average so far: " << (count+0.0)/correct << endl;
		}
		if (count > pow(2,30)) 
		{
			max_correct = -1;
			break;
		}
	}
	cout << "average: " << (count+0.0)/max_correct << endl;
}


void testBoomerangTrail()
{

	ASCON cypher = ASCON();
	int rot_value = 31;
	// 2 --> 3
	// (0-1 -- 0-1)-2
	string upper_difference_string[5] = {"0000000000000000","0000000000000001","0000000000000001","0000000000000000","0000000000000000"};
	string lower_difference_string[5] = {"0000000004000101","2001209002000049","0000000000000000","0000000000000000","0000000000000000"};

	uint8_t upper_difference[5][8];
	uint8_t lower_difference[5][8];
	for (uint32_t j = 0; j < 5; j++)
	{
		for (uint32_t k = 0; k < 8; k++)
		{
			upper_difference[j][k] = std::stoul(upper_difference_string[j].substr(2*k,2), nullptr, 16);
			lower_difference[j][k] = std::stoul(lower_difference_string[j].substr(2*k,2), nullptr, 16);
		}
		ROR(upper_difference[j],64-rot_value);
	}
	uint8_t p0[5][8], p1[5][8], p2[5][8], p3[5][8];
	uint64_t count = 0;
	int correct = 0;
	int max_correct = 10;
	while (correct < max_correct)
	{
		count++;
		// generate plaintexts
		getPlainText(p0);
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				p1[i][j] = p0[i][j] ^ upper_difference[i][j]; 
				p2[i][j] = p0[i][j];
				p3[i][j] = p1[i][j];
			}
		}
		// encryption
		// 1
		cypher.roundFunction(p2,0);
		cypher.roundFunction(p3,0);
		cypher.roundFunction(p2,1);
		cypher.roundFunction(p3,1);
		cypher.roundFunction(p2,2);
		cypher.roundFunction(p3,2);
		cypher.roundFunction(p2,3);
		cypher.roundFunction(p3,3);


		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				p2[i][j] = p2[i][j] ^ lower_difference[i][j];
				p3[i][j] = p3[i][j] ^ lower_difference[i][j];
			}
		}
		cypher.invRoundFunction(p2,3);
		cypher.invRoundFunction(p3,3);
		cypher.invRoundFunction(p2,2);
		cypher.invRoundFunction(p3,2);
		cypher.invRoundFunction(p2,1);
		cypher.invRoundFunction(p3,1);
		cypher.invRoundFunction(p2,0);
		cypher.invRoundFunction(p3,0);

		bool flag = true;
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				if ((p2[i][j] ^ p3[i][j]) != upper_difference[i][j]) flag = false; 
			}
		}
		if (flag)
		{
			// saveTrail(p0,upper_difference,lower_difference);
			correct++;
			cout << "average so far: " << (count+0.0)/correct << endl;
		}
		if (count > pow(2,30)) 
		{
			max_correct = -1;
			break;
		}
	}
	cout << "average: " << (count+0.0)/max_correct << endl;
	cout << "trails spread: ";
	for (int i = 0; i < states_length; i++) cout << states_num[i] << " ";
	cout << "trails: ";
	for (int i = 0; i < states_length; i++) 
	{
		printBoomerang(i);
		cout << endl;
	}
}


int main()
{
	rand_generator.seed(time(0));
	testBoomerangTrailNonBlackBox_4rounds();
	// testBoomerangTrailNonBlackBox_5rounds();
	return 0;
}
