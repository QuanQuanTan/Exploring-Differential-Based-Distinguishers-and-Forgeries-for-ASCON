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

// each state contains 8 bits

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


void testBoomerangTrail()
{

	ASCON cypher = ASCON();
	int rot_value = 12;
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
	int max_correct = 100;
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
		cypher.roundFunction(p2,1);
		cypher.roundFunction(p3,1);
		cypher.roundFunction(p2,2);
		cypher.roundFunction(p3,2);
		cypher.roundFunction(p2,3);
		cypher.roundFunction(p3,3);
		cypher.roundFunction(p2,4);
		cypher.roundFunction(p3,4);


		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				p2[i][j] = p2[i][j] ^ lower_difference[i][j];
				p3[i][j] = p3[i][j] ^ lower_difference[i][j];
			}
		}
		cypher.invRoundFunction(p2,4);
		cypher.invRoundFunction(p3,4);
		cypher.invRoundFunction(p2,3);
		cypher.invRoundFunction(p3,3);
		cypher.invRoundFunction(p2,2);
		cypher.invRoundFunction(p3,2);
		cypher.invRoundFunction(p2,1);
		cypher.invRoundFunction(p3,1);

		bool flag = true;
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				if ((p2[i][j] ^ p3[i][j]) != upper_difference[i][j]) flag = false; 
			}
		}
		if (flag) correct++;
		if (count > pow(2,30)) 
		{
			max_correct = -1;
			break;
		}
	}
	cout << "average: " << (count+0.0)/max_correct << endl;
}


int main()
{
	rand_generator.seed(time(0));
	testBoomerangTrail();
	return 0;
}
