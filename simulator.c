#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define MAKE 20
#define R 10000

int policy[3][101][101];
int prob1[21] = {0,0,0,0,0,0,1,1,1,1,1,2,2,2,2,3,3,3,4,4,5};
int prob2[21] = {0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5};

float simulate(int i1_init, int i2_init, int s_init)
{
    double total_costs = 0.0;
    for(int repetitions = 0; repetitions < R; repetitions++)
    {
        int i1 = i1_init;
        int i2 = i2_init;
        int s = s_init;
        double cost = 0;
        for(int time = 0; time < 1000; time++)
        {
            double thistimecost = 0.0;

            //
            int action = policy[s][i1][i2];
            if (action > 0)
            {
                if (action == s)
                {
                    if (action == 1) i1 += MAKE; else i2 += MAKE;
                }
                else
                {
                    thistimecost += 400;
                    if (action == 1) i1 += MAKE/2; else i2 += MAKE/2;
                }
            }
            //
            
            int d1 = prob1[rand() % 21];
            int d2 = prob2[rand() % 21];


            if (d1 > i1)
            {
                i1 = 0;
                thistimecost += (d1-i1) * 100;
            }
            else i1 -= d1;

            if (d2 > i2)
            {
                i2 = 0;
                thistimecost += (d2-i2) * 100;
            }
            else i2 -= d2;

            i1 = MIN(i1,100);
            i2 = MIN(i2,100);

            thistimecost += (i1+i2)/100.0;

            cost += thistimecost*pow(0.99,time);
            s = action;
        }
        total_costs += cost / (1.0 * R);
    }
    return total_costs;
}

int main(int argc, char const *argv[])
{
    FILE* fp = fopen("mu.csv","r");
    for(int s = 0; s < 3; s++)
        for(int i = 0; i < 101; i++)
            for(int j = 0; j < 101; j++)
            {
                float f;
                fscanf(fp, "%f",&f);
                policy[s][i][j] = (int)f;
            }
    printf("%f\n",simulate(0,0,0));
    return 0;
}
