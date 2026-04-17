/*
 * test_initpop.c  —  fully self-contained, zero external linkage
 *
 * Reproduces the exact initPopulation() logic from psomatrixcircuit.c
 * and prints every particle's structure after the call.
 *
 * Configuration wired in:
 *   numInputs  = 4   ->  2^4 = 16 truth-table rows
 *   numOutputs = 2
 *   representation = INTEGERB (0)
 *   Matrix: numRows=4, numCols=3  ->  tMat=12 gate cells
 *   numGates = 5  (AND=0 OR=1 NOT=2 WIRE=3 XOR=4)
 *   tPop = 6,  cardinality = 1
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Representation constants */
#define BINARY   2
#define INTEGERA 1
#define INTEGERB 0

/* Particle structure (from psomatrixcircuit.h) */
typedef struct {
    double   *vi;
    unsigned *chromX;
    unsigned  numGates;
    unsigned  numNoGates;
    unsigned  numEqual;
    double    fitness;
} particle;

/* Global state */
static unsigned  tPop, nAllele, nVar;
static unsigned  representation, cardinality;
static unsigned  numRows, numCols, tMat;
static unsigned  numGates, numInputs, numOutputs;
static unsigned  numRowsTT, numTotalOutputs;
static unsigned *lInf, *lSup;
static particle *population;

/* Minimal RNG (mirrors random.c) */
static unsigned initRandom(unsigned seed)
{
    unsigned s = seed ? seed : (unsigned)time(NULL);
    srand(s);
    return s;
}
static double rndF(void) { return (double)rand() / RAND_MAX; }
static unsigned rndIR(unsigned lower, unsigned upper)
{
    return (unsigned)(lower + rndF() * (upper - lower));
}
static unsigned flip(double prob)
{
    return rndF() <= prob ? 1 : 0;
}

/* initPopulation — copied verbatim from psomatrixcircuit.c */
static void initPopulation(void)
{
    unsigned i, j;
    for (i = 0; i < tPop; i++)
        for (j = 0; j < nAllele; j++)
            population[i].chromX[j] =
                representation == BINARY
                    ? flip(0.5)
                    : rndIR(lInf[j], (lSup[j] + 1) * cardinality) % (lSup[j] + 1);
}

/* Display helpers */
static const char *gateName(unsigned g)
{
    switch (g) {
        case 0: return "AND ";
        case 1: return "OR  ";
        case 2: return "NOT ";
        case 3: return "WIRE";
        case 4: return "XOR ";
        default: return "????";
    }
}

static void printParticle(unsigned idx, particle *p)
{
    unsigned c, r, cell, in1, in2, gtyp, j, o;

    printf("\n+-------------------------------------------------------------+\n");
    printf("|  Particle %u\n", idx);
    printf("|  fitness=%.2f  numEqual=%u  numGates=%u  numNoGates=%u\n",
           p->fitness, p->numEqual, p->numGates, p->numNoGates);

    /* Raw chromosome */
    printf("|\n|  chromX[%u]  (triplets: in1 in2 gtp | ...)\n|  ", nAllele);
    for (j = 0; j < nAllele; j++) {
        printf("%2u", p->chromX[j]);
        if ((j + 1) % 3 == 0) printf(" | ");
        else                    printf(" ");
    }
    printf("\n");

    /* Velocities */
    printf("|\n|  vi[%u]  (all 0.0 at init)\n|  ", nAllele);
    for (j = 0; j < nAllele; j++) printf("%+.1f ", p->vi[j]);
    printf("\n");

    /* Decoded matrix */
    printf("|\n|  Decoded matrix (%u rows x %u cols)\n", numRows, numCols);
    printf("|  cell index = col*numRows + row  |  allele = cell*3 + {0,1,2}\n|\n");
    printf("|         ");
    for (c = 0; c < numCols; c++) printf("   col%-2u          ", c);
    printf("\n|         ");
    for (c = 0; c < numCols; c++) printf("  gate(in1,in2)   ");
    printf("\n");
    for (r = 0; r < numRows; r++) {
        printf("|  row%-2u  ", r);
        for (c = 0; c < numCols; c++) {
            cell = c * numRows + r;
            in1  = p->chromX[cell * 3 + 0] % numRows;
            in2  = p->chromX[cell * 3 + 1] % numRows;
            gtyp = p->chromX[cell * 3 + 2] % numGates;
            printf("  %s(r%u,r%u)     ", gateName(gtyp), in1, in2);
        }
        printf("\n");
    }

    /* Circuit outputs */
    printf("|\n|  Circuit outputs (last col, rows 0..%u):\n", numOutputs - 1);
    for (o = 0; o < numOutputs; o++) {
        cell = (numCols - 1) * numRows + o;
        in1  = p->chromX[cell * 3 + 0] % numRows;
        in2  = p->chromX[cell * 3 + 1] % numRows;
        gtyp = p->chromX[cell * 3 + 2] % numGates;
        printf("|    output[%u]  <-  col%u-row%u : %s(r%u, r%u)\n",
               o, numCols - 1, o, gateName(gtyp), in1, in2);
    }
    printf("+-------------------------------------------------------------+\n");
}

int main(void)
{
    unsigned i, j;

    /* 1. Fixed configuration */
    numInputs       = 4;
    numOutputs      = 4;
    numRowsTT       = 1u << numInputs;
    numTotalOutputs = numOutputs * numRowsTT;
    representation  = INTEGERB;
    cardinality     = 1;
    numGates        = 5;
    numRows         = 4;
    numCols         = 4;
    tMat            = numRows * numCols;
    tPop            = 6;

    /* 2. Derived sizes */
    nVar    = numRows * numCols * 3;
    nAllele = nVar;

    /* 3. Bounds (mirrors initBounds) */
    lInf = (unsigned *)calloc(nVar, sizeof(unsigned));
    lSup = (unsigned *)malloc(sizeof(unsigned) * nVar);
    for (i = 0; i < nVar; i++)
        lSup[i] = ((i + 1) % 3) ? numRows - 1 : numGates - 1;

    /* 4. Allocate population */
    population = (particle *)calloc(tPop, sizeof(particle));
    for (i = 0; i < tPop; i++) {
        population[i].chromX = (unsigned *)calloc(nAllele, sizeof(unsigned));
        population[i].vi     = (double *)  calloc(nAllele, sizeof(double));
    }

    /* 5. Seed and run */
    unsigned seed = initRandom(0);

    printf("=================================================================\n");
    printf("  initPopulation() structure dump\n");
    printf("=================================================================\n");
    printf("  Circuit  : %u inputs, %u outputs, %u truth-table rows\n",
           numInputs, numOutputs, numRowsTT);
    printf("  Matrix   : %u rows x %u cols  (tMat = %u gate cells)\n",
           numRows, numCols, tMat);
    printf("  Encoding : Integer B  (1 allele per variable)\n");
    printf("  nVar=%u  nAllele=%u  numGates=%u  cardinality=%u\n",
           nVar, nAllele, numGates, cardinality);
    printf("  tPop=%u   seed=%u\n", tPop, seed);
    printf("-----------------------------------------------------------------\n");
    printf("  Allele bounds (pattern repeats every 3):\n");
    printf("    field      allele positions    lInf  lSup\n");
    printf("    input1     i%%3 == 0             0     %u\n", numRows - 1);
    printf("    input2     i%%3 == 1             0     %u\n", numRows - 1);
    printf("    gateType   i%%3 == 2             0     %u\n", numGates - 1);
    printf("-----------------------------------------------------------------\n");
    printf("  initPopulation formula (Integer B):\n");
    printf("    chromX[j] = rndIR(lInf[j], (lSup[j]+1)*cardinality)\n");
    printf("                %% (lSup[j]+1)\n");
    printf("    -> input alleles: uniform in [0, %u]\n", numRows - 1);
    printf("    -> gate  alleles: uniform in [0, %u]\n", numGates - 1);
    printf("=================================================================\n");

    /* THE CALL UNDER TEST */
    initPopulation();

    /* 6. Print every particle */
    printf("\n=== Population after initPopulation() ===\n");
    for (i = 0; i < tPop; i++)
        printParticle(i, &population[i]);

    /* 7. Gate-type frequency */
    unsigned gateFreq[5]  = {0};
    unsigned totalCells   = tPop * (nAllele / 3);
    for (i = 0; i < tPop; i++)
        for (j = 2; j < nAllele; j += 3)
            gateFreq[population[i].chromX[j] % numGates]++;

    printf("\n=== Gate-type distribution across all %u gate cells ===\n", totalCells);
    printf("  (expected ~20%% each for uniform Integer B)\n\n");
    printf("  Gate   Count    %%\n");
    printf("  -----  -----  -----\n");
    const char *gnames[5] = {"AND ", "OR  ", "NOT ", "WIRE", "XOR "};
    for (i = 0; i < numGates; i++)
        printf("  %s   %3u    %5.1f%%\n",
               gnames[i], gateFreq[i], 100.0 * gateFreq[i] / totalCells);

    /* 8. Input-wire distribution */
    unsigned inFreq[4]       = {0};
    unsigned totalInputRefs  = tPop * (nAllele / 3) * 2;
    for (i = 0; i < tPop; i++)
        for (j = 0; j < nAllele; j += 3) {
            inFreq[population[i].chromX[j+0] % numRows]++;
            inFreq[population[i].chromX[j+1] % numRows]++;
        }

    printf("\n=== Input-wire distribution across all %u references ===\n", totalInputRefs);
    printf("  (expected ~25%% each for uniform Integer B)\n\n");
    printf("  Row   Count    %%\n");
    printf("  ---   -----  -----\n");
    for (i = 0; i < numRows; i++)
        printf("  r%u    %3u    %5.1f%%\n",
               i, inFreq[i], 100.0 * inFreq[i] / totalInputRefs);

    return 0;
}