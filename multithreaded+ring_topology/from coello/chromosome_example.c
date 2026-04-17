/*
 * chromosome_example.c
 *
 * Minimal, self-contained demo of how a chromosome is interpreted in * Coello / Hernández Luna style matrix-PSO circuit encoding:
 *   - Same decode() and booleanString() logic as matrixpso.c
 *   - Separate buffers for input1, input2, gateType (see note below)
 *
 * Build (from this directory):
 *   gcc -std=c99 -O -o chromosome_example chromosome_example.c -lm
 *
 * Note: In the original matrixpso.c, reserveMatrixMemory() assigns
 * input1, input2, and gateType to the same malloc (memVarX). That
 * makes the three arrays alias the same memory; this example uses
 * three distinct arrays so decode/evaluate/expression behave as intended.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- mirrors circuits.h gate ids --- */
enum { AND = 0, OR = 1, NOT = 2, WIRE = 3, XOR = 4, NOT1 = 5, WIRE1 = 6, XOR1 = 7 };

#define BINARY   2
#define INTEGERA 1
#define INTEGERB 0

#define MAX_ROWS 16
#define MAX_COLS 16
#define MAX_TMAT (MAX_ROWS * MAX_COLS)
#define MAX_NVAR (MAX_TMAT * 3)
#define MAX_BOOL 16000

static unsigned numRows, numCols, tMat, nVar, nAllele;
static unsigned representation;
static unsigned numGates;
static unsigned numInputs;
static unsigned bitVariable[MAX_NVAR];
static unsigned lInf[MAX_NVAR], lSup[MAX_NVAR];

static unsigned input1[MAX_TMAT], input2[MAX_TMAT], gateType[MAX_TMAT];

/* Decode chromosome M into input1[], input2[], gateType[] (matrixpso.c decode) */
static void decode(const unsigned *M)
{
    int k;
    unsigned i, j, num, in;

    for (i = 0, j = 0, in = 0; i < nVar; i++) {
        if (representation != BINARY)
            num = M[i];
        else {
            num = 0;
            for (k = (int)bitVariable[i] - 1; k >= 0; k--, j++)
                if (M[j])
                    num += (unsigned)pow(2.0, k);
        }

        switch (i % 3) {
        case 0:
            input1[in] = (num % numRows);
            break;
        case 1:
            input2[in] = (num % numRows);
            break;
        case 2:
            gateType[in++] = (num % numGates);
            break;
        }
    }
}

/* Recursive string builder (matrixpso.c booleanString) */
static void booleanString(unsigned cell, int col, char *str, size_t strCap)
{
    unsigned new1, new2;
    char gateStr[8];
    char varStr[3];

    varStr[1] = '\0';
    switch (gateType[cell]) {
    case AND:
        strcpy(gateStr, "(AND1 ");
        break;
    case OR:
        strcpy(gateStr, "(OR1 ");
        break;
    case NOT:
    case NOT1:
        strcpy(gateStr, "(NOT1 ");
        break;
    case XOR:
    case XOR1:
        strcpy(gateStr, "(XOR1 ");
        break;
    case WIRE:
    case WIRE1:
        strcpy(gateStr, "(WIRE ");
        break;
    default:
        strcpy(gateStr, "(? ");
        break;
    }

    if (strlen(str) + strlen(gateStr) + 2 >= strCap)
        return;
    strcat(str, gateStr);

    if (col <= 0) {
        varStr[0] = (char)(input1[cell] % numInputs + (int)'A');
        if (strlen(str) + 2 >= strCap)
            return;
        strcat(str, varStr);
        if (gateType[cell] != WIRE && gateType[cell] != WIRE1 &&
            gateType[cell] != NOT && gateType[cell] != NOT1) {
            varStr[0] = (char)(input2[cell] % numInputs + (int)'A');
            if (strlen(str) + 3 >= strCap)
                return;
            strcat(str, " ");
            strcat(str, varStr);
        }
    } else {
        new1 = numRows * (unsigned)(col - 1) + input1[cell];
        new2 = numRows * (unsigned)(col - 1) + input2[cell];
        booleanString(new1, col - 1, str, strCap);
        if (gateType[cell] != WIRE && gateType[cell] != WIRE1 &&
            gateType[cell] != NOT && gateType[cell] != NOT1) {
            if (strlen(str) + 1 >= strCap)
                return;
            strcat(str, " ");
            booleanString(new2, col - 1, str, strCap);
        }
    }
    if (strlen(str) + 2 >= strCap)
        return;
    strcat(str, ")");
}

/* Root output `out` at last column, row `out` (matrixpso.c expression) */
static void expression(const unsigned *M, unsigned out, char *buf, size_t bufCap)
{
    buf[0] = '\0';
    decode(M);
    booleanString(numRows * (numCols - 1) + out, (int)numCols - 1, buf, bufCap);
}

static void printTriples(const unsigned *M)
{
    unsigned c;
    decode(M);
    printf(" cell  col row in1  in2  gate name\n");
    for (c = 0; c < tMat; c++) {
        unsigned col = c / numRows;
        unsigned row = c % numRows;
        const char *gname = "?";
        switch (gateType[c]) {
        case AND:
            gname = "AND";
            break;
        case OR:
            gname = "OR";
            break;
        case NOT:
            gname = "NOT";
            break;
        case WIRE:
            gname = "WIRE";
            break;
        case XOR:
            gname = "XOR";
            break;
        default:
            break;
        }
        printf("  %3u    %u %u    %2u   %2u   %u %s\n",
               c, col, row, input1[c], input2[c], gateType[c], gname);
    }
}

/* Tiny truth-table sim: same structure as matrixpso.c evaluate() */
static void simulateRow(const unsigned *M, unsigned ttRow, unsigned *outVals, unsigned numOut)
{
    static unsigned inTT[MAX_ROWS];
    static unsigned output[MAX_ROWS];
    unsigned i, j, k, in;

    decode(M);

    /* generateTT-style minterm bits for ttRow */
    for (j = 0; j < numInputs; j++)
        inTT[j] = (ttRow >> j) & 1u;
    for (; j < numRows; j++)
        inTT[j] = inTT[j - numInputs];

    for (j = 0, in = 0; j < tMat; j++) {
        switch (gateType[j]) {
        case AND:
            output[in] = inTT[input1[j]] & inTT[input2[j]];
            break;
        case OR:
            output[in] = inTT[input1[j]] | inTT[input2[j]];
            break;
        case NOT:
        case NOT1:
            output[in] = inTT[input1[j]] ? 0u : 1u;
            break;
        case WIRE:
        case WIRE1:
            output[in] = inTT[input1[j]];
            break;
        case XOR:
        case XOR1:
            output[in] = inTT[input1[j]] ^ inTT[input2[j]];
            break;
        default:
            output[in] = 0;
            break;
        }
        if (!((j + 1) % numRows)) {
            in = 0;
            for (k = 0; k < numRows; k++)
                inTT[k] = output[k];
        } else
            in++;
    }
    for (i = 0; i < numOut; i++)
        outVals[i] = output[i];
}

static void runExample(
    const char *title,
    unsigned rows,
    unsigned cols,
    unsigned inCount,
    unsigned gCount,
    unsigned rep,
    const unsigned *chrom,
    unsigned chromLen,
    unsigned numOut)
{
    unsigned r;
    char buf[MAX_BOOL];

    printf("\n=== %s ===\n", title);
    numRows = rows;
    numCols = cols;
    tMat = numRows * numCols;
    numInputs = inCount;
    numGates = gCount;
    representation = rep;

    nVar = tMat * 3;
    if (representation == BINARY) {
        unsigned i;
        nAllele = 0;
        for (i = 0; i < nVar; i++) {
            lInf[i] = 0;
            lSup[i] = ((i + 1) % 3) ? (numRows - 1) : (numGates - 1);
            bitVariable[i] = (unsigned)ceil(log((double)(lInf[i] + lSup[i] + 1)) / log(2.0));
            nAllele += bitVariable[i];
        }
    } else {
        nAllele = nVar;
    }

    if (chromLen != nAllele) {
        printf("Error: chromosome length %u != expected nAllele %u\n", chromLen, nAllele);
        return;
    }

    printf("numRows=%u numCols=%u numInputs=%u numGates=%u representation=%s\n",
           numRows, numCols, numInputs, numGates,
           rep == BINARY ? "BINARY" : (rep == INTEGERA ? "INTEGERA" : "INTEGERB"));

    printTriples(chrom);

    printf("\nBoolean expressions (Output1 = index 0, etc.):\n");
    for (r = 0; r < numOut; r++) {
        expression(chrom, r, buf, sizeof buf);
        printf("  Output %u: %s\n", r + 1, buf);
    }

    printf("\nSimulated outputs for first few minterms (rows 0..3 of truth table):\n");
    for (r = 0; r < 4 && r < (1u << numInputs); r++) {
        unsigned ov[8];
        simulateRow(chrom, r, ov, numOut);
        printf("  tt row %u: ", r);
        for (unsigned o = 0; o < numOut; o++)
            printf("y%u=%u ", o, ov[o]);
        printf("\n");
    }
}

int main(void)
{
    /* Example A: 3 rows, 2 columns, 2 inputs, 5 gate types, integer chromosome (18 genes) */
    static const unsigned exampleA[] = {
       0, 1, 0,  /* AND row0: inputs 0,1 */
        1, 1, 3,  /* WIRE row1 */
        2, 2, 3,  /* WIRE row2 */
        0, 1, 0,  /* col1 AND */
        0, 1, 4,  /* XOR */
        1, 2, 1   /* OR */
    };

    /* Example B: 5x5, 4 inputs, 5 gates — 75 integers = 25 cells × 3 (same shape as experiment3.dta) */
    static const unsigned exampleB_75[] = {
        0, 2, 1, 2, 4, 1, 1, 2, 3, 3, 1, 4, 4, 2, 0, 0, 1, 1, 3, 3, 2, 4, 3, 2, 4, 2, 4, 1, 4, 4, 3, 0, 0, 3, 4, 0, 0, 2, 0, 0, 1, 4, 3, 1, 3, 2, 3, 4, 3, 2, 4, 2, 2, 1, 1, 2, 4, 3, 1, 4, 1, 4, 0, 1, 2, 2, 2, 4, 3, 3, 2, 1, 1, 3, 4
    };

    printf("chromosome_example — matrix PSO chromosome decode / expression / tiny sim\n");

    runExample("Small 3x2 grid, 2 inputs, 3 outputs (last-column rows 0..2)",
               3, 2, 2, 5, INTEGERB, exampleA,
               (unsigned)(sizeof exampleA / sizeof exampleA[0]), 3);

    runExample("5x5 grid, 4 inputs, 3 outputs — sample 75-gene chromosome (integer B)",
               5, 5, 4, 5, INTEGERB, exampleB_75,
               (unsigned)(sizeof exampleB_75 / sizeof exampleB_75[0]), 3);

    return 0;
}
