#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define STD_TAG 0

int** createMatrix(int nRows, int nCols) {
  int i, j;
  int ** matrix;

  matrix = malloc(nRows * sizeof(int *));
  for (i = 0; i < nRows; i++) {
    matrix[i] = malloc(nCols * sizeof(int));
  }

  return matrix;
}

void checkMinimum(int size, int *a, int *b, int *c) {
  int i, j, k;
  
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
	int aVal = a[k + i * size];
	int bVal = b[j + k * size];
	int cVal = c[j + i * size];
      
	if (aVal != -1 && bVal != -1) {
	  if (cVal == -1 || cVal > aVal + bVal) c[j + i * size] = aVal + bVal;
	}
      }
    }
  }
}

// MAIN
main (int argc, char * argv[]) {
  int P, Q, rank, nRows, nCols, submatrixSize, i, j;
  //double startTime, finishTime;
  MPI_Status status;
  MPI_Request request;

  int **matrix, *localMatrix, *rowMatrix, *colMatrix, *resMatrix, *solucao;
  MPI_Comm rowsComm, colsComm, gridComm;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Q = sqrt(P);

  if (rank == 0) {
    int rows, cols;
    
    // Ler tamanho da matrix
    scanf("%d", &nRows);
    nCols = nRows;

    // Verificar condicoes do problema
    if (P != Q*Q || nRows % Q != 0) {
      // Erro
      submatrixSize = -1;
      MPI_Bcast(&submatrixSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Finalize();

      printf("Error: Settings do not met the Fox algorithm conditions\n");

      return;
    }
    
    // Ler matriz inicial do problema
    matrix = createMatrix(nRows, nCols);
    for (i = 0; i < nRows; i++) {
      for (j = 0; j < nCols; j++) {
	scanf("%d", &matrix[i][j]);
	if (matrix[i][j] == 0 && i != j) {
	  matrix[i][j] = -1;
	}
      }
    }

    //startTime = MPI_Wtime();

    // Divisão e envio das sub-matrizes
    submatrixSize = nRows / Q;
    MPI_Bcast(&submatrixSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *tempMatrix;
    for (i = 0; i < P; i++) {
      if (i == 0)
	localMatrix = malloc((submatrixSize * submatrixSize) * sizeof(int));
      else
	tempMatrix = malloc((submatrixSize * submatrixSize) * sizeof(int));

      int submatrixPosX = i / (int) floor(Q);
      int submatrixPosY = i % (int) floor(Q);
      for (rows = 0; rows < submatrixSize; rows++) {
	for (cols = 0; cols < submatrixSize; cols++) {
	  if (i == 0)
	    localMatrix[cols + rows * submatrixSize] = matrix[rows + (submatrixPosX * submatrixSize)][cols + (submatrixPosY * submatrixSize)];
	  else
	    tempMatrix[cols + rows * submatrixSize] = matrix[rows + (submatrixPosX * submatrixSize)][cols + (submatrixPosY * submatrixSize)];
	}
      }

      if (i != 0) MPI_Send(tempMatrix, submatrixSize*submatrixSize, MPI_INT, i, STD_TAG, MPI_COMM_WORLD);
    }    
  } else {
    // Recebe o tamanho e a sua submatriz
    MPI_Bcast(&submatrixSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (submatrixSize == -1) {
      MPI_Finalize();
      return;
    }
    
    localMatrix = malloc((submatrixSize * submatrixSize) * sizeof(int));
    MPI_Recv(localMatrix, submatrixSize * submatrixSize, MPI_INT, 0, STD_TAG, MPI_COMM_WORLD, &status);
  }

  // Criar matriz catesiana de processos
  int dims[2], periods[2];
  dims[0] = Q;
  dims[1] = Q;
  periods[0] = 1;
  periods[1] = 1;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &gridComm);

  // Comunicador com as colunas
  dims[0] = 1;
  dims[1] = 0;
  MPI_Cart_sub(gridComm, dims, &colsComm);

  // Comunicador com as linhas
  dims[0] = 0;
  dims[1] = 1;
  MPI_Cart_sub(gridComm, dims, &rowsComm);

  rowMatrix = malloc((submatrixSize * submatrixSize) * sizeof(int));
  colMatrix = malloc((submatrixSize * submatrixSize) * sizeof(int));
  resMatrix = malloc((submatrixSize * submatrixSize) * sizeof(int));

  // Precessos na diagonal enviam as suas submatrizes para as linhas
  if ((rank / (int) floor(Q)) == (rank % (int) floor(Q)))
    memcpy(rowMatrix, localMatrix, (submatrixSize * submatrixSize) * sizeof(int));

  // Calcular as suas coordenadas na grelha de processos
  MPI_Cart_coords(gridComm, rank, 2, dims);

  memcpy(colMatrix, localMatrix, (submatrixSize * submatrixSize) * sizeof(int));
  memcpy(resMatrix, localMatrix, (submatrixSize * submatrixSize) * sizeof(int));

  int d;
  int stage, srcCol, destCol;

  // Rodar as matrizes na coluna
  srcCol = (dims[0] + 1) % Q;
  destCol = ((dims[0] - 1) + Q) % Q;

  for (d = 2; d < 2 * (submatrixSize * Q); d *= 2) {
    for (stage = 0; stage < Q; stage++) {
      int bcast = (dims[0] + stage) % Q;

      if (bcast == dims[1]) {
	// Envia a sua matriz para os processos na mesma linha e verifica o minimo
	MPI_Bcast(localMatrix, submatrixSize*submatrixSize, MPI_INT, bcast, rowsComm);
	checkMinimum(submatrixSize, localMatrix, colMatrix, resMatrix);
      } else {
	// Recebe matriz do broadcast na linha e verifica o minimo
	MPI_Bcast(rowMatrix, submatrixSize*submatrixSize, MPI_INT, bcast, rowsComm);
	checkMinimum(submatrixSize, rowMatrix, colMatrix, resMatrix);
      }

      MPI_Sendrecv_replace(colMatrix, submatrixSize*submatrixSize, MPI_INT, destCol, STD_TAG, srcCol, STD_TAG, colsComm, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    // No final do stage, preparar a proxima ronda de calculos
    memcpy(localMatrix, resMatrix, (submatrixSize * submatrixSize) * sizeof(int));
    memcpy(colMatrix, localMatrix, (submatrixSize * submatrixSize) * sizeof(int));
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  // Reunir submatrizes no processo root
  if (rank == 0) solucao = malloc((nRows * nRows) * sizeof(int));
  MPI_Gather(resMatrix, submatrixSize*submatrixSize, MPI_INT, solucao, submatrixSize*submatrixSize, MPI_INT, 0, MPI_COMM_WORLD); 
  
  // Juntar todas as submatrizes numa só
  if (rank == 0) {
    int rows, cols;

    for (i = 0; i < P; i++) {
      int submatrixPosX = i / (int) floor(Q);
      int submatrixPosY = i % (int) floor(Q);
      int pos = i * (int) pow(submatrixSize, 2);

      for (rows = 0; rows < submatrixSize; rows++) {
	for (cols = 0; cols < submatrixSize; cols++) {
	  if (solucao[pos + (cols + rows*submatrixSize)] == -1)
	    matrix[rows + (submatrixPosX * submatrixSize)][cols + (submatrixPosY * submatrixSize)] = 0;
	  else
	    matrix[rows + (submatrixPosX * submatrixSize)][cols + (submatrixPosY * submatrixSize)] = solucao[pos + (cols + rows*submatrixSize)];
	}
      }
    }

    //finishTime = MPI_Wtime();
    //printf("Tempo de execucao: %f\n", (finishTime - startTime));
    
    // Imprimir os resultados
    for (i = 0; i < nRows; i++) {
      for (j = 0; j < nCols; j++) {
	if (j != nCols - 1)
	  printf("%d ", matrix[i][j]);
	else
	  printf("%d", matrix[i][j]);
      }
      printf("\n");
    }
  }
  
  if (rank == 0) {
    for (i = 0; i < nRows; i++) {
      free(matrix[i]);
    }
    free(solucao);
  }

  free(localMatrix);
  free(rowMatrix);
  free(colMatrix);
  free(resMatrix);
  
  MPI_Finalize();
}
