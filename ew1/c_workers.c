#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAG_XTX 1
#define TAG_PROGRESS 2

double* read_npy(const char* filename, int* length_out) {
   FILE* f = fopen(filename, "rb");
   if (!f) {
       fprintf(stderr, "Failed to open file %s\n", filename);
       exit(EXIT_FAILURE);
   }

   fseek(f, 80, SEEK_SET);
   int len;
   fread(&len, sizeof(int), 1, f);
   *length_out = len;

   double* data = (double*)malloc(len * sizeof(double));
   fread(data, sizeof(double), len, f);
   fclose(f);
   return data;
}

int main(int argc, char** argv) {
   MPI_Init(&argc, &argv);
   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   if (rank == 0) {
       MPI_Finalize();
       return 0;
   }

   int X_len, y_len;
   double* X = read_npy("X.npy", &X_len);
   double* y = read_npy("y.npy", &y_len);

   if (X_len != y_len) {
       fprintf(stderr, "X and y have different lengths\n");
       exit(EXIT_FAILURE);
   }

   int chunk_size = X_len / (size - 1);
   int start = (rank - 1) * chunk_size;
   int end = (rank == size - 1) ? X_len : start + chunk_size;

   double XTX = 0.0;
   double XTy = 0.0;

   for (int i = start; i < end; ++i) {
       XTX += X[i] * X[i];
       XTy += X[i] * y[i];
   }

   MPI_Send(&XTX, 1, MPI_DOUBLE, 0, TAG_XTX, MPI_COMM_WORLD);
   MPI_Send(&XTy, 1, MPI_DOUBLE, 0, TAG_XTX, MPI_COMM_WORLD);
   MPI_Finalize();
   return 0;
}
