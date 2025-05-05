#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define TAG_XTX 1
#define TAG_PROGRESS 2

int main(int argc, char** argv) {
   MPI_Init(&argc, &argv);
   int size, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   if (rank != 0) {
       MPI_Finalize();
       return 0;
   }

   double global_XTX = 0.0;
   double global_XTy = 0.0;

   for (int i = 1; i < size; ++i) {
       double partial_XTX, partial_XTy;
       MPI_Recv(&partial_XTX, 1, MPI_DOUBLE, i, TAG_XTX, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       MPI_Recv(&partial_XTy, 1, MPI_DOUBLE, i, TAG_XTX, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

       global_XTX += partial_XTX;
       global_XTy += partial_XTy;
   }

   double weight = global_XTy / global_XTX;
   printf("Estimated linear coefficient w = %.6f\n", weight);

   for (int i = 1; i < size; ++i) {
       double progress;
       MPI_Recv(&progress, 1, MPI_DOUBLE, i, TAG_PROGRESS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       printf("Worker %d completed task with progress: %.2f%%\n", i, progress * 100);
   }

   MPI_Finalize();
   return 0;
}
