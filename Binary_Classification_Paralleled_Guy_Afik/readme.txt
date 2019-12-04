

This program parallels Binary classification algorithm with openMP, Cuda and MPI interfaces, all used to make the performance as best as possible.
The program reads it’s input from a text file, then implement the algorithm mentioned above  

The program has two solutions, 2 different ways to parallel the program. I implemented them both

first solution: each process calculated it's own time zone and return if q < QC was found.

second solution: each process gets the same amount of task (number of points / number of processes we are running with) and works only with his part.
Master - Slave implementation which means processes number 0 is in charge of the entire program, and all the rest processes follow up by his rules.



MPI creates the amount of processes you wish to run with, and runs the program with that amount.
It is an interface used to communicate between processes as well.

openMP is a thread based interface 

Cuda is an interface for GPU programming


Mpi description : https://mpitutorial.com/tutorials/
openMP description: https://www.openmp.org/wp-content/uploads/OpenMP4.0.0.Examples.pdf
Cuda description: https://devblogs.nvidia.com/easy-introduction-cuda-c-and-c/
A description of the algorithm is attached in the word called “Parallel implementation of binary classification”

To conclude, this program uses process based interface, thread based interface and GPU threads interface to implement the Parallel binary classification algorithm

The program is written in c programming language.
