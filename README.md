# eigenvector-calculator
Using power iteration and inverse power iteration algorithms to find eigenvectors with highest and lowest eigenvalue

This code is written for educational purposes to implement object oriented programming, dynamic memory allocation, power iteration, inverse power iteration, and gaussian elimination with partial pivoting.

### Compiling and Running
source.cpp can be compiled with any c++ compiler.
program needs to be run from a command line with 3 arguments in following order:
* name of the file that contains nxn matrix's values
* tolerance value for power iteration algorithm
* arbitrary name for output file

### Outputs
* Highest eigenvalue of the matrix with corresponding eigenvector
* Lowest eigenvalue of the matrix with corresponding eigenvector
* If matrix is singular program prints out "A is a singular matrix!"
and does not print any output.
All values are written into the file with given name

### Exceptional Cases
When power iteration does not converge into any value, program prints out:
* "Matrix does not have dominant eigenvector"
* "Normalized power iteration does not converge"
!!For Matrixes with negative eigenvalues program finds absolute values of eigenvalues.
