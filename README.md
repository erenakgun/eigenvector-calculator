# eigenvector-calculator
Using power iteration and inverse power iteration algorithms to find eigenvectors with largest and smallest eigenvalue

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

### Example
Result for matrix A with eigenvalues 4.00 (largest) and 2.00 (smallest) is illustrated above:
```
**Parameters:** A.txt 1e-6 b.txt 
```
A.txt file:
```
2.7383  -0.5011  0.8817
-0.3039  2.3639  1.4258
0.1285  0.0665  3.8978
```
Results in b.txt
```
Eigenvalue#1: 3.99998
0.3808
0.800739
1
Eigenvalue#2: 2.00001
0.783969
1
-0.0881234
```
