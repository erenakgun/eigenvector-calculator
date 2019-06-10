#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib> 
#include <math.h>
#include <ctime>
using namespace std;
float tolerance = 0;			// tolerance value for power iteration
const float eps = 0.05; 	// Approximate machine precision value for float numbers to be used in singularity check

float rand_FloatRange(float a, float b)
{
    return ((b - a) * ((float)rand() / RAND_MAX)) + a;
}
float abs(float x)
{
/*
  This function returns absolute value of x without changing x.
  
  @param x Float value
  @return Absolute value of x
*/
	if(x<0)
	{
		x=-x;
	}
	return x;
}

class Vector{
		friend class Matrix;
		// dimension of vector
		int dim;
		// values of vector
		float *Vec;
		// infinity norm of vector
		float norm;
	public:
		// constructers
		Vector ();
   		Vector(float *b, int n);
   		Vector(float x, int n);
   		Vector(float min, float max, int n);
   		// copy constructer and copy assignment
		Vector(const Vector& V); 
        Vector& operator= (const Vector& V);
        // destructer
    	~Vector();
		// prints values of vector        
		void print();
		// multiplication and division of values with a float number
     	Vector operator* (float x);
		Vector operator/ (float x);
		// vector-vector elementwise substraction and addition
		Vector operator+(const Vector& V);
     	Vector operator-(const Vector& V);
		//updates infinity norm member of vector
		void infNorm();
		// to change rows of the vector
		void VecRowExc( int row1, int row2);

};

class Matrix {
		//dimension of nxn matrix
		int dim;
		// values of matrix
		float **Mat;
		// Highest eigenvalue of the matrix and corresponding eigenvector
		float eigValHigh;
		Vector eigVecHigh;
		// Lowest eigenvalue of the matrix and corresponding eigenvector
		float eigValLow;
		Vector eigVecLow;
	 public:
	 	// constructers
		Matrix ();
   		Matrix(float **A, int n);
   		Matrix(float x, int n);
   		// copy constructer and copy assignment
		Matrix(const Matrix& M); 
        Matrix& operator= (const Matrix& M);
		// destructer
    	~Matrix();
		// prints values of the matrix	   		
		void print();
		// prints eigenvalue members and corresponding eigenvectors
		void printEig();
		// output eigenvalues and eigen vectors to a file with given filename
		void outToFile(const char* s);
		// matrix-matrix elementwise addition 
     	Matrix operator+ (const Matrix& M);
		// matrix vector multiplication 
        Vector operator* (const Vector& V);
        // calculates eigenvalues and eigenvectors of the matrix
        void highEigen ();
		void lowEigen();
		// linear equation solver
		Vector solve(const Vector& V);
		// to exchange rows of the matrix during linear equation solution
		void MatRowExc(int row1, int row2, int index);
};






int main( int argc, char *argv[] )  {

   if( argc == 4 ) {
   	
   	tolerance = atof(argv[2]);	     	// Tolerance value for power iteration
   	
	string line;				    	// Creating a string object to read files
  	int n=0;						    // Dimensions of matrix A and row number of b
  	
  	ifstream file_mat (argv[1]);
  	if (file_mat.is_open())
  	{
    while ( getline (file_mat,line) )   // finds matrix size n
    {
      n++;
    }
 	
	//Reset stream position 
    file_mat.close();
    file_mat.open(argv[1]);
         
    
    // Dynamic memory allocation for nxn matrix
    float **A= new float*[n];
    for(int i=0;i<n;i++)
	{
	A[i]=new float [n];
	}
	
	// Assign values of matrix file to A array
	for(int i=0;i<n;i++)
	{
		getline (file_mat,line);
		istringstream reader(line);
		for(int j=0;j<n;j++)
		{
			float temp;
			reader>>temp;
			A[i][j]=temp;
		}
	
	}
	Matrix myMat(A,n);
	myMat.highEigen();
	myMat.lowEigen();
	myMat.outToFile(argv[3]);
}
  else
	{
		cout << "Unable to open Matrix files";
	} 



	}


   else {
	
	cout<<"Please enter valid arguments";

   }

   	return 0;
}


/************* Class Matrix Definitions *******************/


// Default constructer
Matrix::Matrix():dim(0),eigValHigh(0),eigValLow(0){
	Mat= 0;
}

// Default constructer with double pointer
Matrix::Matrix(float **A, int n):dim(n){

    this->Mat = A;

}

// Default constructer with element value
Matrix::Matrix(float x, int n):dim(n),eigValHigh(0),eigValLow(0){

    this->Mat = new float*[dim];
    for(int i=0;i<n;i++)
	{
		this->Mat[i]=new float [n];
	}
    for (int i=0;i<n;i++)
	{
		for (int j=0;j<n;j++)
		{
			this->Mat[i][j]=x;
		}
	}

}
// destructer
Matrix::~Matrix () {
	for(int i=0;i<dim;i++)
	{
		delete[] Mat[i];
	}
	delete[] Mat;	
}

// copy constructer
Matrix::Matrix(const Matrix& M){
	this->dim=M.dim;
	this->eigValHigh=M.eigValHigh;
	this->eigValLow=M.eigValLow;
	this->eigVecLow=M.eigVecLow;
	this->eigVecHigh=M.eigVecHigh;


    this->Mat = new float*[dim];
    for(int i=0;i<dim;i++)
	{
		this->Mat[i]=new float [dim];
	}
    for (int i=0;i<dim;i++)
	{
		for (int j=0;j<dim;j++)
		{
			this->Mat[i][j]=M.Mat[i][j];
		}
	}

}

// copy assignment
Matrix& Matrix:: operator= (const Matrix& M){
	// delete allacoted memory
	for(int i=0;i<dim;i++)
	{
		delete[] Mat[i];
	}
	delete[] Mat;
	
	// assign dimension of M to the matrix	
	this->dim=M.dim;
	this->eigValHigh=M.eigValHigh;
	this->eigValLow=M.eigValLow;
	this->eigVecLow=M.eigVecLow;
	this->eigVecHigh=M.eigVecHigh;
	// re-allocate memory that matches matrix M
	this->Mat = new float*[dim];
    for(int i=0;i<dim;i++)
	{
		this->Mat[i]=new float [dim];
	}
    for (int i=0;i<dim;i++)
	{
		for (int j=0;j<dim;j++)
		{
			this->Mat[i][j]=M.Mat[i][j];
		}
	}

	return *this;
}

// Prints values of matrix
void Matrix::print(){
    
	cout << "print called" << std::endl;
    
	for(int i=0;i<dim;i++)
	{
		for(int j=0;j<dim;j++)
		{
			cout<<this->Mat[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
}

// Prints eigenvalues and eigenvectors. Used for debugging purposes
void Matrix::printEig(){
    
	cout << "print eig called" << std::endl;
	cout << "High EigenValue is: ";
    cout << this->eigValHigh<<endl;
    cout << "High eigenVector is: "<< endl;
    this->eigVecHigh.print();
    cout << "Low EigenValue is: ";
    cout << this->eigValLow<<endl;
    cout << "Low EigenVector is: "<< endl;
    this->eigVecLow.print();
    

	cout<<endl;
}

// output to a file
void Matrix:: outToFile(const char* s){
	ofstream file_x (s);	
	if (file_x.is_open())
  	{	
		file_x<<"Eigenvalue#1: "<<eigValHigh<<"\n";
  	  //Writes values of corresponding eigenvector to file
    	for (int i=0; i<dim; i++)
    	{
    		file_x << eigVecHigh.Vec[i]<<"\n";
		}
		file_x<<"Eigenvalue#2: "<<eigValLow<<"\n";
  	  //Writes values of corresponding eigenvector to file
    	for (int i=0; i<dim; i++)
    	{
    		file_x << eigVecLow.Vec[i]<<"\n";
		}
    	file_x.close();
  	}
  	else
	{
		cout << "Unable to open output file x.txt";
	
	} 
}

// elemenwise matrix addition
Matrix Matrix::operator+(const Matrix& A){
	if (dim== A.dim)
	{
		Matrix temp(0.0,dim);
	
		for(int i=0;i<dim;i++)
		{
			for (int j=0;j<dim;j++)
			{
				temp.Mat[i][j]=this->Mat[i][j]+A.Mat[i][j];
			}
    	
    	}
    	return temp;
	}
	else
	{	
		Matrix temp;
		cout<<"Matrix dimensions does not match!";
		return temp;
	}
}


// matrix vector product
Vector Matrix::operator*(const Vector& V){
// this function assumes dimensions match for multiplication
	Vector result(0.0,dim);
	for(int i=0;i<dim;i++)
	{
		for(int j=0;j<dim;j++)
		{
			result.Vec[i]=result.Vec[i]+this->Mat[i][j]*V.Vec[j];
		}
	}
	return result;
}

// Exchange rows of the matrix
void Matrix:: MatRowExc(int row1, int row2, int index)
{
/*	
  This function exchanges values of rows with index greater then given index of a column.
  !! Does not exchange whole rows for computational efficiency since it assumes
  !!values with index smaller than column index are already 0 for both rows
  @param row1 First row to be exchanged
  @param row2 Second row to be exchanged
  @param index The current column position of gaussian elimination
*/
  for (int i=index; i<dim; i++)		// loops over columns after given index
  {
  	float temp= Mat[row1][i];
  	Mat[row1][i]= Mat[row2][i];
  	Mat[row2][i]= temp;
  }
  

}

Vector Matrix::solve(const Vector& V){
/*
	This function solves linear equation  M*x=v for given vector V and outputs vector x.
	First Upper trianguler matrix is obtained with LU factorization.
	Then equations is solved with back subtitution.
*/

//copy constructer to not to change values of current matrix and given vector
Matrix A(*this);
Vector b(V);
int n=dim;

for (int j=0; j<n-1; j++)
  {
  	float max= abs(A.Mat[j][j]);
  	int row=j;						// row index to be exchanged if necessary
	for(int i =j; i<n; i++)			// finds maximum value below diagonal in current column
	  {
  		if(abs(A.Mat[i][j])> max)
		  {
  			row= i;
  			max= A.Mat[i][j];
		  }
	  }

	if (row !=j)
	  {
	  	A.MatRowExc( j, row, j); // Exchanges rows of matrix A for assigning maximum value in a column to the diagonal
	  	b.VecRowExc(j, row);		 // Apply same exchange to b 
	  }
	  
	if (abs(A.Mat[j][j])<eps)			// Checks singularity with predefined precision and terminates if matrix is singular
	  {
	  	cout<<"A is a singular matrix!"<<endl;
	  	cout<<"Inverse power iteration does not converge"<<endl;
	  	exit(0);
	  }
	
	float *m= new float[n-j-1];		// Dynamic allocated array to hold multipliers for current column
	int k;
	for(k=0; k<n-j-1; k++)
	  {
		m[k]= A.Mat[k+j+1][j]/A.Mat[j][j];	// Compute multipliers for current column
	  }  
	for (int i=j+1; i<n; i++)		// Apply transformation to remaning sub matrix.
	  {								// Current column j is not updated for computational efficiency since solution assumes all sub diagonal entries are 0.
		for(int k=j+1; k<n; k++)
		  {
			A.Mat[i][k]= A.Mat[i][k]-m[i-j-1]*A.Mat[j][k];
		  }
		b.Vec[i]=b.Vec[i]-m[i-j-1]*b.Vec[j];	// Apply same transformation to vector b
	  }
	

	delete[] m;						// remove storage
  }
if (abs(A.Mat[n-1][n-1])<eps)			// Checks singularity for last pivot since algorithm does not go over last column
  {
	cout<<"A is a singular matrix!"<<endl;
	cout<<"Inverse power iteration does not converge"<<endl;
	exit(0);
  }

/* After LU factorization is complete, result vector is obtained using back substitution */

float *x= new float[dim];


 int j;
 for(j=n-1;j>=0;j--)			// Loop backwards over columns of A
 {
 	x[j]= b.Vec[j]/A.Mat[j][j]; 		// Compute solution component
 	
	for (int i=0; i<j; i++)
	{
		b.Vec[i]=b.Vec[i]-A.Mat[i][j]*x[j];	// Update all components of b using the last calculated value of solution x[]
	}
 }
 // creates a vector object for given values of x
 Vector res(x,dim);
 return res;	

}



//Highest eigenvalue calculator
void Matrix::highEigen(){
	// Random vector is created for iteration
	Vector temp(0.0,1.0,dim) ;
	// Result vector
	Vector result(0.0,dim);
	// Vector to test convergence condition
	Vector compare(0.0,dim);
	int flag=0; // to test divergence
	int count=0;	// to count iteration number
	do
	{
		result=(*this)*temp;
		// result is normalized
		temp=result/result.norm;
		eigValHigh=result.norm;
		
		// Divergence condition is checked
		if (compare.norm<(((*this)*temp)-(temp*eigValHigh)).norm || count>1000)
		{	
			flag++;
			/* Function accepted as diverging if calculated norm in this iteration is not smaller than last iteration
			for successive 5 iteration. Because due to machine precision, vector norm could randomly get larger 
			*/
			if(flag==5)
			{	
				cout<<"Normalized power iteration does not converge"<<endl;
				cout<<"Matrix does not have a dominant real eigenvector"<<endl;
				
				break;
			}

		}
		else
		{	
			flag=0;			
		}
		count++;
		// compare vector is measurment of convergece
		compare=((*this)*temp)-(temp*eigValHigh);
	}
	//
	while(compare.norm>tolerance && (((*this)*temp)+(temp*eigValHigh)).norm > tolerance|| count>1000);
	eigVecHigh=temp;
}


//Lowest eigenvalue calculator

void Matrix::lowEigen(){
	// Random vector is created for iteration
	Vector temp(0.0,1.0,dim) ;
	// Result vector
	Vector result(0.0,dim);
	// Vector to test convergence condition
	Vector compare(0.0,dim);
	int flag=0; // to test divergence
	int count=0;	// to count iteration number
	do
	{	
	
		result=this->solve(temp);
		// result vector is normalized
		temp=result/result.norm;
		eigValLow=result.norm;

		if (compare.norm<(((*this)*temp)*eigValLow-temp).norm )
		{	
			/* Function accepted as diverging if calculated norm in this iteration is not smaller than last iteration
			for successive 5 iteration. Because due to machine precision, vector norm could randomly get larger 
			*/
			flag++;
			if(flag==5){
				cout<<"Inverse power iteration does not converge"<<endl;				
				cout<<"Inverse of the matrix does not have a dominant eigenvector"<<endl;
				break;
			}

		}
		else{
			flag=0;
		}
		count++;
		// compare vector is measurment of convergece
		compare=((*this)*temp)*eigValLow-temp;
		
	}
	//
	while(compare.norm>tolerance and (((*this)*temp)*eigValLow+temp).norm>tolerance);
	eigValLow=1/result.norm;
	eigVecLow=temp;
}


/************ Class Matrix Definitions END *******************/


/************* Class Vector Definitions **********************/
// Default constructer
Vector::Vector():dim(0),norm(0.0){
	Vec= 0;
}

// Default constructer with pointer
Vector::Vector(float *V, int n):dim(n){
    this->Vec = V;
	infNorm();
}

// Default constructer with element value
Vector::Vector(float x, int n):dim(n){
    this->Vec = new float[dim];

    for (int i=0;i<n;i++)
	{
		this->Vec[i]=x;
	}
	infNorm();
}

// Default constructer with element value
Vector::Vector(float min,float max, int n):dim(n){
    this->Vec = new float[dim];

    for (int i=0;i<n;i++)
	{	
		float r=rand_FloatRange(min,max);
		this->Vec[i]=r;
	}
	infNorm();
}

// destructer
Vector::~Vector () {
	delete[] Vec;	
}

// copy constructer
Vector::Vector(const Vector& V){
	this->dim=V.dim;
	this->norm=V.norm;
    this->Vec = new float[dim];

    for (int i=0;i<dim;i++)
	{
		this->Vec[i]=V.Vec[i];
	}

}

//copy assignment
Vector& Vector:: operator= (const Vector& V){
	// delete allacoted memory
	delete[] Vec;
	
	// assign dimension of M to the matrix	
	this->dim=V.dim;
	// re-allocate memory that matches matrix M
	this->Vec = new float[dim];
	for (int i=0;i<dim;i++)
	{
		this->Vec[i]=V.Vec[i];
	}
	infNorm();

	return *this;
}


// Prints values of vector
void Vector::print(){
    
{
	for(int i=0;i<dim;i++)
		{
			cout<<Vec[i]<<endl;
		}
	cout<<endl;
}

}

// scaling elements with a scalar
Vector Vector::operator*(float x){

	Vector temp(0.0,dim);
	
	for(int i=0;i<dim;i++)
	{
		temp.Vec[i]=this->Vec[i]*x;	
    }
    temp.infNorm();
    return temp;
}

// divinding elements to a scalar
Vector Vector::operator/(float x){

	Vector temp(0.0,dim);
	
	for(int i=0;i<dim;i++)
	{
		temp.Vec[i]=this->Vec[i]/x;	
    }
    temp.infNorm();
    return temp;
}


// elementwise substraction
Vector Vector::operator-(const Vector& V){
	if(dim== V.dim)
	{
		Vector temp(0.0,dim);
		for(int i=0;i<dim;i++)
		{
			temp.Vec[i]=this->Vec[i]-V.Vec[i];	
    	}
    	temp.infNorm();
    	return temp;
	}
	else 
	{
		cout<<"Vector dimensions should match"<<endl;
	}
}

// elementwise addition
Vector Vector::operator+(const Vector& V){
	if(dim== V.dim)
	{
		Vector temp(0.0,dim);
		for(int i=0;i<dim;i++)
		{
			temp.Vec[i]=this->Vec[i]+V.Vec[i];	
    	}
    	temp.infNorm();
    	return temp;
	}
	else 
	{
		cout<<"Vector dimensions should match"<<endl;
	}
}

// infinity norm calculator
void Vector::infNorm()
{
	float max=0;
	for (int i=0;i<dim;i++)
	{
		if(max<abs(Vec[i]))
		{
			max=abs(Vec[i]);
		}
	}
	this->norm=max;
}


void  Vector::VecRowExc( int row1, int row2)
{
/*
  This function exchanges 2 values of an array.
   
  @param row1 First value to be exchanged
  @param row2 Second value to be exchanged
*/
  float temp = Vec[row1];
  Vec[row1] = Vec[row2];
  Vec[row2] = temp;
    
}


/************* Class Vector Definitions END *******************/




