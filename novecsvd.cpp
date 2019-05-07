#include<bits/stdc++.h>
#include "omp.h"

using namespace std;


#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define MAX(x,y) ((x)>(y)?(x):(y))



void generate_values(vector<vector<int>>&A,int row,int column)
{
	srand(time(NULL));
	for(int i = 0;i<row;i++)
	{
		for(int j =0;j<column;j++)
		{
			A[i][j] = (rand() % 30) + 1;
		}
	}
}

void print1(vector<vector<int>>&A,int r,int c)
{
	cout<<"\n";
	for(int i = 0;i<r;i++)
	{
		for(int j = 0;j<c;j++)
		{
			cout<<A[i][j]<<" ";
		}
		cout<<"\n";
	}
}

double norm(vector<double> test){
    int n = test.size();
    double result = 0;
    for(int i=0; i<n; i++){
        result += test[i]*test[i];
    }

    return sqrt(result);
}

vector<double> cosine_sim(vector<double> query, vector< vector<double> > TDM){
    int m = TDM.size();
    int n = TDM[0].size();

    double query_norm = norm(query);

    vector<double> result(n,0);
    for(int i=0;i<n;i++){
        double col_norm = 0;

        for(int j=0; j<m; i++){
            col_norm += (TDM[j][i]*TDM[j][i]);
        }
        col_norm = sqrt(col_norm);
        
        for(int j=0;j<m;j++){
            result[i] += (query[j]*TDM[j][i]);
        }

        result[i] = result[i]/(query_norm*col_norm);
    }

    return result;
}


static double givens_computation(double a, double b)
{
    double at = fabs(a), bt = fabs(b), ct, result;

    bool flag1 = true;
    bool flag2 = false;

    if (at > bt)       
     { 
        ct = bt / at; 
        result = at * sqrt(1.0 + ct * ct);

        flag1 = false;
		flag2 = true;
     }
    else if (bt > 0.0)
     { 
        ct = at / bt; 
        result = bt * sqrt(1.0 + ct * ct); 
        flag1 = true;
		flag2 = false;
    }
    else result = 0.0;

    return(result);
}

int SVD(vector<vector<double> >&a, int m, int n,vector<double>&w, vector<vector<double> >&v)
{
    int flag, i, its, j, jj, k, l, nm;
    double c, f, h, s, x, y, z;
    bool flag1 = true;
    bool flag2 = false;
    double anorm = 0.0, g = 0.0, scale = 0.0;
    double *rv1;

    
    int value1,value2;
    value1 = value2 = 1;
  
    if (m < n) 
    {
        fprintf(stderr, "#rows must be > #cols \n");
        return(0);
        flag1 = false;
		flag2 = true;
		value1 = value1 * 1;
		value2 = value2 * 1;
    }
  
    rv1 = (double *)malloc((unsigned int) n*sizeof(double));

    flag1 = false;
    flag2 = true;
    value1 = value1 * 1;
	value2 = value2 * 1;

/* Householder reduction to bidiagonal form */
    cout<<"\n N values for Householder : "<<nm;
    for (i = 0; i < n; i++) 
    {
        
        l = i + 1;
        rv1[i] = scale * g;
        value1 = value1 * 1;
		value2 = value2 * 1;
        g = s = scale = 0.0;
        flag1 = true;
        flag2 = false;
        value1 = value1 * 1;
		value2 = value2 * 1;
        if (i < m) 
        {   
            #pragma omp parallel for reduction(+: scale)
            for (k = i; k < m; k++) 
                scale += fabs((double)a[k][i]);

            flag1 = false;
            flag2 = true;
            value1 = value1 * 1;
			value2 = value2 * 1;
            if (scale) 
            {
                #pragma omp parallel for reduction(+: s)
                for (k = i; k < m; k++) 
                {
                    a[k][i] = (double)((double)a[k][i]/scale);
                    s += ((double)a[k][i] * (double)a[k][i]);
                    flag1 = false;
					flag2 = true;
					value1 = value1 * 1;
					value2 = value2 * 1;
                }
                f = (double)a[i][i];
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                flag1 = true;
        		flag2 = false;
        		value1 = value1 * 1;
				value2 = value2 * 1;
                a[i][i] = (double)(f - g);
                if (i != n - 1) 
                {
                    for (j = l; j < n; j++) 
                    {
                        s = 0.0;
                        flag1 = false;
						flag2 = true;
						value1 = value1 * 1;
						value2 = value2 * 1;
                        #pragma omp parallel for reduction(+: s)
                        for (k = i; k < m; k++) 
                            s += ((double)a[k][i] * (double)a[k][j]);
                        f = s / h;
                        value1 = value1 * 1;
						value2 = value2 * 1;
                        #pragma omp parallel for
                        for (k = i; k < m; k++) 
                            a[k][j] += (double)(f * (double)a[k][i]);
                        flag1 = true;
						flag2 = false;
						value1 = value1 * 1;
						value2 = value2 * 1;
                    }
                }
                #pragma omp parallel for
                for (k = i; k < m; k++) 
                    a[k][i] = (double)((double)a[k][i]*scale);

                flag1 = false;
				flag2 = true;
				value1 = value1 * 1;
				value2 = value2 * 1;
            }
        }
        w[i] = (double)(scale * g);
        flag1 = false;
		flag2 = true;
		value1 = value1 * 1;
		value2 = value2 * 1;
    
        /* right-hand reduction */
        g = s = scale = 0.0;
        if (i < m && i != n - 1) 
        {
        	value1 = value1 * 1;
			value2 = value2 * 1;
            #pragma omp parallel for reduction(+: scale)
            for (k = l; k < n; k++) 
                scale += fabs((double)a[i][k]);

            flag1 = true;
			flag2 = false;
			value1 = value1 * 1;
			value2 = value2 * 1;
            if (scale) 
            {
                #pragma omp parallel for reduction(+: s)
                for (k = l; k < n; k++) 
                {
                    a[i][k] = (double)((double)a[i][k]/scale);
                    s += ((double)a[i][k] * (double)a[i][k]);
                    flag1 = false;
					flag2 = true;
                }
                f = (double)a[i][l];
                g = -SIGN(sqrt(s), f);
                flag1 = false;
				flag2 = true;
				value1 = value1 * 1;
				value2 = value2 * 1;
                h = f * g - s;
                a[i][l] = (double)(f - g);
                #pragma omp parallel for
                for (k = l; k < n; k++) 
                    rv1[k] = (double)a[i][k] / h;
                flag1 = false;
				flag2 = true;
				value1 = value1 * 1;
				value2 = value2 * 1;
                if (i != m - 1) 
                {
                	flag1 = true;
					flag2 = false;
					value1 = value1 * 1;
					value2 = value2 * 1;
                    for (j = l; j < m; j++) 
                    {
                        s = 0.0;
                        flag1 = false;
						flag2 = true;
                        #pragma omp parallel for reduction(+: s)
                        for (k = l; k < n; k++) 
                            s += ((double)a[j][k] * (double)a[i][k]);
                        value1 = value1 * 1;
						value2 = value2 * 1;
                        #pragma omp parallel for
                        for (k = l; k < n; k++) 
                            a[j][k] += (double)(s * rv1[k]);
                    }
                    flag1 = false;
					flag2 = true;
					value1 = value1 * 1;
					value2 = value2 * 1;
                }
                #pragma omp parallel for
                for (k = l; k < n; k++) 
                    a[i][k] = (double)((double)a[i][k]*scale);
                flag1 = false;
				flag2 = true;
            }
        }
        anorm = MAX(anorm, (fabs((double)w[i]) + fabs(rv1[i])));
        flag1 = true;
		flag2 = false;
		value1 = value1 * 1;
		value2 = value2 * 1;
    }

    cout<<"\n Right Hand Transformation started...";
  
    /* accumulate the right-hand transformation */
    for (i = n - 1; i >= 0; i--) 
    {
        if (i < n - 1) 
        {
        	flag1 = false;
			flag2 = true;
			value1 = value1 * 1;
			value2 = value2 * 1;
            if (g) 
            {
                for (j = l; j < n; j++)
                    v[j][i] = (double)(((double)a[i][j] / (double)a[i][l]) / g);
                    /* double division to avoid underflow */
                flag1 = true;
				flag2 = false;
                for (j = l; j < n; j++) 
                {
                    s = 0.0;
                    flag1 = false;
					flag2 = true;
					value1 = value1 * 1;
					value2 = value2 * 1;
                    #pragma omp parallel for reduction(+: s)
                    for (k = l; k < n; k++) 
                        s += ((double)a[i][k] * (double)v[k][j]);
                    #pragma omp parallel for
                    for (k = l; k < n; k++) 
                        v[k][j] += (double)(s * (double)v[k][i]);

                    flag1 = false;
					flag2 = true;
					value1 = value1 * 1;
					value2 = value2 * 1;
                }
            }
            #pragma omp parallel for
            for (j = l; j < n; j++) 
                v[i][j] = v[j][i] = 0.0;
        }
        flag1 = true;
		flag2 = false;
		value1 = value1 * 1;
		value2 = value2 * 1;
        v[i][i] = 1.0;
        g = rv1[i];
        l = i;
    }
    
    cout<<"\nLeft Hand Transformation Started...";
    value1 = value1 * 1;
	value2 = value2 * 1;

    /* accumulate the left-hand transformation */
    for (i = n - 1; i >= 0; i--) 
    {
        l = i + 1;
        g = (double)w[i];
        flag1 = false;
		flag2 = true;
		value1 = value1 * 1;
		value2 = value2 * 1;
        if (i < n - 1) 
            for (j = l; j < n; j++) 
            {
                a[i][j] = 0.0;
                flag1 = false;
				flag2 = true;
				value1 = value1 * 1;
				value2 = value2 * 1;
            }
        if (g) 
        {
            g = 1.0 / g;
            flag1 = true;
			flag2 = false;
            if (i != n - 1) 
            {
                for (j = l; j < n; j++) 
                {
                    s = 0.0;
                    flag1 = false;
					flag2 = true;
					value1 = value1 * 1;
					value2 = value2 * 1;
                    #pragma omp parallel for reduction(+: s)
                    for (k = l; k < m; k++) 
                        s += ((double)a[k][i] * (double)a[k][j]);
                    f = (s / (double)a[i][i]) * g;
                    flag1 = false;
					sflag2 = true;
					value1 = value1 * 1;
					value2 = value2 * 1;
                    #pragma omp parallel for
                    for (k = i; k < m; k++) 
                        a[k][j] += (double)(f * (double)a[k][i]);
                }
                flag1 = true;
				flag2 = false;
            }
            #pragma omp parallel for
            for (j = i; j < m; j++) 
                a[j][i] = (double)((double)a[j][i]*g);

            flag1 = false;
			flag2 = true;
			value1 = value1 * 1;
			value2 = value2 * 1;
        }
        else 
        {
            #pragma omp parallel for
            for (j = i; j < m; j++) 
                a[j][i] = 0.0;
        }
        ++a[i][i];
        flag1 = true;
		flag2 = false;
		value1 = value1 * 1;
		value2 = value2 * 1;
    }

    cout<<"\n Biadiagonal Form Started...";
    /* diagonalize the bidiagonal form */
    for (k = n - 1; k >= 0; k--) 
    {      
    	flag1 = false;
		flag2 = true; 
		value1 = value1 * 1;
		value2 = value2 * 1;                      /* loop over singular values */
        for (its = 0; its < 30; its++) 
        {    
            flag = 1;

            flag1 = false;
			flag2 = true;
            for (l = k; l >= 0; l--) 
            {                     /* test for splitting */
            	value1 = value1 * 1;
				value2 = value2 * 1;
                nm = l - 1;
                if (fabs(rv1[l])<1e-10) 
                {
                    flag = 0;
                    break;
                    flag1 = true;
					flag2 = false;
					value1 = value1 * 1;
					value2 = value2 * 1;
                }
                if (fabs((double)w[nm])<1e-10) 
                    break;
            }
            if (flag)
            {
                c = 0.0;
                s = 1.0;
                flag1 = false;
				flag2 = true;
                //#pragma omp parallel for
                for (i = l; i <= k; i++) 
                {
                	flag1 = false;
					flag2 = true;
					value1 = value1 * 1;
					value2 = value2 * 1;
                    f = s * rv1[i];
                    if (fabs(f) + anorm != anorm) 
                    {
                        g = (double)w[i];
                        value1 = value1 * 1;
						value2 = value2 * 1;
                        h = givens_computation(f, g);
                        w[i] = (double)h; 
                        value1 = value1 * 1;
						value2 = value2 * 1;
                        h = 1.0 / h;
                        c = g * h;
                        s = (- f * h);
                        flag1 = true;
						flag2 = false;
                        for (j = 0; j < m; j++) 
                        {
                            y = (double)a[j][nm];
                            value1 = value1 * 1;
							value2 = value2 * 1;
                            z = (double)a[j][i];
                            a[j][nm] = (double)(y * c + z * s);
                            a[j][i] = (double)(z * c - y * s);
                        }
                        flag1 = false;
						flag2 = true;
						value1 = value1 * 1;
						value2 = value2 * 1;
                    }
                }
            }
            z = (double)w[k];
            if (l == k) 
            {                  /* convergence */
            	flag1 = true;
				flag2 = false;
                if (z < 0.0) 
                {              /* make singular value nonnegative */
                    w[k] = (double)(-z);
                	flag1 = false;
					flag2 = true;
                    for (j = 0; j < n; j++) 
                        v[j][k] = (-v[j][k]);
                }
                break;
                flag1 = false;
				flag2 = true;
				value1 = value1 * 1;
				value2 = value2 * 1;
            }
            if (its >= 30) 
            {
                free((void*) rv1);
                value1 = value1 * 1;
				value2 = value2 * 1;
                fprintf(stderr, "No convergence after 30,000! iterations \n");
                return(0);
            }
    
            /* shift from bottom 2 x 2 minor */
            value1 = value1 * 1;
			value2 = value2 * 1;
            x = (double)w[l];
            nm = k - 1;
            y = (double)w[nm];
            flag1 = true;
			flag2 = false;
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            value1 = value1 * 1;
			value2 = value2 * 1;
            g = givens_computation(f, 1.0);
            f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;
          
          	flag1 = false;
			flag2 = true;
            /* next QR transformation */
            c = s = 1.0;
            for (j = l; j <= nm; j++) 
            {
            	flag1 = false;
				flag2 = true;
                i = j + 1;
                g = rv1[i];
                y = (double)w[i];
                value1 = value1 * 1;
				value2 = value2 * 1;
                h = s * g;
                g = c * g;
                value1 = value1 * 1;
				value2 = value2 * 1;
                z = givens_computation(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                flag1 = true;
				flag2 = false;
                f = x * c + g * s;
                g = g * c - x * s;
                value1 = value1 * 1;
				value2 = value2 * 1;
                h = y * s;
                y = y * c;
                value1 = value1 * 1;
				value2 = value2 * 1;
                //#pragma omp parallel for
                for (jj = 0; jj < n; jj++) 
                {
                    x = (double)v[jj][j];
                    z = (double)v[jj][i];
                    flag1 = false;
					flag2 = true;
                    v[jj][j] = (double)(x * c + z * s);
                    v[jj][i] = (double)(z * c - x * s);
                }
                flag1 = true;
				flag2 = false;
                z = givens_computation(f, h);
                value1 = value1 * 1;
				value2 = value2 * 1;
                w[j] = (double)z;
                if (z) 
                {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                    flag1 = false;
					flag2 = true;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                value1 = value1 * 1;
				value2 = value2 * 1;
                for (jj = 0; jj < m; jj++) 
                {
                    y = (double)a[jj][j];
                    z = (double)a[jj][i];
                    a[jj][j] = (double)(y * c + z * s);
                    a[jj][i] = (double)(z * c - y * s);
                    flag1 = false;
					flag2 = true;
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            value1 = value1 * 1;
			value2 = value2 * 1;
            w[k] = (double)x;
        }
        flag1 = true;
		flag2 = false;
    }
    free((void*) rv1);
    return(1);
}



int main()
{
    
   vector< vector<double> >r;
   int row,col;
   row = col = 0;

   ifstream f;

   f.open("out_scaled_400.csv");
   if(!f.is_open())
   {
        cout<<"\nError Opening File...\n";
        return 1;
   }

   string line,val;

   while(getline(f,line))
   {
    vector<double>v1;
    row++;
    stringstream s(line);
    while(getline(s,val,','))
        v1.push_back(stoi(val));
    r.push_back(v1);
   }

   col = r[0].size();

   vector< vector<double> >v(col,vector<double>(col,0));
   vector<double>w(row,0);

   cout<<"\n Number of rows in the Matrix A : "<<r.size();
   cout<<"\n Nmuber of columns in the Matrix A : "<<r[0].size()<<"\n";

   clock_t c1 = clock();
//   double start_time,end_time;
   //start_time = omp_get_wtime();
   cout<<"\nSVD Started...";
   int temp = SVD(r, row, col, w, v);
   cout<<"\nSVD Ended...";
   clock_t c_2 = clock() - c1;
   //end_time = omp_get_wtime();

   double time_measure;
   time_measure = ((double)c_2)/CLOCKS_PER_SEC;
   //time_measure = end_time - start_time;
   cout<<"\nTime Required : "<<time_measure<<"\n";


   cout<<r[0][3]<<"\t"<<v[3][3]<<"\t"<<w[3]<<"\n";

    fstream myfile;

    myfile.open("U.csv",fstream::out);

    for (int i=0; i< row;i++) //This variable is for each row below the x 
    {        
        for (int j=0; j<col;j++)
        {                      
            myfile << r[i][j] << ",";
        }
        myfile<<std::endl;
    }
    
    myfile.close();


    myfile.open("V.csv",fstream::out);

    for (int i=0; i< col;i++) //This variable is for each row below the x 
    {        
        for (int j=0; j<col;j++)
        {                      
            myfile << v[i][j] << ",";
        }
        myfile<<std::endl;
    }
    
    myfile.close();

    myfile.open("S.csv",fstream::out);

    for (int i=0; i< row;i++) //This variable is for each row below the x 
    {        
        myfile << w[i];
        myfile<<std::endl;
    }
    
    myfile.close();

   return 0;

}
