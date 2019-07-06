## Machine-Learning-Implementation

An implementation of a complete machine learning solution in Python on a real-world dataset. 

## Problem statement :-

The project is based on operations of a distribution center. The goal of this project is build a model that will predict how long it will take to complete the put-away operaition at the DC based on the inputs.

## Data collection, Cleaning & preparation :-

Data was collected over a month by the receivers present at the DC who are unloading trucks as they come in. After that, data was entered into Excel. Analyzed the data to look for missing values and any other discrepancies. Cross-checked the data from SAP and removed data that were not matching from SAP.  

## Model building :-

This is a regression problem since I am trying to predict the time taken which is a numeric and a continuous variable.
I wanted to start small initially, so I started with one regressor varible. I.e. Just the input variable is justntotal number The input variables are the number of different types of products that come in the trucks.

Here is a sample of the data: (Values shownn are just for illustration puposes)

| No. of rolls in truck |  Total time (mins) |
| --------------------- | ------------------ |         
|          30           |        120         |
|          50           |        76          |

Using the above collected data, built the folllowing machine learning models, as we can see above, and evaluated. Used Mean-sq-error to evaluate.


#### List of models built (part1):

CV: Cross-validation
Sig: Significant
Not Sig: Not significant
NA: Not applicable
R-sq: R-square value
Adj. R-sq: Adjusted R-square value
MSE: Mean-square error

|  Case |                                                      | Const |  x  |   x^2   |  R-Sq  | Adj. R-sq |   MSE   |
|:-:|------------------------------------------------------|:-----:|:---:|:-------:|:------:|:---------:|:-------:|
| 1 | Linear regression (without constant, without CV)     |   NA  | Sig |    NA   | 88.00% |   87.80%  |  2143.7 |
| 2 | Linear regression (without constant, with CV)        |   NA  | Sig |    NA   | 86.50% |   86.30%  | 2303.14 |
| 3 | Linear regression (with constant, without CV)        |  Sig  | Sig |    NA   | 42.70% |   42.00%  | 1611.25 |
| 4 | Linear regression (with constant, with CV)           |  Sig  | Sig |    NA   | 41.60% |   40.70%  | 1635.02 |
| 5 | Polynomial regression (without constant, without CV) |  Sig  | Sig | Not sig | 43.20% |   41.80%  |  1597.8 |
| 6 | Polynomial regression (without constant, with CV)    |  Sig  | Sig | Not sig | 41.90% |   40.10%  |  1627.5 |
| 7 | Polynomial regression (with constant, without CV)    |  Sig  | Sig | Not sig | 43.20% |   41.80%  |  1597.8 |
| 8 | Polynomial regression (with constant, with CV)       |  Sig  | Sig | Not sig | 41.90% |   40.10%  |  1627.5 |
| 9 | Random forest                                        |       |     |         | 60.70% |           |  912.7  |

As you can see, I built models with permutation and combinations of:
1) With, Without adding constant term in the model
2) Cross-validation

##### After building and evaluating th metrics, I decided to add more regressors and build models once again.

#### List of models built (part2):

| 6To25 | 25to40 | 40+ | 28in  |  Total time (mins) |
|-------|--------|-----|-------|--------------------|         
|  0    |   48   |  0  |  0    |        120         |
|  0    |    0   | 24  |  0    |        76          |

The first four columns are the different products that comes in the truck and the last column is the Y variable.

| Case |                                          	|   	| Const 	|  x1 	|    x2   	|    x3   	|  x4 	|   R-sq  	| Adj R-sq 	|   MSE   	| Accuracy 	| Avg. error 	|   	|
|----	|------------------------------------------	|---	|:-----:	|:---:	|:-------:	|:-------:	|:---:	|:-------:	|:--------:	|:-------:	|:--------:	|:----------:	|:-:	|
| 10 	| MLR (without constant, without CV)       	|   	|   NA  	| Sig 	|   Sig   	|   Sig   	| Sig 	|  89.10% 	|  88.50%  	| 1945.65 	|          	|            	|   	|
| 11 	| MLR (without constant, with CV)          	|   	|   NA  	| Sig 	|   Sig   	|   Sig   	| Sig 	|  87.80% 	|  87.00%  	| 2090.66 	|          	|            	|   	|
| 12 	| MLR (with constant, without CV)          	|   	|  Sig  	| Sig 	|   Sig   	|   Sig   	| Sig 	|  44.70% 	|  41.90%  	|   1555  	|          	|            	|   	|
| 13 	| MLR (with constant, with CV)             	|   	|  Sig  	| Sig 	| Not sig 	| Not sig 	| Sig 	|  42.90% 	|  39.30%  	|   1598  	|          	|            	|   	|
| 14 	| MLR (Removing insignificant variables)   	|   	|  Sig  	| Sig 	|    NA   	|    NA   	| Sig 	|  38.40% 	|  36.50%  	|   1724  	|          	|            	|   	|
| 15 	| Random forest                            	|   	|       	|     	|         	|         	|     	| 18.014% 	|          	|   1906  	|  71.11%  	|   34.1974  	|   	|
| 16 	| ANN                                      	|   	|       	|     	|         	|         	|     	| -70.00% 	|          	|   4609  	|          	|            	|   	|
| 17 	| SVR - Linear kernel                      	|   	|       	|     	|         	|         	|     	|  50.00% 	|          	|   1503  	|  74.59%  	|   26.817   	|   	|
|    	| SVR - Rbf Kernel                         	|   	|       	|     	|         	|         	|     	|  -0.90% 	|          	|   2507  	|          	|            	|   	|
| 18 	| Ridge regression                         	|   	|       	|     	|         	|         	|     	|  44.71% 	|          	|   1555  	|          	|            	|   	|
| 19 	| Ridge regression - With cross validation 	|   	|       	|     	|         	|         	|     	|  33.24% 	|          	|   2419  	|          	|            	|   	|
| 20 	| Random forest - parameter tuning         	|   	|       	|     	|         	|         	|     	|  18.00% 	|          	|   1915  	|  69.32%  	|    34.85   	|   	|
|    	|                                          	|   	|       	|     	|         	|         	|     	|   20%   	|          	|  1850.9 	|  68.32%  	|   35.143   	|   	|
|    	|                                          	|   	|       	|     	|         	|         	|     	|  21.00% 	|          	|   1834  	|  70.11%  	|    34.71   	|   	|
|    	|                                          	|   	|       	|     	|         	|         	|     	|  12.00% 	|          	|   2038  	|  68.28%  	|   36.0341  	|   	|


| Case|                                       	| Parameter tuning 	|   Rsq  	|   	|   MSE   	|   MAE  	|  MAPE  	| Accuracy 	|
|----	|---------------------------------------	|:----------------:	|:------:	|:-:	|:-------:	|:------:	|:------:	|:--------:	|
| 24 	| SVR                                   	|   Before tuning  	|  4.39% 	|   	| 2162.87 	|  34.4  	| 36.30% 	|  63.70%  	|
|    	|                                       	|        HP1       	| 21.00% 	|   	|   1781  	|  32.4  	| 33.40% 	|  66.56%  	|
|    	|                                       	|        HP2       	| 27.90% 	|   	|  1629.9 	|  29.67 	| 29.38% 	|  70.62%  	|
|    	|                                       	|        HP3       	| 30.96% 	|   	| 1561.92 	|  35.1  	| 34.21% 	|  65.78%  	|
|    	|                                       	|        HP4       	| 32.90% 	|   	| 1516.98 	|  33.2  	| 33.23% 	|  66.80%  	|
|    	|                                       	|        HP5       	| 35.32% 	|   	|   1463  	|  33.13 	| 32.43% 	|  67.56%  	|
|    	|                                       	|        HP6       	| 32.90% 	|   	| 1516.98 	|  33.2  	| 33.23% 	|  66.77%  	|
|    	|                                       	|        HP7       	| 35.50% 	|   	| 1459.18 	|  33.98 	| 33.45% 	|  66.55%  	|
|    	|                                       	|        HP8       	| 35.34% 	|   	| 1462.63 	|  33.47 	| 33.10% 	|  66.91%  	|
|    	|                                       	|        HP9       	| 35.35% 	|   	| 1462.63 	|  33.47 	| 33.09% 	|  66.91%  	|
|    	|                                       	|                  	|        	|   	|         	|        	|        	|          	|
| 25 	| ANN                                   	|                  	| 39.82% 	|   	|   1361  	|  29.19 	| 25.96% 	|  74.04%  	|
|    	|                                       	|        HP1       	| -3.50% 	|   	|   2342  	|  39.25 	| 34.97% 	|  65.02%  	|
|    	|                                       	|        HP2       	|        	|   	|   2362  	|  38.38 	| 32.92% 	|  67.07%  	|
|    	|                                       	|                  	|        	|   	|         	|        	|        	|          	|
| 27 	| Gradient boosting                     	|                  	|  6.20% 	|   	|   2121  	|  33.48 	| 26.04% 	|  73.95%  	|
|    	|                                       	|        HP1       	| 24.40% 	|   	|   1709  	|  32.16 	| 29.83% 	|  70.16%  	|
|    	|                                       	|        HP2       	| 22.16% 	|   	|   1761  	|  33.91 	| 32.08% 	|  67.91%  	|

## Results : 

#### From the results above, the ANN - Artificial Neural network gave the leat MSE and an accuracy of 75%.
