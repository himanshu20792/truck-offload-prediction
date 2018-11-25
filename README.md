## Machine-Learning-Implementation

An implementation of a complete machine learning solution in Python on a real-world dataset. 

## Problem statement :-

The project is based on operations of a distribution center. The goal of this project is build a model that will predict how long it will take to complete the put-away operaition at the DC based on the inputs.

## Data collection, Cleaning & preparation :-

Data was collected over a month by the receivers present at the DC who are unloading trucks as they come in. After that, data was entered into Excel. Analyzed the data to look for missing values and any other discrepancies. Cross-checked the data from SAP and removed data that were not matching from SAP.  

## Model building :-

This is a regression problem since I am trying to predict the time taken which is a numeric and a continuous variable.
I wnated to start small initially, so I started with one regressor varible. I.e. Just the input variable is justntotal number The input variables are the number of different types of products that come in the trucks.

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

|   |                                                      |   |   | Const | x   | x^2     | R-Sq   | Adj. R-sq | MSE     |
|---|------------------------------------------------------|---|---|-------|-----|---------|--------|-----------|---------|
| 1 | Linear regression (without constant, without CV)     |   |   | NA    | Sig | NA      | 88.00% | 87.80%    | 2143.7  |
| 2 | Linear regression (without constant, with CV)        |   |   | NA    | Sig | NA      | 86.50% | 86.30%    | 2303.14 |
| 3 | Linear regression (with constant, without CV)        |   |   | Sig   | Sig | NA      | 42.70% | 42.00%    | 1611.25 |
| 4 | Linear regression (with constant, with CV)           |   |   | Sig   | Sig | NA      | 41.60% | 40.70%    | 1635.02 |
| 5 | Polynomial regression (without constant, without CV) |   |   | Sig   | Sig | Not sig | 43.20% | 41.80%    | 1597.8  |
| 6 | Polynomial regression (without constant, with CV)    |   |   | Sig   | Sig | Not sig | 41.90% | 40.10%    | 1627.5  |
| 7 | Polynomial regression (with constant, without CV)    |   |   | Sig   | Sig | Not sig | 43.20% | 41.80%    | 1597.8  |
| 8 | Polynomial regression (with constant, with CV)       |   |   | Sig   | Sig | Not sig | 41.90% | 40.10%    | 1627.5  |
| 9 | Random forest                                        |   |   |       |     |         | 60.70% |           | 912.7   |

As you can see, I built models with permutation and combinations of:
1) With, Without adding constant term in the model
2) Cross-validation

##### After building and evaluating th metrics, I decided to add more regressors and build models once again.

| 6To25 | 25to40 | 40+ | 28in  |  Total time (mins) |
|-------|--------|-----|-------|--------------------|         
|  0    |   48   |  0  |  0    |        120         |
|  0    |    0   | 24  |  0    |        76          |

The first four columns are the different products that comes in the truck and the last column is the Y variable.

#### List of models built (part2):


