# CS547 -- Deep Dive Project
# Project 6: Drug Overdose Deaths
Milestone 1: Due Oct 30, 2022:

## Team Members:
- Pranav Pamidighantam
- Assma Boughoula
- Yadu Reddy
- Karan Kochar

In this project, we will explore the Drug Overdose Death dataset and build a model to predict the rate of OD based on different features.

## Preprocessing
For the first milestone, we download the Provisional Drug Overdose Death Counts dataset from the CDC website <https://www.cdc.gov/nchs/nvss/vsrr/drug-overdose-data.htm> and perform some preprocessing steps to prepare the data for analysis.

Looking at the columns in the data file, we see that we have 11 features:
- State
- Year
- Month
- Period
- Indicator
- Data Value
- Percent Complete
- Percent Pending Investigation
- State Name
- Footnote
- Footnote Symbol

And the label column is:
- Predicted Value

Our goal for this project is to build a deep learning model to predict the Over Dose Death values represented by the column "Predicted Value" in the data, using the rest of the columns as features. 

## Clean the Data:
### Remove redundancies
The raw data includes redundant information such as "State" and "State Name", and complementary information such as "Percent Complete" and "Percent Pending Investigation." In this preprocessing step we clean the data by removing redundant information.

### Remove Null value data points
Some rows in the data don't have a label or "Predicted Value", these data points can't be used in either training or testing, and thus should be removed.

### Save the Data:
We use ```Pandas DataFrames``` to save the data as csv files, then pickle them.

## Additional Datasets:
In addition to the feature columns provided by the CDC Over Dose Death dataset, other datasets might be useful in predicting the OD Death rate per state, we determined a number of factors and found data sets for:

- Tax Rate
- Unemployment Rate
- Cost of Living Compared to the Average
- High School Graduation Rate
- GDP Per Capita
- Life Expectancy
- Average Age
- Poverty Rate
- Violent Crime Rate
- Population
## Statistical Analysis
### Correlation Scatter Plots
We charted the datasets against one another to see the correlation between the various factors.

### Linear Regression
We ran linear regression on the drug overdose deaths vs the additional datasets to find an initial baseline model. The model was compared to the neural net models to see if there were any improvements.
