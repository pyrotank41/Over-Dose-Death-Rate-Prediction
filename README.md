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
In addition to the feature columns provided by the CDC Over Dose Death dataset, other datasets might be useful in predicting the OD Death rate per state, specifically we look at two additional feature datasets:

### Weather Dataset:
Over dose deaths might be influenced by the weather: the colder, gloomier months might contribute to an increase in over dose deaths. We download the weather data from NOAA to use as additional features in our project <https://www.ncdc.noaa.gov/cdo-web/datasets>

### Rent to Income Dataset:
Another dataset that might contribute to the over dose death rate is the rent to income ratio data. From a psychology point of view, socio-economic standing has an influence in how people deal with stressors in life. The worse the socio-economic standing, the more over dose deaths can be expected. We download the income by zipcode data from <https://www.incomebyzipcode.com/>, and the rent by zipcode data from Redfin here <https://www.redfin.com/news/data-center/>
