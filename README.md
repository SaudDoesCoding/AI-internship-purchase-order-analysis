# AI-internship-purchase-order-analysis
This repository analyses the data file that was provided from Aajil by categorizing items using clustering for English and Arabic descriptions of the items, and visualize spending patterns

1. Overview of the assesment
Aajil provided a data sheet that contained 3150 rows and 11 columns, rows representing the item characterstics such as Item ID, Item Name, Quantity, Total Bcy.. etc while the rows representing the numbers, the data sheet
needed cleaning and analysis to extrct spending insights for Aajil.

The project helps Aajil make a busniess decision whether its identifying trends or managing risks, essentially understanding the bigger picture not just technical tasks. 

The approach to this project was to identiy key patterns and features in the items to differentiate and create a clearer explanation of the data,
When visually inspecting the data, the item IDs are in English, Arabic, or a mix of both, which intially creates a mess,  while also it is a large dataset that is computationally heavy and takes a long time to excute. 
To overcome these challenges, a language detector library was installed showing that there are 1671 items were in English, 1230 in Arabic, and 9 were a mix of both. 
I assumed the items in both Arabic and English were not neccesary to include due to extremely low occurance as there was only 9 items out of 3150, and wanted to focus on the majority of the data while avoiding redundancy. 
To improve the processing speed and reduce memroy usage, I limited the maximum number of features during text vectorization. This trade off allowed faster experimentation and iteration while still capturing the important patterns. 

After importing the neccessary libraries, the data file was loaded. To clean the file, the rows with NaNs were dropped, and converting all the " Total bcy " columns to numerci and drop the ones that cant be converted. 
Additionally, The items were clustered by text using language detector commands and applying K-means clustering method to group similar items as well as adding a " cluster " column to indicate which cluster each item belongs to. 

Furthermore, adding seperate clusters for Arabic items and English items, 6 clusters each, and combining the results into one dataframe. 

To summarize the spend by cluster, a command block was added to calculate the total spending per cluster for Arabic and English items, and used another command line to represent the data visually. 
Moreover, to identify further patterns, the mean of the previous command line was neccessary to see which cluster is high-value. 
A detailed summary table per cluster was added to represent all the numbers per cluster visually for better understanding. A total of six plots per language was provided that shows cluster size, total spend, and average spend. 

Finally, showing the top clusters per language to identify highest spejnding cluster, and printing the top items per cluster. 

The overall purpose of the assesment was to clean and process multilingual purchase order dataset, detect the language, cluster similar items. analyze spending patterns per cluster which I have chosen to be the total spend, average spend per item, cluster size, and visualize results for easy insights. 

The K-Means method was chosen instead of other methods because it works very well with numeric representation of text, and it is best used when it needs to group similar items, efficient for large datsets, flexible and easy to interpert. 

Limitations may omit rare patterns. However, it captures the most common trends with significantly improving processing speed. 

Findings: 
The Arabic cluster with the highest spend corresponded to bulk supplies, while English cluster with highest spend included electronics. Highlighting key areas where Aajil's procurement budget is concentrated.

Future steps should include combining Arabic and English into one shared cluster, trying regression method to predict high-cost items before purchase. 


