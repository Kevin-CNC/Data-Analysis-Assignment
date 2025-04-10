"""By Kevin Caplescu; Handle and work with data from CSV files.
Our current targets:
- Use Pandas for data cleaning, transformation, and aggregation. 
- Apply NumPy for numerical operations. 
- Perform exploratory data analysis (EDA). 
- Generate meaningful Visualisations using Matplotlib and Seaborn. 
"""
# Importing all required modules.
import numpy as NP
import matplotlib.pyplot as PLT
import seaborn as SNS
import pandas as PND        
    

# Loads up messy datasheet and cleans it
def readCsvAndCleanData(csvPath:str):
    messy_df = PND.read_csv(csvPath)
    
    messy_df.drop_duplicates(inplace=True) # removing all duplicate rows found in the file.
    
    # We will operate on columns from left to right:
    # Age -> Gender -> Sales -> Profits -> Date -> Category -> Discount
    
    ################
    # AGE OPERATIONS
    messy_df["Age"] = PND.to_numeric(messy_df["Age"])
    
    AgeMedian = NP.median(messy_df["Age"].dropna()) # We get the median ages with the current ages that we have and use it to fill in blanks
    messy_df["Age"] = messy_df["Age"].fillna(AgeMedian) 
    ###################
    # GENDER OPERATIONS
    
    # Since we cannot assume everyone is just male or female since we have no reliable way of checking gender-neutral names (e.g Charlie)
    # we will simply replace them with "Other"
    
    messy_df["Gender"] = messy_df["Gender"].fillna("Other") # replaces 'nan' with Other
    messy_df["Gender"] = messy_df["Gender"].replace("","Other") # replaces empty strings with Other
    
    ###################
    # SALES OPERATIONS
    
    # tries to convert all types to numerical; if can't then makes them NaN
    messy_df["Sales"] = PND.to_numeric(messy_df["Sales"], errors='coerce')
    
    salesMedian = NP.median(messy_df["Sales"].dropna())
    messy_df["Sales"] = messy_df["Sales"].fillna(salesMedian)
    ####################
    # PROFIT OPERATIONS
    
    # tries to convert all types to numerical; if can't then makes them NaN
    messy_df["Profit"] = PND.to_numeric(messy_df["Profit"], errors='coerce')
    
    profitsMedian = NP.median(messy_df["Profit"].dropna())
    messy_df["Profit"] = messy_df["Profit"].fillna(profitsMedian)
    ####################
    # DATE OPERATIONS
    
    # we must make sure all dates are converted properly & handles invalid dates
    messy_df["Date"] = PND.to_datetime(messy_df["Date"], errors='coerce')

    # to keep as much data as possible, we will use a modal date to fill in the blanks
    modal_date = messy_df["Date"].mode()[0] # gets the most common date in the column
    
    # we fill in the NaT values with the modal date
    messy_df["Date"] = messy_df["Date"].fillna(modal_date)
    ######################
    # CATEGORY OPERATIONS
    
    # due to not knowing the category of the items, we will simply drop the row as we cannot deduce many outcomes from just the price / discounts alone
    messy_df = messy_df.dropna(subset=["Category"])
    
    ######################
    # DISCOUNT OPERATIONS
    
    # we handle nan and non-numeric values
    messy_df["Discount"] = PND.to_numeric(messy_df["Discount"], errors='coerce') # converts "N/A" to NaN & converts strings to numericals
    messy_df["Discount"] = messy_df["Discount"].fillna(0) # fills NaN with 0
    

    ###### CLEANING OUTLIERS #########
    # we are tasked to remove outliers using interquartile ranges
    ranges = {
        "lower-range":messy_df["Sales"].quantile(1/4),
        "higher-range":messy_df["Sales"].quantile(3/4)
    }
    
    interQuartRange = ranges["higher-range"] - ranges["lower-range"]
    upperB,lowerB = [ranges["higher-range"] + 1.5 * interQuartRange , ranges["lower-range"] - 1.5 * interQuartRange]
    
    # makes sure that the dataframe includes only rows with no outlying sales
    messy_df = messy_df[ ~(( messy_df["Sales"] < lowerB) | (messy_df["Sales"] > upperB)) ]


    # returns the cleaned up dataframe
    return messy_df
    
    
def performDataWrangling(cleaned_df:PND.DataFrame):
    # show grouping & aggregation by creating the new "Profit Margin" column & grouping discounts by types of products sold.
    cleaned_df["Profit Margin"] = cleaned_df["Profit"] / cleaned_df["Sales"]
    
    # groups discounts by their associated 'category' type
    discountByProducts = cleaned_df.groupby("Category")["Discount"].sum().reset_index()
    print(discountByProducts)
    
    # lambda application by classifying age groups
    # 25+ = Mature ; <25 = Not Mature
    cleaned_df["Mature Status"] = cleaned_df["Age"].apply(lambda x: "Mature" if x >= 25 else "Not Mature")
    
    return cleaned_df


def performEDA(cleaned_df:PND.DataFrame):
    # in this case we will perform EDA to see the correlation between categories, discounts and sales.
    
    # first, we print a summary:
    print(cleaned_df.describe())
    
    # we then perform correlation analysis between discounts & sales
    print("Correlation matrix:")
    corrMatrix = cleaned_df[["Sales", "Discount"]].corr()
    print(corrMatrix)
    
    
    # use of pivot table 
    pivot = cleaned_df.pivot_table(
        values="Sales", 
        index="Category", 
        columns="Discount", 
        aggfunc="mean"
    ).round(2)
    print("\nAverage Sales by Category and Discounts:")
    print(pivot)
    
    
    # finds out the average discount & sale amount per category
    avgGroupedDiscAndSales = cleaned_df.groupby("Category")[["Discount", "Sales"]].mean().round(2) #we round to 2 figures as sales do require it
    print("Average Discount and Sales per Category:")
    print(avgGroupedDiscAndSales)

    
    # display this as scatterplot
    PLT.figure(figsize=(10, 6))
    PLT.title("Discount vs Sales by Category (SCATTERPLOT)")
    PLT.xlabel("Discount (%)")
    PLT.ylabel("Sales (Units)")
    SNS.scatterplot(data=cleaned_df, x="Discount", y="Sales", hue="Category", alpha=0.7)
    PLT.legend(title="Category")
    PLT.grid(True)
    PLT.show()
    
    # now display this as boxplot 
    PLT.figure(figsize=(8, 5))
    SNS.boxplot(data=cleaned_df, x="Category", y="Discount")
    PLT.title("Discount Distribution by Category (BOXPLOT)")
    PLT.ylabel("Discount (%)")
    PLT.grid(True)
    PLT.show()
    
    
    # extra graphs to satisfy assignment requirements (already implemented scatter and boxplot)
    # histogram:
    PLT.hist(cleaned_df["Discount"], bins=20, color="firebrick") 
    PLT.title("Discounts distribution") 
    PLT.xlabel("Discount (%)") 
    PLT.ylabel("Frequency of discount (Units)") 
    PLT.show()
    
    
if __name__ == "__main__":
    cleaned = readCsvAndCleanData("file.csv")
    cleaned = performDataWrangling(cleaned)
    performEDA(cleaned)
    cleaned.to_csv("cleanFile.csv", index=False)