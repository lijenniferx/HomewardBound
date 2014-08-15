def get_columns(dataframe):
    import pandas as pd
    import pickle as pk
       
    result = dataframe[dataframe.columns[~dataframe.columns.isin(['Status','ArrivalDate','AnimalID','LengthofStay'])]].get_values()
    columns =  dataframe[dataframe.columns[~dataframe.columns.isin(['Status','ArrivalDate','AnimalID','LengthofStay'])]].columns
    

    
    return result, columns
    
def get_columns_poisson(dataframe):
    import pandas as pd
    
    result = dataframe[dataframe.columns[~dataframe.columns.isin(['LengthofStay','ArrivalDate','AnimalID'])]].get_values()
    columns =  dataframe[dataframe.columns[~dataframe.columns.isin(['LengthofStay','ArrivalDate','AnimalID'])]].columns
    
    return result, columns
    