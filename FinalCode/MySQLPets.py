def csv_to_mysql(filename, tablename):
    '''reads in a .csv file and saves it to a MySQL database
        'filename' and 'tablename' need to be strings: i.e. 'Pets.csv'
        '''
    
    import pymysql as mdb
    import pandas as pd
    from pandas.io import sql
    import re
    
    con =  mdb.connect('localhost', 'root','shreddie131','HomewardBound'); 
    
    
    data = pd.read_csv(filename)
    
    ## fixing table columns
    data.columns = [i.replace(' ','') for i in data.columns]
    
    ## renaming column that has name 'Condition', which is a SQL keyword
    data.columns = [re.sub('Condition','PetCondition',i) for i in data.columns]

    
    ### replacing nan with null to make it mysql friendly
    data = data.where((pd.notnull(data)), None)
    
    with con:
        cur = con.cursor()
        cur.execute("USE HomewardBound")
        data.to_sql(con = con, name = tablename, if_exists = 'replace', flavor = 'mysql')
    

        
if __name__ == '__main__':
    csv_to_mysql('Pets.csv','Pets')
    