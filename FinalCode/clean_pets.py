import pymysql as mdb
import pandas as pd
from pandas.io import sql
import re

con =  mdb.connect('localhost', 'root','shreddie131','HomewardBound'); 

with con:
        cur = con.cursor()
        cur.execute("USE HomewardBound")
        cur.execute('''DROP TABLE IF EXISTS Pets_cleaned''')
        cur.execute('''CREATE TABLE Pets_cleaned
                            (AnimalID bigint(20),
                            ArrivalDate varchar(63), 
                            ArrivedAs varchar(63), 
                            Breed varchar(63), 
                            Gender varchar(63), 
                            PetName varchar(63),
                            Status varchar(63),
                            PetCondition varchar(63),
                            ArrivalPrecinct varchar(63),
                            LengthofStay varchar(63),
                            Fixed bigint(20),
                            Age varchar(63),
                            Size varchar(63),
                            PrimaryColor varchar(63),
                            SecondaryColor varchar(63))''')   
        cur.execute(''' INSERT INTO Pets_cleaned
                            SELECT 
                            DISTINCT AnimalID,
                            ArrivalDate, 
                            ArrivedAs, 
                            Breed, 
                            Gender, 
                            PetName,
                            Status,
                            PetCondition,
                            ArrivalPrecinct,
                            LengthofStay,
                            Fixed,
                            Age,
                            Size,
                            PrimaryColor,
                            SecondaryColor
                            FROM Pets 
                            WHERE 
                            AnimalType = 'DOG' AND
                            ArrivedAs in ('STRAY' , 'UNABLE TO CARE FOR' , 'UNWANTED') AND
                            (PetCondition <> 'DEAD ON ARRIVAL' OR PetCondition IS NULL) AND
                            AnimalID NOT IN ('5780', '5777')''')
        