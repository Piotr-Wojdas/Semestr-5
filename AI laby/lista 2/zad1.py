import pandas as pd

df = pd.read_csv('AI laby/lista 2/penguins_lter.csv')

# CZYSZCZENIE DATASETU

# usunięcie rekordów, które posiadają braki
df.dropna(inplace=True)

# lista kolumn do usunięcia
columns_to_drop = [
    'studyName', 
    'Region', 
    'Island', 
    'Stage', 
    'Individual ID', 
    'Clutch Completion', 
    'Date Egg',  
    'Comments',
    "Delta 15 N (o/oo)",
    "Delta 13 C (o/oo)"]

df.drop(columns=columns_to_drop, inplace=True)

    



