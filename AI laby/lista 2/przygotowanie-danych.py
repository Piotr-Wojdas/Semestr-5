import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('AI laby/lista 2/penguins_lter.csv')

# CZYSZCZENIE DATASETU

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
    "Delta 13 C (o/oo)",
    'Sample Number'
    ]

df.drop(columns=columns_to_drop, inplace=True)

# usunięcie rekordów, które posiadają braki
df.dropna(inplace=True)

# One Hot Encoding
def map_species(species):
    if 'Chinstrap' in species:
        return 1
    elif 'Adelie' in species:
        return 2
    elif 'Gentoo' in species:
        return 0

def map_sex(sex):
    if sex == 'MALE':
        return 1        
    elif sex == 'FEMALE':
        return 0    
    
df['Species'] = df['Species'].apply(map_species)
df['Sex'] = df['Sex'].apply(map_sex)

# Skalowanie wartości do przedziału [0, 1]
columns_to_scale = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
scaler = MinMaxScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
df[columns_to_scale] = df[columns_to_scale].round(2)

# Podział na zbiór treningowy i testowy
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Zapis do plików CSV
train_df.to_csv('AI laby/lista 2/penguins_train.csv', index=False)
test_df.to_csv('AI laby/lista 2/penguins_test.csv', index=False)





