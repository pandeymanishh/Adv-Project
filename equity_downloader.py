# Function to write the csv file to a directory


def download_file(url, directory,timeout):
    import requests
    import os
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Get the file name from the URL
    file_name = url.split('/')[-1]
    
    # Prepare the file path
    file_path = os.path.join(directory, file_name)
    
    try:
        # Send a GET request to the URL with a timeout of 2 seconds
        response = requests.get(url, timeout=timeout)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Save the file
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"File downloaded successfully to '{file_path}'")
        else:
            print("Failed to download the file")
    except requests.exceptions.Timeout:
        print("Request timed out. Skipping download.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {str(e)}")

# Function to change todays date into the string format


def format_date(date_to_chng):
    from datetime import date
    formatted_date = date_to_chng.strftime("%d%m%Y")
    return formatted_date


# Function to create the date list


def generate_date_list(start,end):
    from datetime import date, timedelta
    start_date = date(start[0], start[1], start[2])
    end_date = date(end[0], end[1], end[2])

    # Add NSE trading holidays for the specified years
    trading_holidays = [date(2023,1,26),date(2023,3,7),date(2023,3,30),date(2023,4,4),date(2023,4,7),date(2023,4,14),date(2023,5,1)
                        ,date(2023,6,28),date(2023,8,15),date(2023,9,19),date(2023,10,2),date(2023,10,24),date(2023,11,14)
                        , date(2023,11,27),date(2022,1,26),date(2022,3,1),date(2022,3,18),date(2022,4,14),date(2022,4,15)
                        , date(2022,5,3),date(2022,8,9),date(2022,8,15),date(2022,8,31),date(2022,10,5),date(2022,10,24)
                        , date(2022,10,26),date(2022,11,8),date(2021,1,26),date(2021,3,11),date(2021,3,29),date(2021,4,2)
                        , date(2021,4,14) ,date(2021,4,21),date(2021,5,13),date(2021,7,21) ,date(2021,8,19),date(2021,9,10)
                        , date(2021,10,15),date(2021,11,4),date(2021,11,5),date(2021,11,19)]

    date_list = []
    current_date = start_date

    while current_date <= end_date:
        if current_date.weekday() < 5 and current_date not in trading_holidays:
            date_list.append(current_date)
        current_date += timedelta(days=1)

    return [format_date(x) for x in date_list]


# Take a date and return list
def extract_date_components(input_date):   
    from datetime import date
    year = input_date.year
    month = input_date.month
    day = input_date.day
    return [year, month, day]

# Combine all the files together 
def combine_files(folder_path, output_file):
    import pandas as pd
    import os
    all_data = pd.DataFrame()  # Initialize an empty DataFrame

    for file in os.listdir(folder_path):
        print(file)
        if file.endswith('.csv'):  # Assuming all files in the folder are CSV files
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path)
            all_data = pd.concat([all_data, data], ignore_index=True)


    all_data.to_csv(output_file, index=False)
    print(f"Combined data saved to '{output_file}'")
    
# Define the conversion function
def convert_date(date_string, format = "%d-%b-%Y"):
    from datetime import datetime
    date_format = format
    date = datetime.strptime(date_string.strip(), date_format).date()
    return date

# Clean a folder from all files
def clean_folder(folder_path):
    import os

    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate over the files and remove them one by one
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    print("All files have been removed from the folder.")
    
# Copy files from one folde to another folder  
def copy_files(source_folder, destination_folder):
    import os
    import shutil
    # Get a list of all files in the source folder
    file_list = os.listdir(source_folder)

    # Iterate over the files and copy them to the destination folder
    for file_name in file_list:
        source_file = os.path.join(source_folder, file_name)
        if os.path.isfile(source_file):
            destination_file = os.path.join(destination_folder, file_name)
            shutil.copy2(source_file, destination_file)

def best_cutoff(actual, prediction):
    import numpy as np
    from sklearn.metrics import confusion_matrix
    best_cutoff = None
    best_accuracy = 0.0

    for cutoff in np.arange(0.01, 1.0, 0.01):
        y_pred = np.where(prediction >= cutoff, 1, 0)
        accuracy = np.mean(actual == y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_cutoff = cutoff
            
    y_pred = np.where(prediction >= best_cutoff, 1, 0)
    
    best_minor_accuracy = confusion_matrix(actual, y_pred, normalize='pred')[1,1]
    usability = confusion_matrix(actual, y_pred)[1,1]
    return {"Accuracy_All":best_accuracy, 'Best_Cutoff':best_cutoff, "Accuracy":best_minor_accuracy
            , 'confusion_matrix':confusion_matrix(actual, y_pred)}


def eq_data_pred(eq_df_industry, symbol, keep_ma=True):
    import numpy as np
    eq_symbol = eq_df_industry[eq_df_industry['SYMBOL']==symbol].copy()
    eq_symbol.sort_values('DATE', inplace=True)

    # Define the lag values
    lags = [7, 14, 30, 45, 60, 90, 180]
    value = [5, 5, 10, 12, 15, 25, 30]

    # Calculate EMAs for each lag value and create crossover flag
    for lag, value_lag in zip(lags, value):

        ema_column_name = f'MA_{lag}'
        crossover_column_name = f'CROSSOVER_{lag}'
        target_col = f'TARGET_{lag}'
        deliv_col = f'MA_TRADE_{lag}'
        target_val = f'TARGET_VAL_{lag}'
        gap_col = f'GAP_MA_{lag}'
        grad_col = f'GRAD_{lag}'
        vol_grad_col = f'VOL_GRAD_{lag}'

        # Calculate EMA
        eq_symbol[ema_column_name] = eq_symbol['CLOSE_PRICE'].rolling(lag).mean()
        eq_symbol[deliv_col] = eq_symbol['TTL_TRD_QNTY'].rolling(lag).mean()

        # Create crossover flag
        eq_symbol[crossover_column_name] = eq_symbol.apply(lambda row: 1 if row[ema_column_name] <= row['CLOSE_PRICE'] else 0, axis=1)

        if lag<=30:
            eq_symbol['TEMP'] = 100*(eq_symbol['CLOSE_PRICE'][::-1].rolling(lag).max() - eq_symbol['CLOSE_PRICE'])/eq_symbol['CLOSE_PRICE']

            eq_symbol[target_col] = eq_symbol.apply(lambda row: 1 if row['TEMP'] >= value_lag else 0, axis=1)
            eq_symbol.loc[eq_symbol['TEMP'].isna(),target_col] = np.NaN
            
            eq_symbol[target_val] = eq_symbol['CLOSE_PRICE'][::-1].rolling(lag).max()

        eq_symbol[gap_col] = eq_symbol['CLOSE_PRICE'] - eq_symbol[ema_column_name]

        eq_symbol[grad_col] = np.gradient(eq_symbol[ema_column_name])
        eq_symbol[vol_grad_col] = np.gradient(eq_symbol[deliv_col]) 
        
        if not keep_ma:
            eq_symbol.drop([ema_column_name],axis=1, inplace=True)
    
    eq_symbol.drop(['TEMP'],axis=1, inplace=True)
    eq_symbol['GRAD_CLOSE'] = np.gradient(eq_symbol['CLOSE_PRICE'])
    eq_symbol['VOL_GRAD_CLOSE'] = np.gradient(eq_symbol['TTL_TRD_QNTY'])    
    eq_symbol['DAY'] = eq_symbol['DATE'].dt.day
    eq_symbol['MONTH'] = eq_symbol['DATE'].dt.month
    eq_symbol['WEEK_NUMBER'] = eq_symbol['DATE'].dt.isocalendar().week

    return eq_symbol

def collate_grps(lst1, lst2):
    from itertools import chain
    chk = [x for x in lst1 if x in lst2]
    if len(chk)==0:
        return None
    else:
        return list(set(chain.from_iterable([lst1, lst2])))

def best_cutoff_acc(train_actual, train_prediction, test_actual = None, test_prediction = None):
    import numpy as np
    from sklearn.metrics import confusion_matrix
    from scipy.stats import percentileofscore
    import pandas as pd
    
    df = pd.DataFrame({'actual':train_actual, "prob":train_prediction})
    perc = list(np.unique(df['prob'].quantile(np.arange(0.00, 1.0, 0.025))))
    perc.append(1)
    # Break the column into categories based on percentiles
    df['Categories'] = pd.cut(df['prob'], bins=perc, labels = perc[:-1]) 
    pivot = df.pivot_table(index="Categories", columns="actual", aggfunc = 'count', values="prob").reset_index()
    pivot.columns = ['Categories', 'Actual_0', 'Actual_1']
    pivot.sort_values("Categories", ascending=False, inplace=True)
    pivot['Total'] = pivot['Actual_0']+pivot['Actual_1']
    pivot['CumTotal'] = pivot['Total'].cumsum()
    pivot['CumActual1'] = pivot['Actual_1'].cumsum()
    pivot['Cum_Accuracy'] = np.round(100*pivot['CumActual1']/pivot['CumTotal'],1)
    pivot['Bin_Accuracy'] = np.round(100*pivot['Actual_1']/pivot['Total'],1)
    
    pivot['PopPerc'] = np.round(100*pivot['CumTotal']/pivot['Total'].sum(),1)
    # Ensuring a more than 70% accuracy in the bin to be selected
    pivot_f = pivot[pivot['Bin_Accuracy']>=70].copy()
    # get the cutoff 
    best_cutoff = pivot_f['Categories'].min()
    
    best_rank = percentileofscore(df['prob'], best_cutoff)

    best_train_accuracy = pivot_f['Cum_Accuracy'].min()
            
    train_pred = np.where(train_prediction >= best_cutoff, 1, 0)
    train_confusion = confusion_matrix(train_actual, train_pred)
    train_acc = np.mean(train_actual==train_pred)
    
    if (test_actual is not None) & (test_prediction is not  None):
        test_pred = np.where(test_prediction >= best_cutoff, 1, 0)
        test_confusion = confusion_matrix(test_actual, test_pred)
        test_accuracy = np.round(100*(test_confusion[1,1]/(test_confusion[1,1]+test_confusion[0,1])),2)
        
        return {'Best_Cutoff':best_cutoff, "Train_Accuracy":best_train_accuracy, "Test_Accuracy": test_accuracy
                , 'train_confusion_matrix':train_confusion, "test_confusion_matrix":test_confusion , 'Rank_Cutoff':best_rank}
    else:
        return {'Best_Cutoff':best_cutoff, "Train_Accuracy":best_train_accuracy, 'train_confusion_matrix':train_confusion, 'Rank_Cutoff':best_rank}

def drop_corr_vars(df, cols, cutoff=0.8):

    pred_cols1 = [x for x  in cols if not x.startswith("CROSSOVER")]
    corr_matrix = df[pred_cols1].corr().reset_index()

    corr_df = pd.DataFrame(columns=['Index', 'Column2', 'Correlation'])
    for col in corr_matrix.columns[1:]:
        tmp1 = corr_matrix[['index', col]].copy()
        tmp1['Column2'] = col
        tmp1.columns = ['Index', 'Correlation', 'Column2']
        tmp1 = tmp1[['Index', 'Column2', 'Correlation']]
        corr_df = pd.concat([corr_df, tmp1], ignore_index=True)

    corr_df = corr_df[(corr_df['Index']!=corr_df['Column2']) & (corr_df['Correlation']>cutoff)].copy()
    corr_df.sort_values(by=['Correlation'], ascending=False, inplace=True)

    lst = [[x,y] for x,y in zip(corr_df['Index'].tolist(),corr_df['Column2'].tolist())]
    for x in range(len(lst)):
        lst[x].sort()

    corr_df['Key']=lst
    corr_df.drop_duplicates(subset=['Key'], inplace=True)
    corr_df.sort_values(by=['Index', 'Correlation'], ascending =[True,False], inplace=True)

    grps=[corr_df.iloc[0,3]]

    for i in range(corr_df.shape[0]-1):
        for x, y in enumerate(grps):
            val_lst = []
            chk = collate_grps(lst1 = corr_df.iloc[i+1,3], lst2 = y)
            if chk is not None:
                val_lst.append(x)
                val_lst.append(chk)
        if len(val_lst)>0:
            grps[val_lst[0]] = val_lst[1] 
        else:
            grps.append(corr_df.iloc[i+1,3])

    drop_vars = list(chain.from_iterable([x[1:] for x in grps]))

    return  [x for x in cols if x not in drop_vars]
