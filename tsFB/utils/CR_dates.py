import numpy as np
import datetime as dt

def create_CR_date_dictionary(CR_table_path:str):
    ch_dt_arr = np.loadtxt(CR_table_path,dtype=str,delimiter='\t')

    CR_dates = {}
    for row in ch_dt_arr:
        CR_dates[row[0]] = row[1:]

    return CR_dates   # format = '%Y.%m.%d_%H:%M:%S'

def get_start_end_dates(CR_dates:dict,
                        carr_rot_num:str,
                        ):
    start = CR_dates[carr_rot_num][0]
    end = CR_dates[carr_rot_num][-1]

    start_dt = dt.datetime.strptime(start,'%Y.%m.%d_%H:%M:%S')
    end_dt = dt.datetime.strptime(end,'%Y.%m.%d_%H:%M:%S')

    return start_dt,end_dt