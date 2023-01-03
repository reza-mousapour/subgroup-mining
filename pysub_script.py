import pysubgroup as ps
import pandas as pd
import numpy as np
import pickle
import os


def save_pysub(result, file_name='rules'):
    if not os.path.exists('pickle_data'):
        os.makedirs('pickle_data')
    result.to_dataframe().to_pickle('pickle_data/' + file_name + '.pkl')


def save_id_set(result, dataframe, file_name="id_set"):
    id_set = {}
    for i in range(result.to_dataframe().shape[0]):
        query = ''
        query_list = []
        output = result.to_dataframe().iloc[i]['subgroup']
        output_list = output.split('AND')
        for item in output_list:
            if ':' in item:
                item_list = item.split(':')
                key = item_list[0]
                begin_interval = item_list[1][2:]
                end_interval = item_list[2][:-1]
                temp = key + ' >= ' + begin_interval + ' & ' + key + ' <= ' + end_interval
                query_list.append(temp)
            else:
                query_list.append(item)
        for index in range(len(query_list)):
            if index != len(query_list)-1:
                query += query_list[index] + ' & '
            else:
                query += query_list[index]

        print(query)
        res = sorted(dataframe.query(query)['PID'].to_list())

        id_set[i] = set(res)
    if not os.path.exists('pickle_data'):
        os.makedirs('pickle_data')
    with open('pickle_data/' + file_name + '.pkl', 'wb') as handle:
        pickle.dump(id_set, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_ids(dataframe, file_name='ids'):
    if not os.path.exists('pickle_data'):
        os.makedirs('pickle_data')
    temp = dataframe['PID'].to_list()
    with open('pickle_data/' + file_name + '.pkl', 'wb') as handle:
        pickle.dump(temp, handle, protocol=pickle.HIGHEST_PROTOCOL)



 
if __name__ == "__main__":
    subj = pd.read_excel('Data/LAB_1400.xlsx')

    temp = [1 if (item > 46) else 0 for item in subj['LDLcholestrol']]
    subj["result"] = temp

    target = ps.BinaryTarget('result', True)

    searchspace = ps.create_selectors(subj,
                                      ignore=['result', 'PID', 'LDLcholestrol', 'Field43'])
    task = ps.SubgroupDiscoveryTask(
        subj,
        target,
        searchspace,
        result_set_size=20,
        depth=100,
        qf=ps.WRAccQF())
    result = ps.BeamSearch().execute(task)
    print(result.to_dataframe().to_string())
    temp = result.to_dataframe()
    # print(subj[subj['cholesterol'])
    save_id_set(result, subj)
    save_ids(subj)
    save_pysub(result)