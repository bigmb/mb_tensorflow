##PCA file for search

from sklearn.decomposition import PCA,KernelPCA
import numpy as np
from mt.pandas import dfload
import pandas as pd
from mb_utils.src.logging import logger
import ast
from scipy.spatial import distance as dist_scipy
from sklearn.manifold import TSNE
import umap

def read_file_emb(train_file : str,val_file : str ,dim_red = 'ori',logger=logger):
    """
    Read the files, wrangle ,get embeddings and perform dimensionality reduction if needed
    Inputs:
        train_file : training file. Files from where we get the refrence embeddings. Default is None
        val_file : validation file. Files with events for testing. Default is None
        dim_red : dimensionality reduction type if needed. Default is original embeddings. Options are 'pca' and 'tsne'
        logger : logger
    Returns:
        df_train : training dataframe
        df_val : validation dataframe
    """
    pass






def distance(x1,y1,x2,y2):
    """
    Distance between two points
    Inputs:
        x1,y1,x2,y2 : coordinates of the points
    Returns:
        distance between the points
    """
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def distance_3d(x1,y1,z1,x2,y2,z2):
    """
    Distance between two points
    Inputs:
        x1,y1,z1,x2,y2,z2 : coordinates of the points
    Returns:
        distance between the points
    """
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def distance_scipy(event1,event2):
    """
    Distance between two points
    Inputs:
        event1 : list of coordinates of the points
        event2 : list of coordinates of the points
    Returns:
        distance between the points
    """
    return dist_scipy.euclidean(event1,event2)

def get_data(df,df_avg_ori = None,data_type='emb' ,dim_type= 'pca',dim_comp =2, logger=None):
    """
    Get the data from the file
    Inputs:
        df : dataframe (df with embeddings)
        df_avg_ori : dataframe with item_id (original csv). Required if data_type is 'item'
        data_type : type of data. Options are 'emb' and 'item'. Default is 'emb'
        dim_type : dimensionality reduction type. Options are 'pca','tsne' and 'umap'. Default is 'pca'
        dim_comp : number of pca components , also can be tsne or umap components. Default is 2
        logger : logger
    Returns:
        data : dataframe
    """
    if logger:
        logger.info('Loading data from the file')
    data = dfload(df)
    if logger:
        logger.info('Data loaded')
        logger.info('Data shape {}'.format(str(data.shape)))
        logger.info('Data columns {}'.format(str(data.columns)))
        logger.info('Performing dim_type = {}'.format(dim_type))

    if dim_type == 'pca':
        pca = PCA(n_components=dim_comp)
        pca_emb = pca.fit_transform(list(data['embedding']))
    elif dim_type == 'tsne':
        tsne = TSNE(n_components=dim_comp, perplexity=15, random_state=42, init='random', learning_rate=200)
        pca_emb = tsne.fit_transform(np.array(list(data['embedding'])))
        #pca_emb = TSNE(n_components=dim_comp).fit_transform(list(data['embedding']))
    elif dim_type == 'umap':
        pca_emb = umap.UMAP(n_components=dim_comp).fit_transform(list(data['embedding']))
    if logger:
        logger.info(pca_emb[:3])
    temp_pca = list(pca_emb)
    data['pca_res'] = temp_pca
    data.drop('embedding',axis=1,inplace=True)
    
    df_unique_taxcode = data['taxcode'].unique()
    df_for_item_id = dfload(df_avg_ori)
    data = data.merge(df_for_item_id[['event_id','item_id']],on='event_id',how='left')
    #data['item_id'] = df_for_item_id['item_id']
    df_unique_itemid = data['item_id'].unique()
    #data.to_csv('./data_item.csv')
    if logger:
        logger.info('Number of unique taxcodes {}'.format(len(df_unique_taxcode)))
        logger.info('Number of unique itemids {}'.format(len(df_unique_itemid)))
    pca_mean = {}
    if data_type=='emb':
        for taxcode in df_unique_taxcode:
            df_temp = data[data['taxcode']==taxcode]
            pca_mean[taxcode]  = np.mean(df_temp['pca_res'],axis=0)
    if data_type=='item':
        for itemid in df_unique_itemid:
            df_temp = data[data['item_id']==itemid]
            pca_mean[itemid]  = np.mean(df_temp['pca_res'],axis=0)
    # for taxcode in df_unique_taxcode:
    #     df_temp = data[data['taxcode']==taxcode]
    #     pca_mean[taxcode]  = np.mean(df_temp['pca_res'],axis=0)
    return pca_mean

def get_distance_val(val_file,val_file_csv,pca_mean,data_type='emb',dim_type = 'pca',dim_comp=2,limit=None,logger=logger):
    """
    Get the distance between the embeddings
    Inputs:
        val_file : validation file with embeddings
        val_file_csv : validation file with predictions
        pca_mean : pca mean (refrence embeddings)
        data_type : type of data. Options are 'emb' and 'item'. Default is 'emb'
        dim_type : dimensionality reduction type. Options are 'pca','tsne' and 'umap'. Default is 'pca'
        dim_comp : number of pca components , also can be tsne or umap components. Default is 2
        limit : limit the dataset
        logger : logger
    Returns:
        df_val : dataframe
    """
    df_val = dfload(val_file)
    df_val = df_val[['event_id','taxcode','embedding']]
    df_val_csv = dfload(val_file_csv)
    df_val_csv = df_val_csv[['event_id','gt_taxcode','predictions','item_id','prediction_item_ids']]
    #df_val['predictions_no_ast'] = df_val_csv['predictions']
    #df_val['gt_taxcode'] = df_val_csv['gt_taxcode']
    assert len(df_val)== len(df_val_csv)

    df_val_final = df_val.merge(df_val_csv,on='event_id',how='left')

    if limit:
        df_val = df_val[:limit]
        if logger:
            logger.info('Limiting the data to {}'.format(str(limit)))

    if logger:
        logger.info('df_val_final shape {}'.format(str(df_val_final.shape)))
        logger.info('df_val_final columns {}'.format(str(df_val_final.columns)))
        logger.info('df_val_final head {}'.format(str(df_val_final.head(2))))

    df_val_ast = [ast.literal_eval(df_val_final['predictions'].iloc[i]) for i in range(len(df_val_final))]
    df_val_final['predictions'] = df_val_ast
    df_val_ast = [ast.literal_eval(df_val_final['prediction_item_ids'].iloc[i]) for i in range(len(df_val_final))]
    df_val_final['prediction_item_ids'] = df_val_ast
    
    if logger:
        logger.info(df_val_final['prediction_item_ids'].iloc[:3])
        logger.info('Performing data_type of prediction item id = {}'.format(type(df_val_final['prediction_item_ids'].iloc[0])))
        logger.info('Performing dim_type = {}'.format(dim_type))
        logger.info('df_val_final temp itemid {}'.format(str(df_val_final['prediction_item_ids'].iloc[:3])))

    if dim_type=='pca':
        pca = PCA(n_components=dim_comp)
        pca_emb = pca.fit_transform(list(df_val_final['embedding']))
    if dim_type=='tsne':
        tsne = TSNE(n_components=dim_comp, perplexity=15, random_state=42, init='random', learning_rate=200)
        pca_emb = tsne.fit_transform(np.array(list(df_val_final['embedding'])))
        #pca_emb = TSNE(n_components=dim_comp).fit_transform(list(df_val_final['embedding']))
    if dim_type=='umap':
        pca_emb = umap.UMAP(n_components=dim_comp).fit_transform(list(df_val_final['embedding']))
    
    df_val_final['pca_pred'] = list(pca_emb)
    df_val_final.drop('embedding',axis=1,inplace=True)
    df_val_final = df_val_final.reset_index(drop=True)
    
    all_dist_taxcode = []
    best_dist_taxcode = []
    all_dist_itemid = []
    best_dist_itemid = []

    rows_with_missing_taxcode = []
    rows_with_missing_itemid = []
    
    if data_type=='emb':
        for i,k in df_val_final.iterrows():
            temp_dist = []
            for j in range(len(df_val_final['predictions'].iloc[i])):
                temp_taxcode = df_val_final['predictions'].iloc[i]

                if temp_taxcode[j] in pca_mean.keys():
                    temp_dist.append(distance_scipy(pca_mean[temp_taxcode[j]],df_val_final['pca_pred'].iloc[i]))
            all_dist_taxcode.append(temp_dist)
            if len(temp_dist)!=len(df_val_final['predictions'].iloc[i]):
                rows_with_missing_taxcode.append(1)
            else:
                rows_with_missing_taxcode.append(0)
            if len(temp_dist)== 0:
                best_dist_taxcode.append(int(1000))
            else:
                best_dist_taxcode.append(df_val_final['predictions'].iloc[i][np.argmin(temp_dist)])

    if data_type=='item':
        for i,k in df_val_final.iterrows():
            temp_dist = []
            for j in range(len(df_val_final['prediction_item_ids'].iloc[i])):
                temp_itemid = df_val_final['prediction_item_ids'].iloc[i]
                
                if temp_itemid[j] in pca_mean.keys():
                    temp_dist.append(distance_scipy(pca_mean[temp_itemid[j]],df_val_final['pca_pred'].iloc[i]))
            all_dist_itemid.append(temp_dist)
            if len(temp_dist)!=len(df_val_final['prediction_item_ids'].iloc[i]):
                rows_with_missing_itemid.append(1)
            else:
                rows_with_missing_itemid.append(0)
            if len(temp_dist)== 0:
                best_dist_itemid.append(int(1000))
            else:
                best_dist_itemid.append(df_val_final['prediction_item_ids'].iloc[i][np.argmin(temp_dist)])

    
    if data_type == 'emb':
        df_val_final['all_dist_taxcode'] = all_dist_taxcode
        df_val_final['best_dist_taxcode'] = best_dist_taxcode   
        df_val_final['rows_with_missing_taxcode'] = rows_with_missing_taxcode

        if logger:
            logger.info("Missing row : {}".format(str(sum(rows_with_missing_taxcode))))

        match_list = [df_val_final['best_dist_taxcode'].iloc[i]==df_val_final['gt_taxcode'].iloc[i] for i in range(len(df_val_final))]
        df_val_final['match'] = match_list
        df_val_without_missing = df_val_final[df_val_final['rows_with_missing_taxcode']==0]
    
    if data_type == 'item':
        df_val_final['all_dist_itemid'] = all_dist_itemid
        df_val_final['best_dist_itemid'] = best_dist_itemid
        df_val_final['rows_with_missing_itemid'] = rows_with_missing_itemid

        if logger:
            logger.info('Missing rows {}'.format(str(sum(rows_with_missing_itemid))))
        match_list = [df_val_final['best_dist_itemid'].iloc[i]==df_val_final['item_id'].iloc[i] for i in range(len(df_val_final))]
        df_val_final['match'] = match_list
        df_val_without_missing = df_val_final[df_val_final['rows_with_missing_itemid']==0]
    
    if logger:
        logger.info('Type of data_type : {}'.format(data_type))
        logger.info('Number of matches total : {}'.format(sum(match_list)))
        logger.info('Total number of event : {}'.format(len(match_list)))
        logger.info('Number of matches with missing taxcode/item_id: {}'.format(sum(df_val_without_missing['rows_with_missing_taxcode'])))
        logger.info('Total number of event with missing taxcode/item_id: {}'.format(len(df_val_without_missing)))
        logger.info('Accuracy total: {}'.format(sum(match_list)/len(match_list)))
        logger.info('Accuracy without missing taxcode: {}'.format(sum(df_val_without_missing['match'])/len(df_val_without_missing)))
    return df_val_final


def selecting_events(df,no_of_emb=5,logger=None):
    """
    Selecting 5 events for search from eveny taxcode
    Inputs:
        df : dataframe
        no_of_emb : number of events to select
        logger : logger
    Returns:
        df_selected : dataframe
    """
    if logger:
        logger.info('Selecting {} events for search'.format(no_of_emb))
    df_unquie_taxcode = df['taxcode'].unique()
    if logger:
        logger.info('Number of unique taxcodes {}'.format(len(df_unquie_taxcode)))
    df_selected = pd.DataFrame()
    for i in df_unquie_taxcode:
        df_temp = df[df['taxcode']==i]
        if len(df_temp)>no_of_emb:
            df_temp = df_temp.sample(n=no_of_emb)
        df_selected = df_selected.append(df_temp)
    if logger:
        logger.info('Selected {} events for search'.format(len(df_selected)))
    return df_selected

def get_distance(df_selected,emb_to_search,logger=None):
    """
    Get the distance between the embeddings
    Inputs:
        df_selected : dataframe
        emb_to_search : embedding to search
        logger : logger
    Returns:
        df_selected : dataframe
    """
    if logger:
        logger.info('Calculating distance')
    df_selected['distance'] = df_selected['pca_emb'].apply(lambda x: distance(x[0],x[1],emb_to_search[0],emb_to_search[1]))
    if logger:
        logger.info('Distance calculated')
    return df_selected

def get_top1(df_selected,logger=None):
    """
    Get the top1 event for the embedding
    Inputs:
        df_selected : dataframe
        logger : logger
    Returns:
        df_selected.iloc[0] : dataframe
    """
    if logger:
        logger.info('Getting the top1 event')
    df_selected = df_selected.sort_values(by='distance')
    if logger:
        logger.info('Top1 event_id {}'.format(df_selected.iloc[0]['event_id']))
        logger.info('Top1 distance {}'.format(df_selected.iloc[0]['distance']))
        logger.info('Top1 taxcode {}'.format(df_selected.iloc[0]['taxcode']))
    return df_selected.iloc[0]
