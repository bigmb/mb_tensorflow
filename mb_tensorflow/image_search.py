##fastdup file from embeddings

import fastdup
import numpy as np
from mt.pandas import dfload
from mb_utils.src.logging import logger
import pandas as pd
import os
import shutil

# ## Adding event_id to a csv
# def event_ids_add_components(path):
#     """
#     Adding events to compoments
#     Args:
#         path(str) : path of the file to search
#     """
#     a_lookup = pd2.read_csv(path+'/main_lookup_file.txt')
#     a_atrain_fea = pd2.read_csv(path+'/atrain_features.dat.csv')
#     a_cc = pd2.read_csv(path+'/connected_components.csv')
#     a_ci = pd2.read_csv(path +'/component_info.csv')
    
#     a_cc['filename'] = a_atrain_fea['filename']
#     for i in range(len(a_cc)):
#         x1  = a_lookup[a_lookup['local_path']==a_atrain_fea['filename'].iloc[i]]
#         a_cc['__id'].iloc[i]= x1['event_id'].iloc[0]
#     #a_cc.to_csv(path+'/connected_components.csv')
#     #a_cc['component_id']=a_cc['__id']
#     a = a_cc[['__id','component_id','pagerank','min_distance']]
#     a.to_csv(path+'/connected_components.csv')



def sliced_search(location,training_file,dataset=[10000,20000,50000,100000,200000,300000,500000,1000000] ,no_of_img=3,image_threshold=0,logger=logger):
    """
    location : str
        path to the directory where the results will be saved
    training_file : str
        path to the training file
    dataset : list, optional
        list of datasets to be used, by default [10000,20000,50000,100000,200000,300000,500000,1000000]
    no_of_img : int, optional
        number of images to be returned, by default 3
    no_of_img_val : int, optional
        number of images to be returned for validation, by default 3
    image_threshold : int, optional
        image threshold for the numbor of images, by default 0
    logger : logging.Logger or equivalent, optional
        logger for logging messages, by default logger
    """

    if logger:
        logger.info("Loading embeddings using mb pandas")
    df = dfload(training_file)
    if logger:
        logger.info("len of file {}".format(len(df)))
        logger.info("File loaded")
        logger.info("Columns {}".format(df.columns))

    if image_threshold > 0:
        df2 = df.groupby('taxcode').filter(lambda x: len(x) > image_threshold)
        if logger:
            logger.info("len of file after filtering {}".format(len(df2)))
            logger.info("File filtered")
        df2 = df2.reset_index(drop=True)
    else:
        df2 = df.copy()

    local_path = df2['local_paths'].to_list()
    ax = df2['embedding'].to_list()
    features_arr32 = {i: np.float32(ax[i]) for i in range(len(ax))}
    array = np.array(list(features_arr32.values()))

    #del df,df2

    for i in dataset:
        if logger:
            logger.info("Dataset {}".format(i))
        path = location + '/dataset_' + str(i)
        if os.path.exists(path):
            if logger:
                logger.info("Path already exists. Removing the old dataset")
            shutil.rmtree(path)
            os.mkdir(path)
        else:
            os.mkdir(path)
            if logger:
                logger.info("Path created")        
        
        tmp_local_path = local_path[:i]
        tmp_array = array[:i]
        fastdup.save_binary_feature(path, tmp_local_path, tmp_array)
        #fastdup.run(local_path,work_dir=path,run_mode=2,license="magical",d=240,verbose=1,threshold=0,nearest_neighbors_k=no_of_img,turi_param='ccthreshold=')
        fastdup.run(local_path,work_dir=path,run_mode=2,license="magical",d=240,verbose=1,threshold=0,nearest_neighbors_k=no_of_img)

        if logger:
            logger.info("Training file done: {}".format(path))
    


def sliced_validation(location,val_file,dataset=[10000,20000,50000,100000,200000,300000,500000,1000000],no_of_img=3,logger=logger):
    """
    location : str
        path to the directory where the results will be saved
    val_file : str
        path to the validation file
    dataset : list, optional
        list of datasets to be used, by default [10000,20000,50000,100000,200000,300000,500000,1000000]
    no_of_img : int, optional
        number of images to be returned, by default 3
    logger : logging.Logger or equivalent, optional
        logger for logging messages, by default logger
    """

    if logger:
        logger.info("Loading embeddings using mb pandas")
    df = dfload(val_file)
    if logger:
        logger.info("len of file {}".format(len(df)))
        logger.info("File loaded")
        logger.info("Columns {}".format(df.columns))


    local_path = df['local_paths'].tolist()
    ax = list(df['embedding'])
    features_arr32 = {i: np.float32(ax[i]) for i in range(len(ax))}
    array = np.array(list(features_arr32.values()))
    
    path_val = location + '/dataset_validation'
    if os.path.exists(path_val):
        if logger:
            logger.info("Path already exists. Removing the old dataset")
        shutil.rmtree(path_val)
        os.mkdir(path_val)
        if logger:
            logger.info("Path created")
    else:
        os.mkdir(path_val)
        if logger:
            logger.info("Path created")

    #fastdup.save_binary_feature(path_val, local_path, array)

    for i in dataset:
        path = location + '/dataset_' + str(i)
        if logger:
            logger.info("Dataset {} loading".format(i))
            logger.info("work dir: {}".format(path))    
            logger.info("test dir: {}".format(path_val))
        
        fastdup.save_binary_feature(path_val, local_path, array)

        fastdup.run(local_path, work_dir=path,test_dir=path_val,license='magical',nearest_neighbors_k=no_of_img,threshold=0.0, d=240 ,run_mode=4)
        if logger:
            logger.info("Validation file done: {}".format(path))
        
def file_accuracy(path : str,orginal_file : pd, val_file : pd,logger=logger):
    """
    path : str
        path to the directory where the results were saved
    orginal_file : pd
        original file used for training containing taxcodes and local_paths
    val_file : pd
        validation file containing taxcodes and local_paths
    logger : logging.Logger or equivalent, optional
        logger for logging messages, by default logger
    """
    similarity_file = dfload(path + '/similarity.csv')
    outlier_file = dfload(path + '/outliers.csv')
    if logger:
        logger.info("Similarity file shape {}".format(similarity_file.shape))
        logger.info("Outlier file shape {}".format(outlier_file.shape))
    
    #res_file = pd.DataFrame(columns=['from','to','distance','ori_taxcode','pred_taxcode','match'])
    # temp_url_dict = {}
    # temp_data_dict = {}
    # temp_distance_dict = {}
    # temp_ori_taxcode_dict = {}
    # temp_pred_taxcode_dict = {}
    # temp_match_dict = {}

    ml_kind_train = orginal_file[orginal_file['ml_kind'] == 'training']
    ml_kind_val = orginal_file[orginal_file['ml_kind'] == 'validation']
    training_url_dict = dict(zip(list(ml_kind_train['local_paths']),list(ml_kind_train['taxcode'])))
    val_url_dict = dict(zip(list(ml_kind_val['local_paths']),list(ml_kind_val['taxcode'])))

    val_urls = list(val_file['filename'])
    simi_file_filter = similarity_file['from'].isin(val_urls)
    simi_file_temp_data = similarity_file[simi_file_filter].sort_values('distance',ascending=False)
    outlier_file_filter = ~simi_file_filter
    outlier_file_temp_data = outlier_file[outlier_file_filter].sort_values('distance',ascending=False)
    temp_data_file = pd.concat([simi_file_temp_data,outlier_file_temp_data])
    temp_data = temp_data_file.sort_values(['from','distance'],ascending=False).drop_duplicates('from').sort_index()
    temp_data['val_taxcode'] = temp_data['from'].map(val_url_dict)
    temp_data['pred_taxcode'] = temp_data['to'].map(training_url_dict)
    temp_data['match'] = temp_data['val_taxcode'] == temp_data['pred_taxcode']

    if logger:
        logger.info("Accuracy for path {} : {}".format(path,temp_data['match'].mean()))
    return temp_data 
    

    # for k,i in val_file.iterrows():
    #     temp_url = i[0]
    #     temp_data = similarity_file.loc[similarity_file['from'] == temp_url]
    #     if temp_data.empty:
    #         temp_data = outlier_file.loc[outlier_file['from'] == temp_url]
    #     temp_data = temp_data.sort_values(by=['distance'],ascending=False)
    #     search_taxcode = orginal_file.loc[orginal_file['local_paths']==temp_data['to'].iloc[0]]['taxcode']
    #     search_taxcode = search_taxcode.iloc[0]
    #     val_ori_taxcode = orginal_file.loc[orginal_file['local_paths']==temp_url]['taxcode']
    #     val_ori_taxcode = val_ori_taxcode.iloc[0]
    #     if search_taxcode == val_ori_taxcode:
    #         match_val = 1
    #     else:
    #         match_val = 0
    #     temp_url_dict[k]=temp_url
    #     temp_data_dict[k]=temp_data['to'].iloc[0]
    #     temp_distance_dict[k]=temp_data['distance'].iloc[0]
    #     temp_ori_taxcode_dict[k]=val_ori_taxcode
    #     temp_pred_taxcode_dict[k]=search_taxcode
    #     temp_match_dict[k]=match_val
    #     if k%1000 == 0:
    #         if logger:
    #             logger.info("Completed {} of {}".format(k,len(val_file)))
    #res_file = pd.DataFrame({'from':temp_url_dict.values(),'to':temp_data_dict.values(),'distance':temp_distance_dict.values(),'ori_taxcode':temp_ori_taxcode_dict.values(),'pred_taxcode':temp_pred_taxcode_dict.values(),'match':temp_match_dict.values()})
    #res_file = res_file.append({'from':temp_url,'to':temp_data['to'].iloc[0],'distance':temp_data['distance'].iloc[0],'ori_taxcode':val_ori_taxcode,'pred_taxcode':search_taxcode,'match':match_val},ignore_index=True)
    # if logger:
    #     logger.info('Result file iloc[0] : {}'.format(res_file.iloc[0]))
    #     logger.info("Result file shape {}".format(res_file.shape))
    # return res_file


def sliced_val_accuracy(location : str, original_file : str, val_file = './dataset_validation/atrain_features.dat.csv', dataset=[10000,20000,50000,100000,200000,300000,500000,1000000], logger=logger) -> None:
    """
    location : str
        path to the directory where the results were saved
    original_file : str
        path to the original file used for training containing taxcodes and local_paths
    val_file : str
        path to the validation file containing the after_image_url of validation images. Default : ./dataset_validation/atrain_features.dat.csv)
    dataset : list, optional
        list of datasets to be used, by default [10000,20000,50000,100000,200000,300000,500000,1000000]
    logger : logging.Logger or equivalent, optional
        logger for logging messages, by default logger
    """
    if logger:
        logger.info("Folder {} used for calculating accuracy".format(location))
        logger.info("Validation file used {}".format(val_file))
        logger.info("Main file used {}".format(original_file))
        logger.info("Datasets used {}".format(dataset))
    df_val = dfload(val_file)
    original_file = dfload(original_file)
    
    for i in dataset:
        path = location + '/dataset_' + str(i)
        if logger:  
            logger.info("Dataset {} loading".format(i))
            logger.info("work dir: {}".format(path))    
        acc = file_accuracy(path,original_file,df_val,logger=logger)
        #if logger:
        #    logger.info("Accuracy for dataset {} is {}".format(i,acc))
        acc.to_csv(path + '/accuracy.csv',index=False)




def search_from_emb(file,file_location_to_save,val_file =None,val_res_dir = None,no_of_img=5,no_of_img_val=2,image_threshold=150,confidence_threshold = 0.0,bounding_box=False,components=False,outliers=False,search=False,logger=logger):
    """
    Search for similar images from the embeddings

    Parameters
    ----------
    file : str
        path to the embeddings file with file_image_name (id), embeddings and location of image
    file_location_to_save : str
        path to the file where the results will be saved
    val_file : str, optional
        path to the validation file, by default None
    val_res_dir : str, optional
        path to the Image Search results directory, by default None
    no_of_img : int, optional
        number of images to be returned, by default 5
    no_of_img_val : int, optional
        number of images to be returned for validation, by default 2
    image_threshold : int, optional
        image threshold for the numbor of images, by default 150
    confidence_threshold : float, optional
        confidence threshold for the image search, by default 0.0
    components : bool, optional
        if True, returns the components of the taxcode created by fastdup, by default False
    search : bool, optional
        if True, returns the image search results - val_file must be there, by default False
    logger : logging.Logger or equivalent, optional
        logger for logging messages, by default logger
    """
    if logger:
        logger.info("Loading embeddings using mb pandas")
    df = dfload(file)
    if logger:
        logger.info("len of file {}".format(len(df)))
        logger.info("File loaded")
        logger.info("Columns {}".format(df.columns))
    #if isinstance(df, pd.DataFrame):
    #    if ['embedding','event_id','taxcode','local_path','ml_kind'] not in df.columns:
    #        raise ValueError("Columns are not correct in the dataset")

    df2 = df.groupby('taxcode').filter(lambda x: len(x) > image_threshold)
    if logger:
        logger.info("len of file after filtering {}".format(len(df2)))
    
    df_training = df2[df['ml_kind'] == 'training']
    if logger:
        logger.info("len of file after ml_kind=training {}".format(len(df_training)))
        logger.info("Creating fastdup object")
    
    path = file_location_to_save
    if os.path.exists(path):
        if logger:
            logger.info("Path already exists")
    else:
        os.mkdir(path)
        if logger:
            logger.info("Path created")
    work_dir = path
    dim = len(df_training['embedding'][0])
    len_n = len(df_training)
    train_taxcode='train_all_' + str(image_threshold)
    local_path = df_training['local_paths'].tolist()

    ax = list(df_training['embedding'])

    features_arr32 = {i: np.float32(ax[i]) for i in range(len(ax))}
    array = np.array(list(features_arr32.values()))
    if logger:
        logger.info('Length of array for features : {}'.format(str(array.shape)))

    filesnames = list(df_training['event_id'])
    files_str = [str(x) for x in filesnames]


    if logger:
        logger.info('Creating Binary features for Search')
    
    fastdup.save_binary_feature(work_dir, local_path, array)
    if logger:
        logger.info('Binary Feature for search created.Starting Image search run')
        logger.info('fasdup version : {}'.format(str(fastdup.__version__)))
        logger.info('Work dir :{}'.format(str(work_dir)))
    
    fastdup.run(local_path,work_dir=work_dir,run_mode=2,license="magical",d=dim,verbose=1,threshold=confidence_threshold,nearest_neighbors_k=no_of_img,turi_param='ccthreshold=')
    logger.info('Image Search Completed. For each image {} similar images saved.'.format(str(no_of_img)))
    
    results_dir =path
    fastdup.save_binary_feature(results_dir, files_str, array)
    fastdup.run(os.path.join(work_dir, "atrain_features.dat.csv"),work_dir=results_dir,d=dim,run_mode=4,threshold=0,nearest_neighbors_k=no_of_img,)   

    if logger:
        logger.info('Creating lookup file')
    temp_lookup = df_training[['event_id','taxcode','local_paths']]
    lookup_file = pd.read_csv(os.path.join(work_dir, "atrain_features.dat.csv"))
    for i in range(len(lookup_file)):
        temp_a = temp_lookup[temp_lookup['event_id']==lookup_file['filename'].iloc[i]]
        lookup_file['event_id'].iloc[i] = temp_a['event_id'].iloc[0]
        lookup_file['local_paths'].iloc[i] = temp_a['local_paths'].iloc[0]
        lookup_file['taxcode'].iloc[i] = temp_a['taxcode'].iloc[0]
    #lookup_file['event_id'] =   df_training['event_id']
    #lookup_file['taxcode'] = df_training['taxcode']
    #lookup_file['local_paths'] = df_training['local_paths']
    lookup_file.to_csv(os.path.join(work_dir, "lookup_file.csv"), index=False)
    if logger:
        logger.info('Lookup file created')

    if logger:
        logger.info('Getting labels')
    dict_name={local_path[i]:df_training['taxcode'].iloc[i] for i in range(len(df_training))}
    def my_label(dict_val):
        return dict_name[dict_val]

    if logger:
        logger.info('Getting event_ids')
    dict_event_id = {df_training['local_paths'].iloc[i]:df_training['event_id'].iloc[i] for i in range(len(df_training))}
    def dict_event_id_func(dict_val):
        return dict_event_id[dict_val]   

    if logger:
        logger.info('Getting bounding box if TRUE')
    if bounding_box:
        dict_bounding_box = {df_training['local_paths'].iloc[i]:df_training['bounding_box'].iloc[i] for i in range(len(df_training))}

        def dict_bounding_box_func(dict_val):
            return dict_bounding_box[dict_val]    

    if logger:
        logger.info('Creating Components gallery if TRUE')
    if components:  
        if bounding_box:
            fastdup.create_components_gallery(path,save_path=path, num_images=15,get_label_func=my_label,get_bounding_box_func=dict_bounding_box_func,get_extra_col_func=dict_event_id_func)
        else:
            fastdup.create_components_gallery(path,save_path=path, num_images=15,get_label_func=my_label,get_extra_col_func=dict_event_id_func)

    if logger:
        logger.info('Creating Outliers gallery if TRUE')
    
    if outliers:
        path_out = file_location_to_save+'/outliers/'
        if bounding_box:
            fastdup.create_outliers_gallery(path + '/outliers.csv',save_path=path_out, num_images=30,how='one',get_bounding_box_func=dict_bounding_box_func,get_extra_col_func=dict_event_id_func)
        else:
            fastdup.create_outliers_gallery(path + '/outliers.csv',save_path=path_out, num_images=30,how='one',get_extra_col_func=dict_event_id_func)

    if search == True:
        search_val(val_file,val_res_dir,work_dir,no_of_img_val,logger=logger)


def search_val(val_file,val_res_dir,work_dir,no_of_img_val,logger=None):
    if logger:
        logger.info('Performing Image search') 
        logger.info('Creating new folder')
        
    if os.path.exists(val_res_dir):
        shutil.rmtree(val_res_dir)
        os.mkdir(val_res_dir)
    else:
        os.mkdir(val_res_dir)
        
    if logger:
        logger.info('Folder created at : {}'.format(str(val_res_dir)))
        logger.info('Loading validation file : {}'.format(str(val_file)))
    df_val = dfload(val_file)
    if logger:
        logger.info('len of validation file {}'.format(str(len(df_val))))
        logger.info('Columns {}'.format(str(df_val.columns)))

    val_emb = df_val['embedding']
    dim = len(val_emb.iloc[0])
    if logger:
        logger.info('Dimension of embedding : {}'.format(str(dim)))
    val_features_arr32 = {i: np.float32(df_val['embedding'].iloc[i]) for i in range(len(val_emb))}
    val_array = np.array(list(val_features_arr32.values()))
    val_files_str = [str(x) for x in df_val['event_id']]
    if logger:
        logger.info('Length of array for features : {}'.format(str(val_array.shape)))
        logger.info('Creating Binary features for Search')
    fastdup.save_binary_feature(val_res_dir, val_files_str, val_array)
    shutil.copy(os.path.join(work_dir, 'nnf.index'), val_res_dir)
    if logger:
        logger.info('Work dir :{}'.format(str(val_res_dir)))
    fastdup.run(os.path.join(work_dir, 'atrain_features.dat.csv'), work_dir=val_res_dir,test_dir = os.path.join(val_res_dir,'atrain_features.dat.csv'),nearest_neighbors_k=no_of_img_val,threshold=0.0, d=dim ,run_mode=4)
    if logger:
        logger.info('Image Search Completed. For each image {} similar images saved.'.format(str(no_of_img_val)))
    fastdup.create_duplicates_gallery(os.path.join(val_res_dir, 'similarity.csv'),save_path=val_res_dir)


def find_nearest(file,original_atrain,lookup_file,logger=None):
    """
    Find the nearest image from the original atrain file
    Input:
        file: file to be searched
        original_atrain: original atrain file
        lookup_file: lookup file
        logger: logger object
    Output:
        df: dataframe with nearest image
    """
    df = dfload(file)
    if logger:
        logger.info('file loaded')
    #df = df.sort_values(by=['distance'],ascending=False).groupby('from') 
    lookup_file = dfload(lookup_file)
    if logger:
        logger.info('lookup file loaded')

    df = df.join(lookup_file.set_index('event_id',drop=True), on='event_id',how='left')

    #df['event_id']= 123
    #df_event_id_list = [lookup_file[lookup_file['event_id']==df['to'].iloc[i]]['event_id'] for i in range(len(df['to']))]
    #df['event_id'] = df_event_id_list
    #df['event_id'] = df['to'].apply(lambda x: lookup_file[lookup_file['event_id']==x]['event_id'])
    if logger:
        logger.info('event_id added')
    

    #df['taxcode']= 'AXX'
    #df_taxcode_list = [lookup_file[lookup_file['event_id']==df['to'].iloc[i]]['taxcode'] for i in range(len(df['to']))]
    #df['taxcode'] = df_taxcode_list
    #df['taxcode'] = df['to'].apply(lambda x: lookup_file[lookup_file['event_id']==x]['taxcode'])
    #if logger:
    #    logger.info('taxcode added')
    
    dirname = os.path.dirname(file)
    if logger:
        logger.info('dirname {}'.format(str(dirname)))
    df.to_csv(dirname+'/nearest.csv',index=False)
    if logger:
        logger.info('nearest.csv saved at {}'.format(str(dirname)))

def taxcode_accuracy(val_nearest,val_ori,logger=None):
    if logger:
        logger.info('Calculating Taxcode Accuracy')
    val_nearest = dfload(val_nearest)
    val_ori = dfload(val_ori)
    if logger:
        logger.info('val_nearest : {}'.format(str(val_nearest.shape)))
        logger.info('val_ori : {}'.format(str(val_ori.shape)))
    val_nearest = val_nearest[['event_id','taxcode']]
    val_ori = val_ori[['event_id','taxcode']]
    val_nearest = val_nearest.drop_duplicates()
    val_ori = val_ori.drop_duplicates()
    if logger:
        logger.info('Merging val_nearest and val_ori')
    val_nearest = val_nearest.merge(val_ori,on='event_id')
    if logger:
        logger.info('val_nearest columns: {}'.format(str(val_nearest.columns())))
    val_nearest = val_nearest[val_nearest['taxcode_x']==val_nearest['taxcode_y']]
    acc= len(val_nearest)/len(val_ori)
    if logger:
        logger.info('Taxcode Accuracy : {}'.format(str(acc)))
    return acc


##
##    event_ids_add_similarity(path,logger=logger)


  #  fastdup_method(file_location_to_save,df_training,path_val,train_taxcode,no_of_img,features_dict,dim,local_paths,no_of_img,threshold)  

