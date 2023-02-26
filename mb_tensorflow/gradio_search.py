import gradio as gr
from mt.pandas import dfload
from PIL import Image
import numpy as np
import argparse
from wml.core import s3
import os
import json

def main(args):

    # Load the Parquet file using Polars
    folder = args.folder

    #local_path = args.local_path

    # Define a function to display images in Gradio
    def display_image(image):
        return Image.open(image)

    # Define a function to extract images from the Parquet file and return them for display in Gradio
    def extract_images(df,value,value_type,no_of_images=3):

        #find value in value_type in df:
        df_temp = df[df[value_type]==value]
        img_temp = []
        img_temp.append(display_image(df_temp[local_path].iloc[0]))
        img_temp_np = np.array(img_temp)
        img_temp_np_h = np.hstack(img_temp_np)
        img_output = Image.fromarray(img_temp_np_h)
        return img_output,dist

    def folder_files(folder):
        isinstance(folder,str),f"folder must be a string, not {type(folder)}"
        isinstance(folder,os.path.exists(folder)),f"folder {folder} does not exist"
        data_similar = dfload(folder+'/similarity.csv')
        data_accuracy = dfload(folder+'/accuracy.csv')
        data_outlier = dfload(folder+'/outliers.csv')
        data_config = json.load(folder+'/config.csv')                       
        return data_similar,data_accuracy,data_outlier,data_config
    
    def data_wrangle(data_similar,data_accuracy,data_outlier,data_config):
        

        return final_df


    # Define the output type for the Gradio interface
    def main_func(text1,dropdown,text2):
        img,k = extract_images(text1,dropdown,text2)
        return img
    
    options = ['event_id','after_image_id','url']

    # Define the Gradio interface
    inf =gr.Interface(fn=main_func,
        inputs=['text',gr.inputs.Dropdown(options,label='input type'),'text'],
        outputs=[gr.Image(shape=(300,200)),'text','text'])

    inf.launch(share=True,server_port=args.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='/home/malav/phong_dataset_search/image_search2')
    #parser.add_argument('--local_path', type=str, default='local_path')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    main(args)
