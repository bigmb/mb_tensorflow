import gradio as gr
from mt.pandas import dfload
from PIL import Image
import numpy as np
import argparse
from wml.core import s3
import cv2
import ast
from mb_pandas.src.transform import rename_columns as mb_rc

def main(args):

    # Load the Parquet file using Polars
    df = dfload(args.file)
    limit = args.limit
    if limit is not None:
        df = df[:limit]
                
    after_image_url = args.after_image_url
    before_image_url = args.before_image_url
    local_path = args.local_path
    bounding_box = args.bounding_box

    k =''
    if after_image_url in df.columns:
        k=k+'after_'
        df = mb_rc(df,'after_image_url',after_image_url)
    if before_image_url in df.columns:
        k=k+'before_'
        df = mb_rc(df,'before_image_url',before_image_url)
    if local_path in df.columns:
        k=k+'url'
        df = mb_rc(df,'local_path',local_path)

    # Define a function to display images in Gradio
    def display_image(image,bounding_box=None):
        img1 = np.array(Image.open(image))
        if bounding_box is not None:
            bounding_box = ast.literal_eval(df[df[after_image_url]==image]['bounding_box'].iloc[0])
            img1 = cv2.rectangle(img1, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 255, 0), 2)
        return img1

    # Define a function to extract images from the Parquet file and return them for display in Gradio
    def extract_images(df,k,value_int,value_txt,value_type):

        #find value in value_type in df:
        if value_type == before_image_url or value_type == after_image_url or value_type == local_path:
            value = value_txt
        else:
            value = value_int
        df_temp = df[df[value_type]==value]
        img_temp = []
        if 'a' in k:
            if df_temp[after_image_url].iloc[0].startswith('s3://') or df_temp[after_image_url].iloc[0].startswith('http'):
                img_temp.append(display_image(s3.cache_localpath(df_temp[after_image_url].iloc[0],bounding_box=bounding_box)))
            else:
                img_temp.append(display_image(df_temp[after_image_url].iloc[0]))
        if 'b' in k:
            if df_temp[before_image_url].iloc[0].startswith('s3://') or df_temp[before_image_url].iloc[0].startswith('http'):
                img_temp.append(display_image(s3.cache_localpath(df_temp[before_image_url].iloc[0])))
            else:
                img_temp.append(display_image(df_temp[before_image_url].iloc[0]))
        if 'u' in k:
            if df_temp[local_path].iloc[0].startswith('s3://') or df_temp[local_path].iloc[0].startswith('http'):
                img_temp.append(display_image(s3.cache_localpath(df_temp[local_path].iloc[0])))
            else:
                img_temp.append(display_image(df_temp[local_path].iloc[0]))
        img_temp_np = np.array(img_temp)
        img_temp_np_h = np.hstack(img_temp_np)
        img_output = Image.fromarray(img_temp_np_h)
        if 'gt_taxcode' in df.columns:
            taxcode = df_temp['gt_taxcode'].iloc[0]
        elif 'taxcode' in df.columns:
            taxcode = df_temp['taxcode'].iloc[0]
        else:
            taxcode = 'None'
        if 'predictions' in df.columns:
            pred_taxcode = df_temp['predictions'].iloc[0]
        else:
            pred_taxcode = 'None'
        return img_output,k,taxcode,pred_taxcode


    # Define the output type for the Gradio interface
    def main_func(text1,text2,dropdown):
        img,k1,tx,pred_tx = extract_images(df,k,text1,text2,dropdown)
        return img,k1,tx,pred_tx
    
    options = ['event_id','after_image_id','after_image_url','image_id','before_image_id','before_image_url','local_path']

    # Define the Gradio interface
    inf =gr.Interface(fn=main_func,
        inputs=[gr.inputs.Number(label='Id'),gr.inputs.Textbox(label='Url') ,gr.inputs.Dropdown(options,label='Input type')],
        outputs=[gr.Image(shape=(300,200)),gr.outputs.Textbox(label='images_type'),gr.outputs.Textbox(label='Taxcode'),gr.outputs.Textbox(label='Predicted Taxcode')])

    inf.launch(share=True,server_port=args.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='/home/malav/phong_dataset_search/final_file.parquet')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--bounding_box', type=str, default=None)
    parser.add_argument('--after_image_url', type=str, default='after_image_url')
    parser.add_argument('--before_image_url', type=str, default='before_image_url')
    parser.add_argument('--local_path', type=str, default='local_path')
    parser.add_argument('--port', type=int, default=8908)
    args = parser.parse_args()
    main(args)
