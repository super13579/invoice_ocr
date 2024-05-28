import streamlit as st
import os
import shutil

import paddleocr
from paddleocr import PaddleOCR, draw_ocr

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import json
import cv2
from glob import glob
from pdf2image import convert_from_path
import cv2, zxingcpp

import tempfile
import datetime

ocr = PaddleOCR(lang='chinese_cht')
convert_dpi = 800

def seven_fetch(pdf_path):
    imgs = convert_from_path(pdf_path, dpi=convert_dpi, poppler_path=r'C:\PDF_fetch\poppler-24.02.0\Library\bin')
    imgs[0].save("./out.png")
    input_img = cv2.imread("./out.png")
    img = CropImage(input_img)
    # plt.imshow(img)
    # plt.show()
    
    results = zxingcpp.read_barcodes(img)
    position_1 = str(results[0].position)
    position_2 = str(results[1].position)
    y1 = int(position_1.split(' ')[2].split('x')[1])
    y2 = int(position_2.split(' ')[0].split('x')[1])
    
    target_txt_1 = Iteration_Scan_711(img, y1, y2)
    target_txt_2 = OCR(img, '編號')
    
    if ':' in target_txt_1:
        target_txt_2 = target_txt_2.split(':')[-1]
    else:
        target_txt_2 = target_txt_2.split('：')[-1]
    
    target_txt_1 = target_txt_1.replace(":","")

    return [target_txt_1, target_txt_2]
    
    # print('----------------------------------------')
    # print(target_txt_1)
    # print(target_txt_2)

def family_fetch(pdf_path):
    imgs = convert_from_path(pdf_path, dpi=convert_dpi, poppler_path=r'C:\PDF_fetch\poppler-24.02.0\Library\bin')
    imgs[0].save("./out.png")
    input_img = cv2.imread("./out.png")
    # input_img = cv2.imread(df_Family.iloc[3].Path)
    img = CropImage(input_img)
    # plt.imshow(img)
    # plt.show()
    
    Target = OCR(img, '訂單編號')
    if ':' in Target:
        Target = Target.split(':')[-1]
    else:
        Target = Target.split('：')[-1]
    
    return Target

def OK_fetch(pdf_path):
    imgs = convert_from_path(pdf_path, dpi=convert_dpi, poppler_path=r'C:\PDF_fetch\poppler-24.02.0\Library\bin')
    imgs[0].save("./out.png")

    input_img = cv2.imread("./out.png")
    # input_img = cv2.imread(df_OK.iloc[0].Path)
    img = CropImage(input_img)
    # plt.imshow(img)
    # plt.show()
    results = zxingcpp.read_barcodes(img)
    result = results[-1]

    return result.text
    # print('----------------------------------------')
    # print(result.text)

def hilife_fetch(pdf_path):
    imgs = convert_from_path(pdf_path, dpi=convert_dpi, poppler_path=r'C:\PDF_fetch\poppler-24.02.0\Library\bin')
    imgs[0].save("./out.png")

    input_img = cv2.imread("./out.png")
    # input_img = cv2.imread(df_HiLife.iloc[5].Path)
    img = CropImage(input_img)
    # plt.imshow(img)
    # plt.show()
    
    results = zxingcpp.read_barcodes(img)
    position = str(results[2].position)
    y1 = int(position.split(' ')[0].split('x')[1])
    y2 = int(position.split(' ')[2].split('x')[1])
    
    # img = img[int(y1):int(y2), :]
    Target = Iteration_Scan_Hilife(img, y1, y2)
    if ':' in Target:
        Target = Target.split(':')[-1]
    else:
        Target = Target.split('：')[-1]
        
    return Target

def shopee_fetch(pdf_path):
    imgs = convert_from_path(pdf_path, dpi=convert_dpi, poppler_path=r'C:\PDF_fetch\poppler-24.02.0\Library\bin')
    imgs[0].save("./out.png")

    input_img = cv2.imread("./out.png")
    # input_img = cv2.imread(df_Shopee.iloc[0].Path)
    img = CropImage(input_img)
    # plt.imshow(img)
    # plt.show()
    results = zxingcpp.read_barcodes(img)
    result = results[-1]

    return result.text
    # print('----------------------------------------')
    # print(result.text)

def pdf2png(file_arr, out_dir, dpi):
    
    for idx in range(len(file_arr)):
        imgs = convert_from_path(file_arr[idx], dpi=dpi)
        imgs[0].save(f'{out_dir}/dpi{dpi}/{file_arr[idx].split("/")[3].replace("pdf", "png")}')
        

def category_dataframe(img_arr):
    
    ### Category
    Pid = []
    Path = []
    Type = []
    for idx in range(len(img_arr)):
        Pid.append(img_arr[idx].split('/')[-1].split('.')[0])
        Path.append(img_arr[idx])

        image = cv2.imread(img_arr[idx])
        image = np.array(image/255, dtype=np.uint8)
        image = image*255
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        kernel = np.ones((25,25), np.uint8)
        dilate = cv2.dilate(thresh, kernel, iterations=1)
        kernel = np.ones((10,10), np.uint8)
        erode = cv2.erode(dilate, kernel, iterations=1)

        image_det = image.copy()
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        areas = [cv2.contourArea(c) for c in cnts]
        max_index = np.argmax(areas)
        x,y,w,h = cv2.boundingRect(cnts[max_index])

        result = ocr.ocr(image_det[y:y+h, x:x+w], cls=False)
        if result[0] is None:
            Type.append('BLUR')
            continue
        txts = [line[1][0] for line in result[0]]
        if '統一超商' in txts:
            Type.append('7-11')
        elif 'Hi-Life' in txts:
            Type.append('Hi-Life')
        elif 'OK寄件' in txts:
            Type.append('OK')
        else:
            for txt in txts:
                if 'Mart' in txt:
                    Type.append('FamilyMart')
                    break
        if len(Type) == idx:
            Type.append('Shopee')
        print(Type[-1])
        
    
    ### Create DataFrame
    Receipt_df = pd.DataFrame(list(zip(Pid, Path, Type)), columns=['Pid', 'Path', 'Type'])
    
    return Receipt_df

def CropImage(input_img):
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = np.ones((35,35), np.uint8)
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    kernel = np.ones((40,40), np.uint8)
    erode = cv2.erode(dilate, kernel, iterations=1)

    image_det = input_img.copy()

    cnts = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    areas = [cv2.contourArea(c) for c in cnts]
    max_index = np.argmax(areas)
    x,y,w,h = cv2.boundingRect(cnts[max_index])

    return image_det[y:y+h, x:x+w]

def OCR(img, key_word):
    result = ocr.ocr(img, cls=False)
    txts = [line[1][0] for line in result[0]]
    for i in range(len(txts)):
        if key_word in txts[i]:
            return txts[i]
        
def Iteration_Scan_Hilife(img, y1, y2):
    for pad in range(0, 800, 100):
        result = ocr.ocr(img[int(y1)-pad:int(y2)+pad, :], cls=False)
        txts = [line[1][0] for line in result[0]]
        for i in range(len(txts)):
            if '代碼' in txts[i]:
                return txts[i]
            
def Iteration_Scan_711(img, y1, y2):
    for pad in range(0, 800, 100):
        result = ocr.ocr(img[int(y1)+pad:int(y2)-pad, :], cls=False)
        txts = [line[1][0] for line in result[0]]
        for i in range(len(txts)):
            if '代碼' in txts[i]:
                return txts[i+1]

def format_number_to_three_digits(number):
    # Using format() to pad the number with leading zeros up to three digits
    return "{:03}".format(number)

# Function to process files based on invoice type, user ID, and customer ID
def process_files(files, invoice_type, user_id, customer_id):
    results = []
    results1 = []
    user_id_list = []
    cus_id_list = []
    upload_list = []
    download_list = []
    store_list = []
    status_list = []
    current_date = datetime.datetime.now()

    # Format the date as YYYYMMDD
    formatted_date = current_date.strftime("%Y%m%d")

    if invoice_type == "7-11":
        upload_text = "71"+customer_id+user_id+formatted_date
        store_id = 1
    elif invoice_type == "OK":
        upload_text = "OK"+customer_id+user_id+formatted_date
        store_id = 4
    elif invoice_type == "Hi-Life":
        upload_text = "Hi"+customer_id+user_id+formatted_date
        store_id = 3
    elif invoice_type == "全家":
        upload_text = "FM"+customer_id+user_id+formatted_date
        store_id = 2
    elif invoice_type == "蝦皮":
        upload_text = "SH"+customer_id+user_id+formatted_date
        store_id = 5

    for idx, file in enumerate(files):
        result_upload_text = upload_text+"-"+format_number_to_three_digits(idx+1)+"UP.zip"
        result_down_text = upload_text+"-"+format_number_to_three_digits(idx+1)+"DN.xlsx" 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            print (file)
            # Write the uploaded file to a new file on disk
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        # Simulate processing (replace this with your actual processing logic)
        if invoice_type == "7-11":
            out_text = seven_fetch(tmp_path)
        elif invoice_type == "OK":
            out_text = OK_fetch(tmp_path)
        elif invoice_type == "Hi-Life":
            out_text = hilife_fetch(tmp_path)
        elif invoice_type == "全家":
            out_text = family_fetch(tmp_path)
        elif invoice_type == "蝦皮":
            out_text = shopee_fetch(tmp_path)

        # result = f"Processed {file.name} for user {user_id} and customer {customer_id} with invoice type {invoice_type}"
        # results.append(out_text)
        user_id_list.append(user_id)
        cus_id_list.append(customer_id)
        upload_list.append(result_upload_text)
        download_list.append(result_down_text)
        store_list.append(store_id)
        status_list.append(1)
        if invoice_type == "7-11":
            results.append(out_text[0])
            results1.append(out_text[1])
        else:
            results.append(out_text)

    empty_list_length = len(user_id_list)  # Assuming user_id_list is correct
    empty_list = [None] * empty_list_length

    if invoice_type == "7-11":
        out_pd = pd.DataFrame({"user_id":user_id_list, "customer_id":cus_id_list, "upload_file_name":upload_list,
             "download_file_name":download_list, "store_id":store_list, "711_data_1":results, "711_data_2":results1, "family_dtata_1":empty_list,
             "hilife_dtata_1":empty_list,"ok_dtata_1":empty_list,"shopee_dtata_1":empty_list,"upload_date":empty_list,"upload_time":empty_list,"build_date":empty_list,
             "build_time":empty_list,"status":status_list})
    elif invoice_type == "OK":
        out_pd = pd.DataFrame({"user_id":user_id_list, "customer_id":cus_id_list, "upload_file_name":upload_list,
             "download_file_name":download_list, "store_id":store_list, "711_data_1":empty_list, "711_data_2":empty_list, "family_dtata_1":empty_list,
             "hilife_dtata_1":empty_list,"ok_dtata_1":results,"shopee_dtata_1":empty_list,"upload_date":empty_list,"upload_time":empty_list,"build_date":empty_list,
             "build_time":empty_list,"status":status_list})
    elif invoice_type == "Hi-Life":
        out_pd = pd.DataFrame({"user_id":user_id_list, "customer_id":cus_id_list, "upload_file_name":upload_list,
             "download_file_name":download_list, "store_id":store_list, "711_data_1":empty_list, "711_data_2":empty_list, "family_dtata_1":empty_list,
             "hilife_dtata_1":results,"ok_dtata_1":empty_list,"shopee_dtata_1":empty_list,"upload_date":empty_list,"upload_time":empty_list,"build_date":empty_list,
             "build_time":empty_list,"status":status_list})
    elif invoice_type == "全家":
        out_pd = pd.DataFrame({"user_id":user_id_list, "customer_id":cus_id_list, "upload_file_name":upload_list,
             "download_file_name":download_list, "store_id":store_list, "711_data_1":empty_list, "711_data_2":empty_list, "family_dtata_1":results,
             "hilife_dtata_1":empty_list,"ok_dtata_1":empty_list,"shopee_dtata_1":empty_list,"upload_date":empty_list,"upload_time":empty_list,"build_date":empty_list,
             "build_time":empty_list,"status":status_list})
    elif invoice_type == "蝦皮":
        out_pd = pd.DataFrame({"user_id":user_id_list, "customer_id":cus_id_list, "upload_file_name":upload_list,
             "download_file_name":download_list, "store_id":store_list, "711_data_1":empty_list, "711_data_2":empty_list, "family_dtata_1":empty_list,
             "hilife_dtata_1":empty_list,"ok_dtata_1":empty_list,"shopee_dtata_1":results,"upload_date":empty_list,"upload_time":empty_list,"build_date":empty_list,
             "build_time":empty_list,"status":status_list})

    return out_pd

st.set_page_config(
    page_title="Invoice OCR",
)


# Set up the title of the application
st.title("發票OCR")

# Check if 'uploaded_files' is in the session state and clear it if needed
if 'uploaded_files' not in st.session_state or st.session_state.clear_files:
    st.session_state.uploaded_files = None
    st.session_state.clear_files = False

# Create a file uploader to accept multiple files
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, key="uploader")

# If new files are uploaded, update the session state
if uploaded_files is not None:
    st.session_state.uploaded_files = uploaded_files
    st.session_state.clear_files = True

# # Create a file uploader to accept multiple files
# uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

# Input for specifying the type of invoice
invoice_type = st.selectbox("發票類型", ["7-11", "蝦皮", "OK", "Hi-Life", "全家"])

# Text input for user ID and customer ID
user_id = st.text_input("User ID")
customer_id = st.text_input("Customer ID")

# Button to process files
if st.button("Process Files"):
    if uploaded_files and invoice_type and user_id and customer_id:
        # Call the process_files function when the button is clicked
        processing_results = process_files(uploaded_files, invoice_type, user_id, customer_id)
        # for result in processing_results:
        #     st.success(result)

        if not processing_results.empty:
            # Display the DataFrame in the app
            st.dataframe(processing_results)
        else:
            st.error("No data to display. Please check the input files and parameters.")
    else:
        st.error("Please fill in all fields before processing.")

# # Displaying uploaded files and the selected invoice type
# if uploaded_files:
#     for uploaded_file in uploaded_files:
#         st.write("Filename:", uploaded_file.name)
#     st.write("Selected Invoice Type:", invoice_type)
#     st.write("User ID:", user_id)
#     st.write("Customer ID:", customer_id)