import streamlit as st
import torch
import numpy as np
from PIL import Image
import datetime
import pickle
import os
import time
# 模拟你原有的 Hydra 和 utils 导入
# from src import utils
# import hydra

# --- 1. 页面配置 ---
st.set_page_config(page_title="LightFF", layout="wide")

# --- 2. 缓存模型加载 (核心优化) ---
@st.cache_resource
def load_model_and_outputs():
    # 这里模拟你的 my_main 初始逻辑
    # model, _ = utils.get_model_and_optimizer(opt)
    # model.load_state_dict(torch.load('./FF_OP_model_2.pth'))
    # with open('outputs.pkl', 'rb') as file:
    #     outputs = pickle.load(file)
    # return model, outputs
    return "MODEL_OBJECT", "OUTPUTS_OBJECT" # 替换为真实模型

model, outputs = load_model_and_outputs()

# --- 3. 初始化 Session State (跨刷新存储变量) ---
if 'order_img' not in st.session_state:
    st.session_state.order_img = 0
if 'lightff_res' not in st.session_state:
    st.session_state.lightff_res = {"label": "→7", "time": 0, "img": "./img/blank.png"}
if 'ff_res' not in st.session_state:
    st.session_state.ff_res = {"label": "→7", "time": 0, "img": "./img/blank.png"}

# --- 4. 核心逻辑函数 ---
def load_sample(idx):
    """加载图片和标签数据"""
    try:
        all_labels_tensor = torch.load('./all_labels_tensor.pt')
        file_path = f'./testsample_mnist/testsample_{idx}.pt'
        sample = torch.load(file_path)
        label = all_labels_tensor[idx]
        
        # 转换 Tensor 为 PIL Image 用于显示
        img_array = np.squeeze(sample.numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_array), label, sample
    except Exception as e:
        st.error(f"加载失败: {e}")
        return None, None, None

# --- 5. UI 布局 ---
st.title('🚀 Lightweight Inference for Forward-Forward Algorithm (FF)')

# 底部输入控制区
with st.container():
    col_ctrl2, col_ctrl1, col_ctrl3 = st.columns([2, 1, 2])
    with col_ctrl1:
        if st.button("Load Next Image", use_container_width=True):
            st.session_state.order_img = (st.session_state.order_img + 1) % 1000
            st.session_state.ff_res["time"] = 0
            st.session_state.lightff_res["time"] = 0
    # with col_ctrl2:
    #     user_input = st.text_input("Or enter 1-1000:", value=str(st.session_state.order_img + 1))
    #     if user_input:
    #         st.session_state.order_img = int(user_input) - 1
    # with col_ctrl3:
    #     st.info("Try: 116, 248, 322, 341, 660, 957")

# 主展示区
current_img, current_label, current_tensor = load_sample(st.session_state.order_img)

col_ff, col_light = st.columns(2)

# --- FF 侧 (左) ---
with col_ff:
    header_col, metric_col = st.columns([1, 1])
    with header_col:
        st.header("FF")
    with metric_col:
        st.metric("Time Consumed", f"{st.session_state.ff_res['time']*1000:.3f} ms")
    if st.button("Run FF", use_container_width=True,key="btn_ff"):
        # 模拟 test_one_by_one_ff
        start = datetime.datetime.now()
        # feedback = model.forward_downstream...(current_tensor)
        time.sleep(0.1) # 模拟计算
        elapsed = (datetime.datetime.now() - start).total_seconds()
        st.session_state.ff_res = {"label": "Predict: 7", "time": elapsed, "img": "./img/ff.png"}
        st.rerun()

    sub_col1, sub_col2 = st.columns([1, 2])
    with sub_col1:
        st.image(current_img, caption=f"Input Label: {current_label}", use_container_width=True)
    with sub_col2:
        st.image(st.session_state.ff_res["img"],use_container_width=True)

# --- LightFF 侧 (右) ---
with col_light:
    header_col, metric_col = st.columns([1, 1])
    with header_col:
        st.header("LightFF")
    with metric_col:
        st.metric("Time Consumed", f"{st.session_state.lightff_res['time']*1000:.3f} ms")
    if st.button("Run LightFF", use_container_width=True,key="btn_lff"):
        # 模拟 test_one_by_one
        start = datetime.datetime.now()
        # feedback = model.forward_downstream_one_by_one(...)
        time.sleep(0.05) # 模拟计算
        elapsed = (datetime.datetime.now() - start).total_seconds()
        st.session_state.lightff_res = {"label": "Predict: 7", "time": elapsed, "img": "./img/lightff1.png"}
        st.rerun()
    sub_col3, sub_col4 = st.columns([1, 2])
    with sub_col3:
        st.image(current_img, caption=f"Input Label: {current_label}", use_container_width=True)
    with sub_col4:
        st.image(st.session_state.lightff_res["img"],use_container_width=True)
    

# --- 底部：能量节省统计 ---
st.divider()
saved_time = st.session_state.ff_res['time'] - st.session_state.lightff_res['time']
if saved_time > 0:
    energy = saved_time / 3600 * 5 * 1000 * 1000 # 沿用你的公式
    st.success(f"⚡ You saved {energy:.3f} μWh Electric Energy")

# import tkinter as tk
# from PIL import Image, ImageTk
# import torch
# import numpy as np
# import time
# from collections import defaultdict
# import hydra
# from omegaconf import DictConfig
# from src import utils
# import datetime
# import os 
# import sys
# import pickle
# from tkinter import messagebox

# root = tk.Tk()
# root.title('Lightweight Inference for Forward Forward Algorithm')
# root.configure(background='white')

# root.grid_rowconfigure(0, weight=1)   # 第一行扩展
# root.grid_columnconfigure(1, weight=1)  # 第二列扩展


# order_img = 0

# lightff_time = 0
# ff_time = 0

# def show_image():
#     save_time.config(text=f" ",font=("Helvetica", 20),bg="white")
#     lightff_label.config(text=f"→7",font=("Helvetica", 20),bg="white",fg="white")
#     ff_label.config(text=f"→7",font=("Helvetica", 20),bg="white",fg="white")
#     lightff_label_energy.config(text=f" ",font=("Helvetica", 20))
#     ff_label_energy.config(text=f" ",font=("Helvetica", 20))

#     image = Image.open('./img/blank.png')
#     image = image.resize((288, 162))  # 调整图片大小
#     photo = ImageTk.PhotoImage(image)
    
#     img_lightff_label.config(image=photo)
#     img_ff_label.config(image=photo)
#     img_lightff_label.image=photo
#     img_ff_label.image=photo

#     global order_img
#     all_labels_tensor = torch.load('./all_labels_tensor.pt')
#     file_path = './testsample_mnist/testsample_'+str(order_img)+'.pt'
#     labels = all_labels_tensor[order_img]
#     #result_label.config(text=f"The ground truth is {labels}  ({order_img+1} out of 1000 Image)")
#     lightff_right_label.config(text=f"Label {labels}")
#     ff_right_label.config(text=f"Label {labels}")
#     order_img+=1
#     # if order_img==1:
#     #     file_path = './testsample_mnist/testsample_115.pt'
#     # elif order_img ==2:
#     #     file_path = './testsample_mnist/testsample_248.pt'
#     sample = torch.load(file_path)
#     if file_path:
        
#         image = np.squeeze(sample.numpy()*255)
#         image = Image.fromarray(image)
#         image = image.resize((81, 81)) 
#         photo = ImageTk.PhotoImage(image)
        
#         # img_label.config(image=photo)
#         # img_label.image = photo  
#         img_ff_label_input.config(image=photo)
#         img_ff_label_input.image = photo  
#         img_lightff_label_input.config(image=photo)
#         img_lightff_label_input.image = photo  

# def test_one_by_one(opt, model,outputs):
#     global order_img
#     global lightff_time
#     file_path = './testsample_mnist/testsample_'+str(order_img-1)+'.pt'
#     inputs = torch.load(file_path)

#     # data_loader = utils.get_data(opt, "onebyone")
#     all_labels_tensor = torch.load('./all_labels_tensor.pt')
#     labels = all_labels_tensor[order_img-1]
    

#     model.eval()
#     second_layer_flag = 0
#     run_layer = 0
#     # all_labels = []
#     with torch.no_grad():
#         #for index,(inputs, labels) in enumerate(data_loader):
#         #     all_labels.extend(labels["class_labels"].numpy())
#         # all_labels_tensor = torch.tensor(all_labels)
#         # torch.save(all_labels_tensor, './all_labels_tensor.pt')


#         start_time = datetime.datetime.now()

#         for i in range(opt.model.num_layers):
#             feedback,output = model.forward_downstream_classification_one_by_one(
#                 inputs, labels, scalar_outputs=outputs,index=i
#             )

#             if feedback == 'contine' and i!=opt.model.num_layers-1:
#                 #start_time = datetime.datetime.now()
#                 second_layer_flag = 1
#                 continue
#             else:
#                 end_time = datetime.datetime.now()
#                 if second_layer_flag==0:
#                     run_layer = 1
#                 elif second_layer_flag==1:
#                     run_layer =2
                    
#                 break
        

#         elaspe_time = (end_time-start_time).total_seconds()

#         lightff_time = elaspe_time
#         #lightff_label.config(text=f"Used {pass_or_not+1} Layer(s), The Predicted Label is {feedback}, The consumed time is {elaspe_time*1000:.3f} ms",font=("Helvetica", 20))
#         lightff_label.config(text=f"→{feedback}",font=("Helvetica", 20),fg="black")
#         lightff_label_energy.config(text=f"The consumed time is {elaspe_time*1000:.3f} ms",font=("Helvetica", 20))
#         if run_layer==1:
#             image = Image.open('./img/lightff1.png')
#         elif run_layer==2:
#             image = Image.open('./img/lightff2.png')

#         image = image.resize((288, 162))  
#         photo = ImageTk.PhotoImage(image)
#         img_lightff_label.config(image=photo)
#         img_lightff_label.image = photo


# def test_one_by_one_ff(opt, model):
#     global ff_time
#     global order_img
#     file_path = './testsample_mnist/testsample_'+str(order_img-1)+'.pt'
#     inputs = torch.load(file_path)

#     all_labels_tensor = torch.load('./all_labels_tensor.pt')
#     labels = all_labels_tensor[order_img-1]
    
#     model.eval()

#     with torch.no_grad():
#         scalar_outputs = None
#         if scalar_outputs is None:
#             scalar_outputs = {
#                 "Loss": torch.zeros(1, device=opt.device),
#             }

#         start_time = datetime.datetime.now()

#         scalar_outputs,feedback = model.forward_downstream_classification_model(
#             inputs, labels,scalar_outputs=scalar_outputs,index=opt.model.num_layers-1
#         )
#         end_time = datetime.datetime.now()
#         elaspe_time = (end_time-start_time).total_seconds()
#         ff_time = elaspe_time
#         #ff_label.config(text=f"Used {opt.model.num_layers} Layers, The Predicted Label is {feedback.numpy()[0]}, The consumed time is {elaspe_time*1000:.3f} ms",font=("Helvetica", 20))
#         ff_label.config(text=f"→{feedback.numpy()[0]}",font=("Helvetica", 20),fg="black")
#         ff_label_energy.config(text=f"The consumed time is {elaspe_time*1000:.3f} ms",font=("Helvetica", 20))

#         image = Image.open('./img/ff.png')
#         image = image.resize((288, 162))  # 调整图片大小
#         photo = ImageTk.PhotoImage(image)
#         img_ff_label.config(image=photo)
#         img_ff_label.image = photo

# @hydra.main(config_path=".", config_name="config", version_base=None)
# def my_main(opt: DictConfig) -> None:
    
#     opt = utils.parse_args(opt)
#     #print(dict(opt))

#     model, optimizer = utils.get_model_and_optimizer(opt)
#     model.load_state_dict(torch.load('./FF_OP_model_2.pth'))

#     # print('Calculate the Threshold')
#     # outputs=validate_or_test_meanstd(opt, model)

#     # with open('outputs.pkl', 'wb') as file:
#     #     pickle.dump(outputs, file)

#     with open('outputs.pkl', 'rb') as file:
#         outputs = pickle.load(file)

#     print('our inference_in_main')
#     test_one_by_one(opt, model,outputs)
#     saved_time = ff_time-lightff_time #s
#     save_time.config(text=f"You save {saved_time/3600*5*1000*1000:.3f} μWh Electric Energy",font=("Helvetica", 20))


# @hydra.main(config_path=".", config_name="config", version_base=None)
# def my_main_ff(opt: DictConfig) -> None:
#     opt = utils.parse_args(opt)

#     model, optimizer = utils.get_model_and_optimizer(opt)
#     model.load_state_dict(torch.load('./FF_OP_model_2.pth'))

#     print('FF inference_in_main')
#     test_one_by_one_ff(opt, model)

# def act_to_input(event=None):
#     user_input = entry.get()
#     save_time.config(text=f" ",font=("Helvetica", 20),bg="white")
#     lightff_label.config(text=f"→7",font=("Helvetica", 20),bg="white",fg="white")
#     ff_label.config(text=f"→7",font=("Helvetica", 20),bg="white",fg="white")
#     lightff_label_energy.config(text=f" ",font=("Helvetica", 20))
#     ff_label_energy.config(text=f" ",font=("Helvetica", 20))


#     image = Image.open('./img/blank.png')
#     image = image.resize((288, 162))  # 调整图片大小
#     photo = ImageTk.PhotoImage(image)
    
#     img_lightff_label.config(image=photo)
#     img_ff_label.config(image=photo)
#     img_lightff_label.image=photo
#     img_ff_label.image=photo

#     try:
#         global order_img
#         order_img = int(user_input)-1
#         all_labels_tensor = torch.load('./all_labels_tensor.pt')
#         file_path = './testsample_mnist/testsample_'+str(order_img)+'.pt'
#         labels = all_labels_tensor[order_img]
#         #result_label.config(text=f"The ground truth is {labels}  ({order_img+1} out of 1000 Image)")
#         lightff_right_label.config(text=f"Label {labels}")
#         ff_right_label.config(text=f"Label {labels}")
#         order_img+=1
#         sample = torch.load(file_path)
#         if file_path:
            
#             image = np.squeeze(sample.numpy()*255)
#             image = Image.fromarray(image)
#             image = image.resize((81, 81)) 
#             photo = ImageTk.PhotoImage(image)
            
#             # img_label.config(image=photo)
#             # img_label.image = photo  
#             img_ff_label_input.config(image=photo)
#             img_ff_label_input.image = photo  
#             img_lightff_label_input.config(image=photo)
#             img_lightff_label_input.image = photo  
#         return 0
    
#     except ValueError:
#         messagebox.showerror("Error", "Please Input An Integer")
#     except IndexError:
#         messagebox.showerror("Error", "Please Between 1-1000")

# class ToolTip:
#     def __init__(self, widget, text):
#         self.widget = widget
#         self.text = text
#         self.tooltip_window = None
        
       
#         self.widget.bind("<Enter>", self.show_tooltip)
#         self.widget.bind("<Leave>", self.hide_tooltip)

#     def show_tooltip(self, event=None):
#         if self.tooltip_window is not None:
#             return
        
        
#         x = self.widget.winfo_rootx() + 20
#         y = self.widget.winfo_rooty() + 20
#         self.tooltip_window = tk.Toplevel(self.widget)
#         self.tooltip_window.wm_overrideredirect(True)  
#         self.tooltip_window.wm_geometry(f"+{x}+{y}")  
#         label = tk.Label(self.tooltip_window, text=self.text, bg="lightyellow", borderwidth=1, relief="solid")
#         label.pack()

#     def hide_tooltip(self, event=None):
#         if self.tooltip_window is not None:
#             self.tooltip_window.destroy()
#             self.tooltip_window = None

# def run_function():
#     my_main()
    
# def run_function_ff():
#     my_main_ff()




# # frame_load = tk.Frame(root,bg="white")
# # frame_load.pack(pady=10)

# # result_label = tk.Label(frame_load, text=" ",font=("Helvetica", 20),bg="white")
# # result_label.pack(side=tk.RIGHT,pady=10)

# # img_label = tk.Label(frame_load,bg="white")
# # img_label.pack(side=tk.RIGHT,pady=10)



# frame_ff = tk.Frame(root,bg="white",width=200, height=200)
# frame_ff.pack(side='left',pady=150,expand=True,padx=(100,0),anchor='n')

# run_button = tk.Button(frame_ff, text="FF", font=("Helvetica", 30,"bold"), command=run_function_ff,bg="white",width=20)
# run_button.grid(row=0, column=1, padx=0)

# ff_label = tk.Label(frame_ff, text=" ",font=("Helvetica", 20),bg="white")  
# ff_label.grid(row=1, column=2, padx=0) 

# ff_right_label = tk.Label(frame_ff, text=" ",font=("Helvetica", 20),bg="white")  
# ff_right_label.grid(row=2, column=0, padx=0) 

# img_ff_label_input = tk.Label(frame_ff,bg="white")
# img_ff_label_input.grid(row=1, column=0, padx=0)

# img_ff_label = tk.Label(frame_ff,bg="white")
# img_ff_label.grid(row=1, column=1, padx=0)

# ff_label_energy = tk.Label(frame_ff, text=" ",font=("Helvetica", 20),bg="white")  
# ff_label_energy.grid(row=2, column=1, padx=0) 


# frame = tk.Frame(root,bg="white",width=200, height=200)
# frame.pack(side='right',pady=150,expand=True,padx=(0, 100),anchor='n')

# run_button = tk.Button(frame, text="LightFF", font=("Helvetica", 30,"bold"), command=run_function,bg="white",width=20)
# run_button.grid(row=0, column=1, padx=0) 

# lightff_label = tk.Label(frame, text=" ",font=("Helvetica", 20),bg="white") 
# lightff_label.grid(row=1, column=2, padx=0) 

# lightff_label_energy = tk.Label(frame, text=" ",font=("Helvetica", 20),bg="white") 
# lightff_label_energy.grid(row=2, column=1, padx=0) 

# lightff_right_label = tk.Label(frame, text=" ",font=("Helvetica", 20),bg="white")  
# lightff_right_label.grid(row=2, column=0, padx=0) 

# img_lightff_label_input = tk.Label(frame,bg="white")
# img_lightff_label_input.grid(row=1, column=0, padx=0)


# img_lightff_label = tk.Label(frame,bg="white")
# img_lightff_label.grid(row=1, column=1, padx=0)

# # frame_time = tk.Frame(root,bg="white")
# # frame_time.pack(side='bottom',pady=50)

# save_time = tk.Label(frame, text=f" ",font=("Helvetica", 30),bg="white")  
# save_time.grid(row=3, column=1, padx=0)
 


# frame_input = tk.Frame(root,bg="white")
# frame_input.pack(side='bottom',pady=50)

# img_button = tk.Button(root, text="Load An Image", font=("Helvetica", 30,"bold"), command=show_image,bg="white",width=20)
# #img_button.grid(row=0, column=1, padx=0)
# img_button.pack(side='bottom',pady=10)


# input_label = tk.Label(frame_input, text="or enter",font=("Helvetica", 20),bg="white")
# input_label.pack(side=tk.LEFT,pady=10)
# #input_label.grid(row=1, column=0, padx=0)

# entry = tk.Entry(frame_input, width=10)
# entry.pack(side=tk.LEFT,pady=10)
# #entry.grid(row=1, column=1, padx=0)

# enter_label = tk.Label(frame_input, text="from 1-1000",font=("Helvetica", 20),bg="white")
# enter_label.pack(side=tk.LEFT,pady=10)
# #enter_label.grid(row=1, column=2, padx=5)

# tooltip = ToolTip(entry, "Try 116, 248, 322, 341, 382, 446, 496, 583, 584, 660, 685, 692, 721, 883, 948, 957, 966")

# entry.bind("<Return>", act_to_input)



# root.mainloop()