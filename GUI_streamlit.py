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

