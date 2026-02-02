import streamlit as st
import torch
import numpy as np
from PIL import Image
import datetime
import pickle
import os
import time
import hydra
from omegaconf import DictConfig
from src import utils
from hydra import compose, initialize


# --- 1. 页面配置 ---
st.set_page_config(page_title="LightFF", layout="wide")

st.markdown("""
<style>
div[data-testid="stAlert"] {
    padding: 30px;        /* 变大 */
    font-size: 18px;      /* 字体变大 → 框也变大 */
    margin-top: 24px !important;
}
</style>
""", unsafe_allow_html=True)


# --- 2. 缓存模型加载 (核心优化) ---
@st.cache_resource
def load_model_and_outputs(_opt):
    # 这里模拟你的 my_main 初始逻辑
    model, _ = utils.get_model_and_optimizer(_opt)
    model.load_state_dict(torch.load('./FF_OP_model_2.pth'))
    with open('outputs.pkl', 'rb') as file:
         outputs = pickle.load(file)
    return model, outputs

if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
    initialize(config_path=".", version_base=None)

opt = compose(config_name="config", overrides=[])
opt = utils.parse_args(opt)

model, outputs = load_model_and_outputs(opt)

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
        data=sample.clone().detach().cpu().numpy().copy()
        data = np.clip(data, 0, 1).reshape(28,28)
        # 转换 Tensor 为 PIL Image 用于显示
        img_array = np.squeeze(data* 255).astype(np.uint8)
        return Image.fromarray(img_array), label, sample
    except Exception as e:
        st.error(f"load error: {e}")
        return None, None, None

# --- 5. UI 布局 ---
st.title('🚀 Lightweight Inference for Forward-Forward Algorithm')

# 底部输入控制区
with st.container():
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 2, 1])
    with col_ctrl2:
        if st.button("Step 1: Load New Image", use_container_width=True,type="primary"):
            st.session_state.lightff_res = { "img": "./img/blank.png"}
            st.session_state.ff_res = {"img": "./img/blank.png"}
            st.session_state.order_img = (st.session_state.order_img + 1) % 1000
            st.session_state.ff_res["time"] = 0
            st.session_state.lightff_res["time"] = 0
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1.8, 2])
    with col_ctrl2:
        #user_input = st.text_input("", value=str(st.session_state.order_img + 1))
        user_input = st.number_input("or enter image index from 1 to 1000) (try 116, 248, 322, 341, 660, 957):", min_value=0, max_value=1000)
        if user_input:
            st.session_state.order_img = int(user_input) - 1
    #with col_ctrl2:
        #st.write("Or enter 1-1000 (Try: 116, 248, 322, 341, 660, 957):")
        #st.info("Try: 116, 248, 322, 341, 660, 957")

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
    if st.button("Step 2: Run FF", use_container_width=True,key="btn_ff",type="primary"):
        st.session_state.ff_res = {"img": "./img/blank.png"}
        st.session_state.ff_res["time"] = 0
        # 模拟 test_one_by_one_ff
        #start = datetime.datetime.now()
        # feedback = model.forward_downstream...(current_tensor)
        #time.sleep(0.1) # 模拟计算
        model.eval()
        with torch.no_grad():
            scalar_outputs = None
            if scalar_outputs is None:
                scalar_outputs = {
                    "Loss": torch.zeros(1, device=opt.device),
                }

            start_time = datetime.datetime.now()

            scalar_outputs,feedback = model.forward_downstream_classification_model(
                current_tensor.clone(), current_label,scalar_outputs=scalar_outputs,index=opt.model.num_layers-1
            )
            
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            st.session_state.ff_res = {"label": feedback.numpy()[0], "time": elapsed, "img": "./img/ff.png"}
            st.rerun()

    sub_col1, sub_col2,sub_col3 = st.columns([1, 2,1])
    with sub_col1:
        st.image(current_img, caption=f"Input Label: {current_label}", use_container_width=True)
    with sub_col2:
        img_slot = st.empty()
        #st.image(st.session_state.ff_res["img"],caption='',use_container_width=True)
    with sub_col3:
        info_slot = st.empty()
    img_slot.image(st.session_state.ff_res["img"],caption='',use_container_width=True)
    if st.session_state.ff_res["time"]!=0:
        info_slot.info(f"Predict Label: {st.session_state.ff_res['label']}")
            #st.info(f"Predict Label: {st.session_state.ff_res['label']}")
        #else:
            #st.info("               ")

# --- LightFF 侧 (右) ---
with col_light:
    header_col, metric_col = st.columns([1, 1])
    with header_col:
        st.header("LightFF")
    with metric_col:
        st.metric("Time Consumed", f"{st.session_state.lightff_res['time']*1000:.3f} ms")
    if st.button("Step 3:  Run LightFF", use_container_width=True,key="btn_lff",type="primary"):
        st.session_state.lightff_res = { "img": "./img/blank.png"}
        st.session_state.lightff_res["time"] = 0
        # 模拟 test_one_by_one
        #start = datetime.datetime.now()
        # feedback = model.forward_downstream_one_by_one(...)
        #time.sleep(0.05) # 模拟计算
        model.eval()
        second_layer_flag = 0
        run_layer = 0
        with torch.no_grad():
            start_time = datetime.datetime.now()
            for i in range(opt.model.num_layers):
                feedback,output = model.forward_downstream_classification_one_by_one(
                    current_tensor.clone(), current_label, scalar_outputs=outputs,index=i
                )
                if feedback == 'contine' and i!=opt.model.num_layers-1:
                    second_layer_flag = 1
                    continue
                else:
                    end_time = datetime.datetime.now()
                    if second_layer_flag==0:
                        run_layer = 1
                    elif second_layer_flag==1:
                        run_layer =2
                    break
            elapsed = (end_time - start_time).total_seconds()
        if run_layer==1:
            st.session_state.lightff_res = {"label": feedback, "time": elapsed, "img": "./img/lightff1.png"}
        elif run_layer==2:
            st.session_state.lightff_res = {"label": feedback, "time": elapsed, "img": "./img/lightff2.png"}
        st.rerun()
    sub_col3, sub_col4,sub_col5 = st.columns([1, 2,1])
    with sub_col3:
        st.image(current_img, caption=f"Input Label: {current_label}", use_container_width=True)
    with sub_col4:
        img_slot = st.empty()
        #st.image(st.session_state.lightff_res["img"],caption='',use_container_width=True)
    with sub_col5:
        info_slot = st.empty()

    img_slot.image(st.session_state.lightff_res["img"],caption='',use_container_width=True)
    if st.session_state.lightff_res["time"]!=0:
        info_slot.info(f"Predict Label: {st.session_state.lightff_res['label']}")

# --- 底部：能量节省统计 ---
#st.divider()
if st.session_state.lightff_res["time"]!=0 and st.session_state.ff_res["time"]!=0:
    saved_time = st.session_state.ff_res['time'] - st.session_state.lightff_res['time']
    if saved_time > 0:
        energy = saved_time / 3600 * 5 * 1000 * 1000 # 沿用你的公式
        st.success(f"⚡ You saved {energy:.3f} μWh Electric Energy")
