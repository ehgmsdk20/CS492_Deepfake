#!/usr/bin/env python
# coding: utf-8

# <a id='1'></a>
# # Import modules

# In[1]:


import keras.backend as K
import os

# <a id='4'></a>
# # Model Configuration

# In[2]:


K.set_learning_phase(0)

file_dir=os.path.dirname(os.path.abspath(__file__))
# In[3]:


# Input/Output resolution
RESOLUTION = 64 # 64x64, 128x128, 256x256
assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, 256"


# In[4]:


# Architecture configuration
arch_config = {}
arch_config['IMAGE_SHAPE'] = (RESOLUTION, RESOLUTION, 3)
arch_config['use_self_attn'] = True
arch_config['norm'] = "instancenorm" # instancenorm, batchnorm, layernorm, groupnorm, none
arch_config['model_capacity'] = "standard" # standard, lite


# <a id='5'></a>
# # Define models

# In[5]:


from deepfake.networks.faceswap_gan_model import FaceswapGANModel


# In[6]:

def all_thing(input):
    model = FaceswapGANModel(**arch_config)


    # <a id='6'></a>
    # # Load Model Weights

    # In[7]:

    models_dir = os.path.join("/home/ubuntu/cs320/media", input.cleaned_data['useremail'], "models")
    model.load_weights(path= models_dir)


    # <a id='12'></a>
    # # Video Conversion

    # In[8]:


    from deepfake.converter.video_converter import VideoConverter
    from deepfake.detector.face_detector import MTCNNFaceDetector


    # In[9]:


    mtcnn_weights_dir = os.path.join(file_dir,"mtcnn_weights")

    fd = MTCNNFaceDetector(sess=K.get_session(), model_path=mtcnn_weights_dir)
    vc = VideoConverter()


    # In[10]:


    vc.set_face_detector(fd)
    vc.set_gan_model(model)


    # ### Video conversion configuration
    # 
    # 
    # - `use_smoothed_bbox`: 
    #     - Boolean. Whether to enable smoothed bbox.
    # - `use_kalman_filter`: 
    #     - Boolean. Whether to enable Kalman filter.
    # - `use_auto_downscaling`:
    #     - Boolean. Whether to enable auto-downscaling in face detection (to prevent OOM error).
    # - `bbox_moving_avg_coef`: 
    #     - Float point between 0 and 1. Smoothing coef. used when use_kalman_filter is set False.
    # - `min_face_area`:
    #     - int x int. Minimum size of face. Detected faces smaller than min_face_area will not be transformed.
    # - `IMAGE_SHAPE`:
    #     - Input/Output resolution of the GAN model
    # - `kf_noise_coef`:
    #     - Float point. Increase by 10x if tracking is slow. Decrease by 1/10x if trakcing works fine but jitter occurs.
    # - `use_color_correction`: 
    #     - String of "adain", "adain_xyz", "hist_match", or "none". The color correction method to be applied.
    # - `detec_threshold`: 
    #     - Float point between 0 and 1. Decrease its value if faces are missed. Increase its value to reduce false positives.
    # - `roi_coverage`: 
    #     - Float point between 0 and 1 (exclusive). Center area of input images to be cropped (Suggested range: 0.85 ~ 0.95)
    # - `enhance`: 
    #     - Float point. A coef. for contrast enhancement in the region of alpha mask (Suggested range: 0. ~ 0.4)
    # - `output_type`: 
    #     - Layout format of output video: 1. [ result ], 2. [ source | result ], 3. [ source | result | mask ]
    # - `direction`: 
    #     - String of "AtoB" or "BtoA". Direction of face transformation.

    # In[11]:


    options = {
        # ===== Fixed =====
        "use_smoothed_bbox": True,
        "use_kalman_filter": True,
        "use_auto_downscaling": False,
        "bbox_moving_avg_coef": 0.65,
        "min_face_area": 35 * 35,
        "IMAGE_SHAPE": model.IMAGE_SHAPE,
        # ===== Tunable =====
        "kf_noise_coef": 3e-3,
        "use_color_correction": "hist_match",
        "detec_threshold": 0.7,
        "roi_coverage": 0.9,
        "enhance": 0.,
        "output_type": 3,
        "direction": "AtoB",
    }


    # # Start video conversion
    # 
    # 
    # - `input_fn`: 
    #     - String. Input video path.
    # - `output_fn`: 
    #     - String. Output video path.
    # - `duration`: 
    #     - None or a non-negative float tuple: (start_sec, end_sec). Duration of input video to be converted
    #     - e.g., setting `duration = (5, 7.5)` outputs a 2.5-sec-long video clip corresponding to 5s ~ 7.5s of the input video.

    # In[12]:


    input_fn = "/home/ubuntu/cs320/deepfake/손3.mp4"
    output_fn = os.path.join("/home/ubuntu/cs320/media", input.cleaned_data['useremail'],"OUTPUT_VIDEO.mp4")
    duration = None 


# In[13]:


    vc.convert(input_fn=input_fn, output_fn=output_fn, options=options, duration=(0, 15))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




