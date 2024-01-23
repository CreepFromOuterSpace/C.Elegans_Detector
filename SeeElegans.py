#Library imports
import numpy as np
import streamlit as st
import cv2
import torch
import tensorflow as tf
from keras.preprocessing import image
from io import BytesIO
import onnxruntime as rt

#Setting up the page
#st.image('logo.png')
st.title("See Elegans: **Automating _C. elegans_ Cataloging**")
st.markdown("""
This app uses two trained models, Model 1(object detector) finds the *C. elegans*. An intermediary crops these images to send to Model 2(image classifier), which determines which life stage the worm is at.

**Made by Vaibhav Rastogi & Anshul Rastogi**

""")

#Loading Model 1
model=torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.eval()
#Loading Model 2
session=rt.InferenceSession("/content/model2.onnx")

#Setting up image for bounding box display
font=cv2.FONT_HERSHEY_SIMPLEX

#Image upload widget
img=st.file_uploader("Upload an image...", type=['png','jpg','webp'])
if img!=None:
  #Preview the image (so that the user can check if they uploaded the right one)
  st.image(img,caption="Preview",width=500)
  conf_threshold=st.slider("Confidence Threshold",
                           min_value=0.00,
                           max_value=1.00,
                           value=model.conf,
                           step=0.01,
                           format="%.2f")
  #To run inference on image
  if st.button('Find & Count'):
    #Convert the file to an opencv image
    file_bytes=np.asarray(bytearray(img.read()),dtype=np.uint8)
    img=cv2.imdecode(file_bytes,1)
    #Runs the model on the image
    
    classes_to_counts=lambda classes:{cls:0 for cls in classes}
    model1_counts=classes_to_counts(["Worm","Eggs"])
    model2_counts=classes_to_counts(["Null","L1-L2","L3-L4","Adult"])#["Adult","Eggs","Unlabeled","Worm"]
    model1_colors=dict(zip(list(model1_counts),[(255,0,0),(0,0,255)]))
    
    model.conf=conf_threshold
    results=model(img)
    #Saving the results
    crops=results.crop(save=True)
    #counts_updater(results,model1_counts)
    result_img=img.copy()
    for cropped in crops:
      obj_label=cropped["label"].split(" ")[0]
      model1_counts[obj_label]+=1
      box=[round(float(coord)) for coord in cropped["box"]]
      cv2.rectangle(result_img,(box[0],box[1]),(box[2],box[3]),model1_colors[obj_label],5)
      final_label=cropped["label"]
      if obj_label==list(model1_counts)[0]:
        #Running Model 2 on individual images
        onnx_pred=session.run(None,{"sequential_input":np.expand_dims(cv2.resize(cropped["im"],(416,416)),axis=0).astype("float32")})
        score=tf.nn.softmax(onnx_pred[0])
        label=list(model2_counts)[np.argmax(score)]
        conf=100*np.max(score)
        model2_counts[label]+=1
        final_label=final_label.replace(obj_label,label)
        #Viewing cropped objects (for debugging):
        #st.image(cropped["im"],width=200,caption=label)
      cv2.putText(result_img,final_label,(box[0],box[1]-7),fontFace=font,fontScale=3,color=model1_colors[obj_label],thickness=4)
    #Displaying the results
    for cls,num in model1_counts.items():
      st.markdown(f"**{cls}**: {str(num)}")
    for cls,num in model2_counts.items():
      st.markdown(f"**{cls}**: {str(num)}")
    #if st.button('View Results'):
    st.image(result_img,caption="Results",width=500,channels="BGR")

st.markdown("""**References:**
Ifeanyi Nneji [GitHub](https://github.com/Nneji123)""")
