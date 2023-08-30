import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.title("Automobile data case study")
st.write("## Automobile price prediction")
st.image("A.png")
#To accept data from user 
#numeric data
symbolizing=[-2,-1,0,1,2,3]
sym=st.selectbox('Symbolic',symbolizing,index=symbolizing.index(-2))
nl=st.number_input('Normalsed losses')
w=st.number_input('Width')
h=st.number_input('Height')
es=st.number_input('Engine Size',format='%d')#integer
hp=st.number_input('Horse power')#integer
cm=st.number_input('City Milage',format='%d')
hm=st.number_input('Highway Milage',format='%d')

#string Data
brand=['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
       'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury',
       'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche', 'renault',
       'saab', 'subaru', 'toyota', 'volkswagen', 'volvo']
make=st.selectbox('Select car brand',brand,index=brand.index('alfa-romero'))#object type
fuel=['gas', 'diesel']
ft=st.selectbox('Select fuel type',fuel,index=fuel.index('gas'))#object type
body=['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop']
bs=st.selectbox('Body Style',body,index=body.index('convertible'))#object type
drive=['rwd', 'fwd', '4wd']
dw=st.selectbox('Drive wheels',drive,index=drive.index('rwd'))#object type
enginel=['front', 'rear']
el=st.selectbox('Engine Location',enginel,index=enginel.index('front'))#object type
enginet=['dohc', 'ohcv', 'ohc', 'l', 'rotor', 'ohcf', 'dohcv']
et=st.selectbox('Engine type',enginet,index=enginet.index('dohc'))#object type


if st.button("Predict"):
    import warnings
    warnings.filterwarnings('ignore')
    file1=open('scale1.pkl','rb') # here scale.pkl exist and file1 is a temporary file
    file2=open('model1.pkl','rb') #here model.pkl exist and file2 is a temporary file
    file3_1=open("LabelEncoder1.pkl","rb")
    file3_2=open("LabelEncoder2.pkl","rb")
    file3_3=open("LabelEncoder3.pkl","rb")
    file3_4=open("LabelEncoder4.pkl","rb")
    file3_5=open("LabelEncoder5.pkl","rb")
    file3_6=open("LabelEncoder6.pkl","rb")

    import pickle
    #To read data from file , use inbuilt method load() of pickle library
    scale=pickle.load(file1) #scale user defined variable
    model=pickle.load(file2)#model user defined variable
    e_label1=pickle.load(file3_1)
    e_label2=pickle.load(file3_2)
    e_label3=pickle.load(file3_3)
    e_label4=pickle.load(file3_4)
    e_label5=pickle.load(file3_5)
    e_label6=pickle.load(file3_6)

    import numpy as np
    x1=e_label1.transform([make])
    x1=np.array([x1])
    x2=e_label2.transform([ft])
    x2=np.array([x2])
    x3=e_label3.transform([bs])
    x3=np.array([x3])
    x4=e_label4.transform([dw])
    x4=np.array([x4])
    x5=e_label5.transform([el])
    x5=np.array([x5])
    x6=e_label6.transform([et])
    x6=np.array([x6])
    xf=[x1[0],x2[0],x3[0],x4[0],x5[0],x6[0],sym,nl,w,h,es,hp,cm,hm]
    #change list x2 in numpy array 
    X=np.array([xf])
    features=scale.transform(X)
    features

    Y_pred=model.predict(features)
    #print("Predicted price = ",Y_pred)
    st.write("Predicted price = ",Y_pred)
    

