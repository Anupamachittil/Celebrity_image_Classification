import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm


img_folder = "celebrities_data"
messi_img = os.listdir(img_folder+'/lionel_messi')
sharapova_img = os.listdir(img_folder+'/maria_sharapova')
roger_img = os.listdir(img_folder+'/roger_federer')
serena_img = os.listdir(img_folder+'/serena_williams')
kohli_img = os.listdir(img_folder+'/virat_kohli')

data = []
label = []
img_size = (128,128)

for i , image_name in tqdm(enumerate(messi_img),desc="messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_folder+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        data.append(np.array(image))
        label.append(0)
for i , image_name in tqdm(enumerate(sharapova_img),desc="sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_folder+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        data.append(np.array(image))
        label.append(1)

for i , image_name in tqdm(enumerate(roger_img),desc="roger"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_folder+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        data.append(np.array(image))
        label.append(2)

for i , image_name in tqdm(enumerate(serena_img),desc="serena"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_folder+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        data.append(np.array(image))
        label.append(3)
for i , image_name in tqdm(enumerate(kohli_img),desc="kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_folder+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        data.append(np.array(image))
        label.append(4)

dataset=np.array(data)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalaising the Dataset. \n")

# x_train=x_train.astype('float')/255
# x_test=x_test.astype('float')/255 

# Same step above is implemented using tensorflow functions.

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

print("--------------------------------------\n")


model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(5,activation='softmax')
])
print("--------------------------------------\n")
model.summary()
print("--------------------------------------\n")

model.compile(optimizer='adam',
            loss='SparseCategoricalCrossentropy',
            metrics=['accuracy'])


print("--------------------------------------\n")
print("Training Started.\n")
history=model.fit(x_train,y_train,epochs=25,batch_size =128,validation_split=0.1)
print("Training Finished.\n")
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy=model.evaluate(x_test,y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")
y_pred=model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
print('classification Report\n',classification_report(y_test,y_pred_labels))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Model Prediction.\n")
def make_prediction(img, model):
    try:
        img_data = cv2.imread(img)
        if img_data is None:
            print("Error: Image not found or couldn't be read.")
            return
        
        img_pil = Image.fromarray(img_data)
        img_resized = img_pil.resize((128, 128))
        img_np = np.array(img_resized)
        input_img = np.expand_dims(img_np, axis=0)
        
        predictions = model.predict(input_img)
        class_labels = ["Messi","sharapova","roger","serena","Kohli"]
        
        predicted_class_index = np.argmax(predictions)
        
        predicted_class = [class_labels[predicted_class_index]]
        
        print("Predicted Celebrity:", predicted_class)
    
    except Exception as e:
        print("Error occurred during prediction:", e)      
make_prediction(r'celebrities_data\roger_federer\roger_federer1.png',model)
print("--------------------------------------\n")


