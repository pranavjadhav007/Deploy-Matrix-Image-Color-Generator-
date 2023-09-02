from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt
import numpy as np
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
  

def imsho(title="Image",img=None,size=10):
  w,h=img.shape[0],img.shape[1]
  aspect_ratio=w/h
  plt.figure(figsize=(size*aspect_ratio,size))
  plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
  plt.title(title)
  plt.show()   
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename1 = secure_filename("processed_" + file.filename)
        filename2= secure_filename("processed2_" + file.filename)
        filename3= secure_filename("processed3_" + file.filename)
        filename4= secure_filename("processed4_" + file.filename)
        filename5= secure_filename("processed5_" + file.filename)
        filename6= secure_filename("processed6_" + file.filename)
        filename7= secure_filename("processed7_" + file.filename)
        filename8= secure_filename("processed8_" + file.filename)
        filename9= secure_filename("processed9_" + file.filename)
        filename10= secure_filename("processed10_" + file.filename)
        filename11= secure_filename("processed11_" + file.filename)
        filename12= secure_filename("processed12_" + file.filename)


        # original
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = cv2.imread(file_path)
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        B,G,R=cv2.split(img)
        processed_img2=cv2.merge([B+100,G+30,R])
        
        processed_img3=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
        zeros=np.zeros(img.shape[:2],dtype="uint8")
        processed_img4=cv2.merge([zeros,G,zeros])
        
        processed_img5=cv2.merge([B+20,G+30,R+40])
        
        processed_img6=cv2.merge([B+10,G+30,R+160])
        
        processed_img7=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)

        processed_img8=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        processed_img9=cv2.merge([B+10,G+60,R+5])
        
        M=np.ones(img.shape,dtype='uint8')*100
        processed_img10=M+img
        
        processed_img11 = img-M 
        
        processed_img12=cv2.Canny(img,50,200)


        

        #processed
        processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        cv2.imwrite(processed_file_path, processed_img)
        
        processed_file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        cv2.imwrite(processed_file_path2, processed_img2)

        processed_file_path3 = os.path.join(app.config['UPLOAD_FOLDER'], filename3)
        cv2.imwrite(processed_file_path3, processed_img3)
        
        processed_file_path4 = os.path.join(app.config['UPLOAD_FOLDER'], filename4)
        cv2.imwrite(processed_file_path4, processed_img4)
        
        processed_file_path5 = os.path.join(app.config['UPLOAD_FOLDER'], filename5)
        cv2.imwrite(processed_file_path5, processed_img5)
        
        processed_file_path6 = os.path.join(app.config['UPLOAD_FOLDER'], filename6)
        cv2.imwrite(processed_file_path6, processed_img6)
        
        processed_file_path7 = os.path.join(app.config['UPLOAD_FOLDER'], filename7)
        cv2.imwrite(processed_file_path7, processed_img7)

        processed_file_path8 = os.path.join(app.config['UPLOAD_FOLDER'], filename8)
        cv2.imwrite(processed_file_path8, processed_img8)

        processed_file_path9 = os.path.join(app.config['UPLOAD_FOLDER'], filename9)
        cv2.imwrite(processed_file_path9, processed_img9)
        
        processed_file_path10 = os.path.join(app.config['UPLOAD_FOLDER'], filename10)
        cv2.imwrite(processed_file_path10, processed_img10)

        processed_file_path11 = os.path.join(app.config['UPLOAD_FOLDER'], filename11)
        cv2.imwrite(processed_file_path11, processed_img11)
        
        processed_file_path12 = os.path.join(app.config['UPLOAD_FOLDER'], filename12)
        cv2.imwrite(processed_file_path12, processed_img12)

        flash('T4 GPU')
        return render_template('index.html', filename=filename, filename1=filename1, filename2=filename2,filename3=filename3,filename4=filename4,filename5=filename5,filename6=filename6,filename7=filename7,filename8=filename8,filename9=filename9,filename10=filename10,filename11=filename11,filename12=filename12)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

def imsho(title="Image", img=None, size=10):
    w, h = img.shape[0], img.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(img, cmap='gray')  
    plt.title(title)
    plt.axis('off')  
    plt.show()
 
 
@app.route('/display/<filename>')
def display_image(filename):
    loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(loc)

    imsho("Processed Image", img)  

    return redirect(url_for('static', filename='uploads/' + filename), code=301) 
if __name__ == "__main__":
    app.run()