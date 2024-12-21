from flask import Flask, render_template, request, redirect, url_for, jsonify
import subprocess, os, requests
from web_scraping import *
from rename import *
from yolov8_script import *
from decision_tree import *
from random_forest import *
import sys
import shutil
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from yolov8_script import *
from firebase_initialize import *
from sort_with_yolov8 import *
from hurdle_cords import *
from a_star import *
from path_cords import *
from eff_net import *
initialize()
from station_data_py import *
from social_forces import *
from abm import *
from sfm_a import *
from sfm_a_abm import *
from cnn_bh import *
from rooms_split import *
from safe_path import *
from fire_parameters import *
from pathfinder_pywin import *
from prioritized_path import *
import json 
from PIL import Image
import io

paths = [r'..\yolov5\runs\detect', r'..\yolov5\runs\val', r'..\yolov5\runs\train', '/uploads', './runs/detect']
for i in paths:
  if os.path.exists(i):
    if os.path.isfile(i):
      os.remove(i)
    elif os.path.isdir(i):
      shutil.rmtree(i)

for i in os.listdir('../static'):
   if i not in ['show', 'model.glb']:
      if os.path.exists(f'../static{i}'):
        os.remove(f'../static{i}')

app = Flask(__name__, template_folder='../templates', static_url_path='', static_folder='../static',)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("user_ui/index.html")

@app.route('/user_inp', methods=['GET', 'POST'])
def user_inp():
    return render_template("user_ui/user_input.html")

@app.route('/home1', methods=['GET', 'POST'])
def home1():
    return render_template("user_ui/home1.html")

@app.route('/home2', methods=['GET', 'POST'])
def home2():
    return render_template("user_ui/home2.html")

@app.route('/home3', methods=['GET', 'POST'])
def home3():
    return render_template("user_ui/home3.html")

@app.route('/lit_sur', methods=['GET', 'POST'])
def lit_sur():
    return render_template("user_ui/lit_sur.html")

@app.route('/glb', methods=['GET', 'POST'])
def glb():
    return render_template("user_ui/glb_viewer.html")

@app.route('/arch', methods=['GET', 'POST'])
def arch():
    return render_template("user_ui/arch.html")

@app.route('/web_scraping', methods=['GET', 'POST'])
def web_scraping():
    return render_template("web_scraping.html", loader=True)

@app.route('/adult_children_elderly', methods=['GET', 'POST'])
def adult_children_elderly():
    return render_template("adult_vs_children.html", loader=True)

@app.route('/hbyolo', methods=['GET', 'POST'])
def hbyolo():
    return render_template("human_behaviour yolo.html", loader=True)

@app.route('/hbyolov8', methods=['GET', 'POST'])
def hbyolov8():
    return render_template("human_behaviour yolov8.html", loader=True)

@app.route('/hsyolov8', methods=['GET', 'POST'])
def hsyolov8():
    return render_template("hazardous_yolo.html", loader=True)

@app.route('/sfm', methods=['GET', 'POST'])
def sfm():
    return render_template("user_ui/social_forces_model.html")

@app.route('/hurdle_cord', methods=['GET', 'POST'])
def hurdle_cord():
    return render_template("hurdle_cord.html")

@app.route('/dist_sort1', methods=['GET', 'POST'])
def dist_sort():
    return render_template("user_ui/dist_sort.html")

@app.route('/sfm_predict', methods=['GET', 'POST'])
def sfm_predict():
    social_forces_predict()
    imgs = [i for i in os.listdir(f'../static/') if (i.startswith('sfm'))]
    return render_template("user_ui/social_forces_model.html", sample_imgs=True, imgs=imgs)

@app.route('/a_star', methods=['GET', 'POST'])
def a_star():
    return render_template("a_star.html")

@app.route('/sfm_a', methods=['GET', 'POST'])
def sfm_a():
    return render_template("user_ui/sfm_a.html")

@app.route('/sfm_a_abm', methods=['GET', 'POST'])
def sfm_a_abm():
    return render_template("user_ui/sfm_a_abm.html")

@app.route('/path_cords', methods=['GET', 'POST'])
def path_cords():
    return render_template("user_ui/path_cords.html", loader=True)

@app.route('/safe_path_cords', methods=['GET', 'POST'])
def safe_path_cords():
    return render_template("user_ui/safe_path.html")

@app.route('/pyrosim', methods=['GET', 'POST'])
def pyrosim():
    return render_template("pyrosim.html")

@app.route('/picfrom3d', methods=['GET', 'POST'])
def picfrom3d():
    return render_template("pic_from_3D.html")

@app.route('/room_split', methods=['GET', 'POST'])
def room_split():
    return render_template("user_ui/room_split.html")

@app.route('/process-coordinates', methods=['POST'])
def process_coordinates():
    data = request.get_json()
    x, y, z = data.get('x'), data.get('y'), data.get('z')
    print(f"Received coordinates: x={x}, y={y}, z={z}")

    # Process the coordinates (Example: Store them or perform calculations)
    result = {"message": "Coordinates processed successfully", "received": data}

    return jsonify(result)

@app.route('/lpg_scrap', methods=['GET', 'POST'])
def lpg_scrap():
    #result = web_scrap(lpg, '/content/lpg_dataset')
    result = web_scrap(['https://www.freepik.com/free-photos-vectors/lpg-gas'],'./lpg_dataset')
    print(result)
    imgs = os.listdir('./lpg_dataset')[:4]
    print(imgs)
    for i in imgs:
      print(os.getcwd())
      shutil.copy(f'./lpg_dataset/{i}', "../static/")
    #rename_files_in_folder(folder_path = r'/content/static/', file_extension='.jpg')
    return render_template("web_scraping.html", urls=lpg, result=result, sample_imgs = True, imgs=imgs, loader=True)

@app.route('/flammable_scrap', methods=['GET', 'POST'])
def flammable_scrap():
    #result = web_scrap(lpg, '/content/lpg_dataset')
    result = web_scrap(['https://www.complianceandrisks.com/topics/globally-harmonized-system/'],'./flammable')
    imgs = os.listdir('./flammable')
    for i in imgs:
      shutil.copy(f'./flammable/{i}', "../static/")
    #rename_files_in_folder(folder_path = r'/content/static/', file_extension='.jpg')
    return render_template("web_scraping.html", urls=lpg, result=result, sample_imgs = True, imgs=imgs, loader=True)

@app.route('/child_scrap', methods=['GET', 'POST'])
def child_scrap():
    #result = web_scrap(child_urls, '/content/child')
    result = web_scrap(['https://www.istockphoto.com/search/2/image-film?phrase=children&page=2'],'./child')
    imgs = os.listdir('./child')[:4]
    for i in imgs:
      shutil.copy(f'./child/{i}', "../static/")
    #rename_files_in_folder(folder_path = r'/content/static/', file_extension='.jpg')
    return render_template("web_scraping.html", urls=child_urls, result=result, sample_imgs = True, imgs=imgs, loader=True)

@app.route('/elderly_scrap', methods=['GET', 'POST'])
def elderly_scrap():
    #result = web_scrap(elderly_urls, '/content/elderly')
    result = web_scrap(['https://www.gettyimages.in/search/2/image-film?phrase=indian%20elderly&sort=mostpopular&page=2'],'./elderly')
    imgs = os.listdir('./elderly')[:4]
    for i in imgs:
      shutil.copy(f'./elderly/{i}', "../static/")
    #rename_files_in_folder(folder_path = r'/content/static/', file_extension='.jpg')
    return render_template("web_scraping.html", urls=elderly_urls, result=result, sample_imgs = True, imgs=imgs, loader=True)

@app.route('/hytrain', methods=['POST'])
def hytrain():
    res=''
    imgs=[]
    if 'runs' in os.listdir("../yolov5"):
      if 'train' in os.listdir("../yolov5/runs/"):
        prev_exp = os.listdir("../yolov5/runs/train")
      else:
        prev_exp=[]
    else:
      prev_exp=[]
    epochs = request.form['epochs-input']
    print(epochs)
    train_args = [
    "python", '../yolov5/train.py', '--img', '640', '--batch', '2', '--epochs', str(epochs),
    '--data', '../human_data.yaml', '--weights', 'yolov5s.pt', '--cache'
    ]
    print("Training started")
    try:
      result = subprocess.run(train_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
      print(result.stdout.decode('utf-8'))
      res = result.stdout.decode('utf-8')
      print(res)
      pres_exp = os.listdir("../yolov5/runs/train")
      exp = list(set(pres_exp)-set(prev_exp))[0]
      print(exp)
      res+=f'<br/><br/>Result stored in the folder : ../yolov5/runs/train/{exp}'
      print('Testing ended successfully')
      imgs = os.listdir(f'../yolov5/runs/train/{exp}')
      for i in imgs:
        shutil.copy(f'../yolov5/runs/train/{exp}/{i}', "../static/")
      print('Training ended successfully')
    except subprocess.CalledProcessError as e:
      print(f"Error during training: {e.stderr.decode('utf-8')}")
    print(imgs)
    return render_template("human_behaviour yolo.html", loader=True, result =res.replace('\n', '<br/>'), sample_imgs = True, imgs=imgs)

@app.route('/hytest', methods=['POST'])
def hytest():
    res=''
    imgs=[]
    if 'runs' in os.listdir("../yolov5"):
      if 'val' in os.listdir("../yolov5/runs/"):
        prev_exp = os.listdir("../yolov5/runs/val")
      else:
        prev_exp=[]
    else:
      prev_exp=[]
    train_args = [
    "python", "../yolov5/val.py", '--weights', '../models/yolov5_30.pt', '--data',
    '../human_data.yaml', '--task', 'val'
    ]
    print("Testing started")
    try:
      result = subprocess.run(train_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
      res = result.stdout.decode('utf-8')
      print(res)
      pres_exp = os.listdir("../yolov5/runs/val")
      exp = list(set(pres_exp)-set(prev_exp))[0]
      print(exp)
      res+=f'<br/><br/>Result stored in the folder : ../yolov5/runs/val/{exp}'
      print('Testing ended successfully')
      imgs = os.listdir(f'../yolov5/runs/val/{exp}')
      for i in imgs:
        shutil.copy(f'../yolov5/runs/val/{exp}/{i}', "../static/")
    except subprocess.CalledProcessError as e:
      print(f"Error during training: {e.stderr.decode('utf-8')}")
    return render_template("human_behaviour yolo.html", loader=True, result =res.replace('\n', '<br/>'), sample_imgs = True, imgs=imgs)

@app.route('/hydata', methods=['POST'])
def hydata():
  train = os.listdir("../datasets/human_dataset/images/train")
  val = os.listdir("../datasets/human_dataset/images/val")
  timgs = train[:4]
  for i in timgs:
      shutil.copy(f'../datasets/human_dataset/images/train/{i}', "../static/")
  vimgs = val[-4:]
  for i in vimgs:
      shutil.copy(f'../datasets/human_dataset/images/val/{i}', "../static/")
  imgs = timgs + vimgs
  result = f'''Train images {len(train)}<br/><br>
  Val images : {len(val)}<br/><br>
  Dataset source : Kaggle<br/>
  <a href="https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset">Human Detection Dataset</a><br>
  <a href="https://www.kaggle.com/datasets/jonathannield/cctv-human-pose-estimation-dataset?resource=download">CCTV Human Pose Estimation Dataset</a>'''
  return render_template("human_behaviour yolo.html", loader=True, result = result, sample_imgs = True, imgs = imgs)

@app.route('/hypredict', methods=['GET','POST'])
def hypredict():
    result=''
    if 'runs' in os.listdir("../yolov5"):
      if 'detect' in os.listdir("../yolov5/runs/"):
        prev_exp = os.listdir("../yolov5/runs/detect")
      else:
        prev_exp=[]
    else:
      prev_exp=[]
    tf = request.files['test-file']
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(uploads_dir):
      os.makedirs(uploads_dir)
    file_path = os.path.join(uploads_dir, tf.filename)
    tf.save(file_path)
    print(tf)
    detect_args = ["python", "../yolov5/detect.py", "--weights", "../models/yolov5_30.pt", "--img", "640", "--conf", "0.25", "--source", file_path, "--save-crop"]
    print("Prediction started")
    try:
      result = subprocess.run(detect_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, shell=True)
      print(result.stdout.decode('utf-8'))
      res = result.stdout.decode('utf-8')
      print('Prediction ended successfully')
      pres_exp = os.listdir("../yolov5/runs/detect")
      exp = list(set(pres_exp)-set(prev_exp))[0]
      print(exp)
      res+=f'<br/><br/>Result stored in the folder : ../yolov5/runs/detect/{exp}'
      imgs = [i for i in os.listdir(f'../yolov5/runs/detect/{exp}') if i!='crops']
      print(imgs)
      for i in imgs:
          print(i)
          shutil.copy(f'../yolov5/runs/detect/{exp}/{i}', f"../static/")
    except subprocess.CalledProcessError as e:
      print(f"Error during Prediction: {e.stderr.decode('utf-8')}")
    return render_template("human_behaviour yolo.html", loader=True, result =res.replace('\n', '<br/>'), sample_imgs = True, imgs=imgs)

@app.route('/hyresults', methods=['GET', 'POST'])
def hyresults():
   imgs1 = [f'/yolo_result/epochs_10/'+i for i in os.listdir(f'../static/yolo_result/epochs_10') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs2 = [f'/yolo_result/epochs_20/'+i for i in os.listdir(f'../static/yolo_result/epochs_20') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs3 = [f'/yolo_result/epochs_30/'+i for i in os.listdir(f'../static/yolo_result/epochs_30') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs4 = [f'/yolo_result/epochs_40/'+i for i in os.listdir(f'../static/yolo_result/epochs_40') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs5 = [f'/yolo_result/epochs_50/'+i for i in os.listdir(f'../static/yolo_result/epochs_50') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs6 = [f'/yolo_result/epochs_60/'+i for i in os.listdir(f'../static/yolo_result/epochs_60') if (i.endswith('.jpg') or i.endswith('.png'))]
   return render_template("human_behaviour yolo.html", loader=True, res_table =True, res_imgs_bool=True, res_imgs1=imgs1, res_imgs2=imgs2, res_imgs3=imgs3, res_imgs4=imgs4, res_imgs5=imgs5, res_imgs6=imgs6)

@app.route('/hytrainv8', methods=['POST'])
def hytrainv8():
    res=''
    imgs=[]
    if 'runs' in os.listdir("./"):
      if 'detect' in os.listdir("./runs/"):
        prev_exp = os.listdir("./runs/detect")
      else:
        prev_exp=[]
    else:
      prev_exp=[]
    epochs = request.form['epochs-input']
    print(epochs)
    res=yolov8_train(epochs)
    print("Training started")
    pres_exp = os.listdir("./runs/detect")
    exp = list(set(pres_exp)-set(prev_exp))[0]
    print(exp)
    res=res+f'<br/><br/>Result stored in the folder : ./runs/detect/{exp}'
    print('Testing ended successfully')
    imgs = os.listdir(f'./runs/detect/{exp}')
    for i in imgs:
      shutil.copy(f'./runs/detect/{exp}/{i}', "../static/")
    print('Training ended successfully')
    return render_template("human_behaviour yolov8.html", loader=True, result =str(res).replace('\n', '<br>'), sample_imgs = True, imgs=imgs)

@app.route('/hytestv8', methods=['POST'])
def hytestv8():
    res=''
    imgs=[]
    if 'runs' in os.listdir("./"):
      if 'detect' in os.listdir("./runs/"):
        prev_exp = os.listdir("./runs/detect")
      else:
        prev_exp=[]
    else:
      prev_exp=[]
    res=yolov8_val('../models/yolov8_30.pt')
    print("Testing started")
    pres_exp = os.listdir("./runs/detect")
    exp = list(set(pres_exp)-set(prev_exp))[0]
    print(exp)
    res=str(res)+f'<br/><br/>Result stored in the folder : ./runs/detect/{exp}'
    print('Testing ended successfully')
    imgs = os.listdir(f'./runs/detect/{exp}')
    for i in imgs:
      shutil.copy(f'./runs/detect/{exp}/{i}', "../static/")
    print('Testing ended successfully')
    return render_template("human_behaviour yolov8.html", loader=True, result =str(res).replace('\n', '<br>'), sample_imgs = True, imgs=imgs)

@app.route('/hypredictv8', methods=['GET','POST'])
def hypredictv8():
    res=''
    imgs=[]
    if 'runs' in os.listdir("./"):
      if 'detect' in os.listdir("./runs/"):
        prev_exp = os.listdir("./runs/detect")
      else:
        prev_exp=[]
    else:
      prev_exp=[]
    tf = request.files['test-file']
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(uploads_dir):
      os.makedirs(uploads_dir)
    file_path = os.path.join(uploads_dir, tf.filename)
    tf.save(file_path)
    print(tf)
    print("Prediction started")
    res = pre_yolov8_detect('../models/yolov8_30.pt', file_path)
    print('Prediction ended successfully')
    pres_exp = os.listdir("./runs/detect")
    exp = list(set(pres_exp)-set(prev_exp))[0]
    print(exp)
    res=str(res)+f'<br/><br/>Result stored in the folder : ./runs/detect/{exp}'
    imgs = [i for i in os.listdir(f'./runs/detect/{exp}') if i!='crops']
    print(imgs)
    for i in imgs:
        print(i)
        shutil.copy(f'./runs/detect/{exp}/{i}', f"../static/")
    print(str(res))
    return render_template("human_behaviour yolov8.html", loader=True, result =str(res).replace('\n', '<br>'), sample_imgs = True, imgs=imgs)

@app.route('/pre_hypredictv8', methods=['GET','POST'])
def pre_hypredictv8():
    res=''
    imgs=[]
    if 'runs' in os.listdir("./"):
      if 'detect' in os.listdir("./runs/"):
        prev_exp = os.listdir("./runs/detect")
      else:
        prev_exp=[]
    else:
      prev_exp=[]
    tf = request.files['test-file']
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(uploads_dir):
      os.makedirs(uploads_dir)
    file_path = os.path.join(uploads_dir, tf.filename)
    tf.save(file_path)
    print(tf)
    print("Prediction started")
    res = yolov8_detect('yolov8n.pt', file_path)
    print('Prediction ended successfully')
    pres_exp = os.listdir("./runs/detect")
    exp = list(set(pres_exp)-set(prev_exp))[0]
    print(exp)
    res=str(res)+f'<br/><br/>Result stored in the folder : ./runs/detect/{exp}'
    imgs = [i for i in os.listdir(f'./runs/detect/{exp}') if i!='crops']
    print(imgs)
    for i in imgs:
        print(i)
        shutil.copy(f'./runs/detect/{exp}/{i}', f"../static/")
    print(str(res))
    return render_template("human_behaviour yolov8.html", loader=True, result =str(res).replace('\n', '<br>'), sample_imgs = True, imgs=imgs)

@app.route('/hyresultsv8', methods=['GET', 'POST'])
def hyresultsv8():
   imgs1 = [f'/yolo_result/10 epochs/'+i for i in os.listdir(f'../static/yolo_result/10 epochs') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs2 = [f'/yolo_result/20 epochs/'+i for i in os.listdir(f'../static/yolo_result/20 epochs') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs3 = [f'/yolo_result/30 epochs/'+i for i in os.listdir(f'../static/yolo_result/30 epochs') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs4 = [f'/yolo_result/40 epochs/'+i for i in os.listdir(f'../static/yolo_result/40 epochs') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs5 = [f'/yolo_result/50 epochs/'+i for i in os.listdir(f'../static/yolo_result/50 epochs') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs6 = [f'/yolo_result/60 epochs/'+i for i in os.listdir(f'../static/yolo_result/60 epochs') if (i.endswith('.jpg') or i.endswith('.png'))]
   return render_template("human_behaviour yolov8.html", loader=True, res_table =True, res_imgs_bool=True, res_imgs1=imgs1, res_imgs2=imgs2, res_imgs3=imgs3, res_imgs4=imgs4, res_imgs5=imgs5, res_imgs6=imgs6)

@app.route('/hsdata', methods=['POST'])
def hsdata():
  train = os.listdir("../datasets/hazardous_dataset/images/train")
  val = os.listdir("../datasets/hazardous_dataset/images/val")
  timgs = train[:4]
  for i in timgs:
      shutil.copy(f'../datasets/hazardous_dataset/images/train/{i}', "../static/")
  vimgs = val[-4:]
  for i in vimgs:
      shutil.copy(f'../datasets/hazardous_dataset/images/val/{i}', "../static/")
  imgs = timgs + vimgs
  result = f'''Train images : 417<br/><br>
  Val images : {len(val)}<br/><br>
  Dataset source : Github and Image Scraping<br/>
  <a href="https://github.com/mrl-amrl/HAZMAT13/tree/main">Flammable Materials Signs Dataset</a>'''
  return render_template("hazardous_yolo.html", loader=True, result = result, sample_imgs = True, imgs = imgs)

@app.route('/hstrainv8', methods=['POST'])
def hstrainv8():
    res=''
    imgs=[]
    if 'runs' in os.listdir("./"):
      if 'detect' in os.listdir("./runs/"):
        prev_exp = os.listdir("./runs/detect")
      else:
        prev_exp=[]
    else:
      prev_exp=[]
    epochs = request.form['epochs-input']
    print(epochs)
    res=hsyolov8_train(epochs)
    print("Training started")
    pres_exp = os.listdir("./runs/detect")
    exp = list(set(pres_exp)-set(prev_exp))[0]
    print(exp)
    res=str(res)+f'<br/><br/>Result stored in the folder : ./runs/detect/{exp}'
    print('Testing ended successfully')
    imgs = [i for i in os.listdir(f'./runs/detect/{exp}') if (i.endswith('.jpg') or i.endswith('.png'))]
    for i in imgs:
      shutil.copy(f'./runs/detect/{exp}/{i}', "../static/")
    print('Training ended successfully')
    return render_template("hazardous_yolo.html", loader=True, result =str(res).replace('\n', '<br>'), sample_imgs = True, imgs=imgs)

@app.route('/hstestv8', methods=['POST'])
def hstestv8():
    res=''
    imgs=[]
    if 'runs' in os.listdir("./"):
      if 'detect' in os.listdir("./runs/"):
        prev_exp = os.listdir("./runs/detect")
      else:
        prev_exp=[]
    else:
      prev_exp=[]
    res=hsyolov8_val('../models/hsyolov8_30.pt')
    print("Testing started")
    pres_exp = os.listdir("./runs/detect")
    exp = list(set(pres_exp)-set(prev_exp))[0]
    print(exp)
    res=str(res)+f'<br/><br/>Result stored in the folder : ./runs/detect/{exp}'
    print('Testing ended successfully')
    imgs = os.listdir(f'./runs/detect/{exp}')
    for i in imgs:
      shutil.copy(f'./runs/detect/{exp}/{i}', "../static/")
    print('Testing ended successfully')
    return render_template("hazardous_yolo.html", loader=True, result =str(res).replace('\n', '<br>'), sample_imgs = True, imgs=imgs)

@app.route('/hspredictv8', methods=['GET','POST'])
def hspredictv8():
    res=''
    imgs=[]
    if 'runs' in os.listdir("./"):
      if 'detect' in os.listdir("./runs/"):
        prev_exp = os.listdir("./runs/detect")
      else:
        prev_exp=[]
    else:
      prev_exp=[]
    tf = request.files['test-file']
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(uploads_dir):
      os.makedirs(uploads_dir)
    file_path = os.path.join(uploads_dir, tf.filename)
    tf.save(file_path)
    print(tf)
    print("Prediction started")
    res = hsyolov8_detect('../models/hsyolov8_30.pt', file_path)
    print('Prediction ended successfully')
    pres_exp = os.listdir("./runs/detect")
    exp = list(set(pres_exp)-set(prev_exp))[0]
    print(exp)
    res=str(res)+f'<br/><br/>Result stored in the folder : ./runs/detect/{exp}'
    imgs = [i for i in os.listdir(f'./runs/detect/{exp}') if i!='crops']
    print(imgs)
    for i in imgs:
        print(i)
        shutil.copy(f'./runs/detect/{exp}/{i}', f"../static/")
    print(str(res))
    return render_template("hazardous_yolo.html", loader=True, result =str(res).replace('\n', '<br>'), sample_imgs = True, imgs=imgs)

@app.route('/hsresultsv8', methods=['GET', 'POST'])
def hsresultsv8():
   imgs1 = [f'/yolo_result/epochs 10/'+i for i in os.listdir(f'../static/yolo_result/epochs 10') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs2 = [f'/yolo_result/epochs 20/'+i for i in os.listdir(f'../static/yolo_result/epochs 20') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs3 = [f'/yolo_result/epochs 30/'+i for i in os.listdir(f'../static/yolo_result/epochs 30') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs4 = [f'/yolo_result/epochs 40/'+i for i in os.listdir(f'../static/yolo_result/epochs 40') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs5 = [f'/yolo_result/epochs 50/'+i for i in os.listdir(f'../static/yolo_result/epochs 50') if (i.endswith('.jpg') or i.endswith('.png'))]
   imgs6 = [f'/yolo_result/epochs 60/'+i for i in os.listdir(f'../static/yolo_result/epochs 60') if (i.endswith('.jpg') or i.endswith('.png'))]
   return render_template("hazardous_yolo.html", loader=True, res_table =True, res_imgs_bool=True, res_imgs1=imgs1, res_imgs2=imgs2, res_imgs3=imgs3, res_imgs4=imgs4, res_imgs5=imgs5, res_imgs6=imgs6)

@app.route('/acedata', methods=['GET', 'POST'])
def acedata():
    adult = os.listdir(r"..\datasets\adult_children_elderly\train\adults")
    children = os.listdir(r"..\datasets\adult_children_elderly\train\children")
    elderly = os.listdir(r"..\datasets\adult_children_elderly\train\elderly")
    aimgs = adult[:2]
    for i in aimgs:
      shutil.copy(f"../datasets/adult_children_elderly/train/adults/{i}", "../static/")
    cimgs = children[-2:]
    for i in cimgs:
      shutil.copy(f"../datasets/adult_children_elderly/train\children/{i}", "../static/")
    eimgs = elderly[20:40]
    for i in cimgs:
      shutil.copy(f"../datasets/adult_children_elderly/train\children/{i}", "../static/")
    imgs = aimgs + cimgs + eimgs
    result = f'''
    No. of Classes : 3 (Adult, Children, Elderly)<br><br>
    Train images:<br>
    Adult images :  {len(adult)}<br/>
    Children images : {len(children)}<br/>
    Elderly images : {len(elderly)}<br/><br>
    Dataset source : Kaggle and Image Scraping<br/>
    <a href="https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset">Human Detection Dataset</a><br>
    <a href="https://www.kaggle.com/datasets/jonathannield/cctv-human-pose-estimation-dataset?resource=download">CCTV Human Pose Estimation Dataset</a>'''
    return render_template("adult_vs_children.html", result=result, loader=True)

@app.route('/dttrain', methods=['GET', 'POST'])
def dttrain():
    result = dt_train('./dt.pkl')
    return render_template("adult_vs_children.html", result=result.replace('\n', '<br>').replace(' ', '&nbsp;'), loader=True)

@app.route('/dttest', methods=['GET', 'POST'])
def dttest():
    results = []
    res=''
    ov_res=0
    paths={'Adult':'../datasets/adult_children_elderly/test/adults', 'Children':'../datasets/adult_children_elderly/test/children', 'Elderly':'../datasets/adult_children_elderly/test/elderly', }
    for i in paths:
      result1=0
      ov_res += len(os.listdir(paths[i]))
      for filename in os.listdir(paths[i]):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(paths[i], filename)
            result = dec_dt(image_path)
            if result==i:
              results.append(result)
              result1+=1
      res+=f"Class : {i} Correct Predictions : {result1}<br>"
    res+=f'<br>Overall accuracy: {len(results)/ov_res}'
    return render_template("adult_vs_children.html", result=res, loader=True)

@app.route('/dtpredict', methods=['GET', 'POST'])
def dtpredict():
    tf = request.files['test-file1']
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(uploads_dir):
      os.makedirs(uploads_dir)
    file_path = os.path.join(uploads_dir, tf.filename)
    tf.save(file_path)
    print(tf)
    result = dec_dt(file_path)
    return render_template("adult_vs_children.html", result=f"The predicted class for the image is: {result}", loader=True)

@app.route('/dtresults', methods=['GET', 'POST'])
def dtresults():
   result="""
"""
   return render_template("adult_vs_children.html", loader=True, result=result.replace('\n', '<br>'), table_res=True)

@app.route('/effresults', methods=['GET', 'POST'])
def effresults():
   result="""
"""
   return render_template("adult_vs_children.html", loader=True, result=result.replace('\n', '<br>'), table_res_eff=True)

@app.route('/rftrain', methods=['GET', 'POST'])
def rftrain():
    result = rf_train('./rf.pkl')
    return render_template("adult_vs_children.html", result=result.replace('\n', '<br>').replace(' ', '&nbsp;'), loader=True)

@app.route('/rftest', methods=['GET', 'POST'])
def rftest():
    results = []
    res=''
    ov_res=0
    paths={'Adult':'../datasets/adult_children_elderly/test/adults', 'Children':'../datasets/adult_children_elderly/test/children', 'Elderly':'../datasets/adult_children_elderly/test/elderly', }
    for i in paths:
      result1=0
      ov_res += len(os.listdir(paths[i]))
      for filename in os.listdir(paths[i]):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(paths[i], filename)
            result = predict_image_class(image_path)
            if result==i:
              results.append(result)
              result1+=1
      res+=f"Class : {i} Correct Predictions : {result1}<br>"
    res+=f'<br>Overall accuracy: {len(results)/ov_res}'
    return render_template("adult_vs_children.html", result=res, loader=True)

@app.route('/rfpredict', methods=['GET', 'POST'])
def rfpredict():
    tf = request.files['test-file2']
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(uploads_dir):
      os.makedirs(uploads_dir)
    file_path = os.path.join(uploads_dir, tf.filename)
    tf.save(file_path)
    print(tf)
    result = predict_image_class(file_path)
    return render_template("adult_vs_children.html", result=f"The predicted class for the image is: {result}", loader=True)

@app.route('/rfresults', methods=['GET', 'POST'])
def rfresults():
   result="""
"""
   return render_template("adult_vs_children.html", loader=True, result=result.replace('\n', '<br>'), table_res_rf=True)

@app.route('/cnntrain', methods=['GET', 'POST'])
def cnntrain():
    epochs = request.form['epochs-input']
    print(epochs)
    model_filename = '/content/models/cnn_model_epoch10.keras'
    cnn_train(model_filename, epochs)
    result = "model saved successfully"
    return render_template("adult_vs_children.html", result=result.replace('\n', '<br>').replace(' ', '&nbsp;'), loader=True)

@app.route('/cnntest', methods=['GET', 'POST'])
def cnntest():
    result=cnn_evaluate_on_test('../models/cnn_model_epoch20.h5')
    return render_template("adult_vs_children.html", result=result.replace('\n', '<br>').replace(' ', '&nbsp;'), sample_imgs=True, imgs=['cnn_conf.png', 'cnn_roc.png'], loader=True)

@app.route('/cnnpredict', methods=['GET', 'POST'])
def cnnpredict():
    tf = request.files['test-file3']
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(uploads_dir):
      os.makedirs(uploads_dir)
    file_path = os.path.join(uploads_dir, tf.filename)
    tf.save(file_path)
    print(file_path)
    result=predict_image_class_cnn(file_path, '../models/cnn_model_epoch20.h5')
    return render_template("adult_vs_children.html", result=f'Predicted class : {result}'.replace('\n', '<br>').replace(' ', '&nbsp;'), loader=True)

@app.route('/cnnresults', methods=['GET', 'POST'])
def cnnresults():
    imgs1 = ['/cnn_result/'+i for i in os.listdir(f'../static/cnn_result/') if (i.endswith('.jpg') or i.endswith('.png'))]
    return render_template("adult_vs_children.html", result=result.replace('\n', '<br>').replace(' ', '&nbsp;'), sample_imgs=True, imgs=imgs1, table_res_cnn=True, loader=True)

@app.route('/effpredict', methods=['GET', 'POST'])
def effpredict():
    tf = request.files['test-file6']
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(uploads_dir):
      os.makedirs(uploads_dir)
    file_path = os.path.join(uploads_dir, tf.filename)
    tf.save(file_path)
    print(tf)
    result = eff_predict(file_path)
    return render_template("adult_vs_children.html", result=f"{result}", loader=True)

@app.route('/sortpredict', methods=['GET', 'POST'])
def sortpredict():
    tf = request.files['test-file']
    file_path = "../samples/1.mp4"
    tf.save(file_path)
    result=sort_algo(file_path)
    return render_template("user_ui/dist_sort.html", result=result.replace('\n', '<br>').replace(' ', '&nbsp;'),loader=True)

@app.route('/sortresults', methods=['GET', 'POST'])
def sortresults():
    result=''''''
# Speed ratios for different pedestrian types
    ref = db.reference('/')

    data = ref.get()
# Speed ratios for different types of pedestrians
    speed = {
    'child': data.get('children'),
    'adult': data.get('adult'),
    'elderly': data.get('elderly')
    }

# Calculate speed ratios relative to 'adult'
    adult_speed = speed['adult']

    speed_ratios = {
    'child': speed['child'] / adult_speed,
    'elderly': speed['elderly'] / adult_speed,
    'adult': 1.0
    }
    imgs1 = ['show/cloud.png', 'show/demo.png']
    result+='Speeds (in pixles/sec) : \n'
    for i in speed:
       result+=f'{i} : {speed[i]}\n'
    result+='\nSpeed Ratios : \n'
    for i in speed_ratios:
       result+=f'{i} : {speed_ratios[i]}\n'
    return render_template("user_ui/dist_sort.html", result=result.replace('\n', '<br>').replace(' ', '&nbsp;'), sample_imgs=True, imgs=imgs1, table_res_cnn=True, loader=True)

@app.route('/cord_convert', methods=['GET', 'POST'])
def cord_convert():
    tf = request.files['test-file']
    file_path = "../samples/1.png"
    tf.save(file_path)
    result = hurd_convert(file_path)
    imgs1 = ['show/binaryimage.png','show/plotimage.png']
    return render_template("hurdle_cord.html", result=result.replace('\n', '<br>').replace(' ', '&nbsp;'), sample_imgs=True, imgs=imgs1, table_res_cnn=True, loader=True)

@app.route('/hurdleresults', methods=['GET', 'POST'])
def hurdleresults():
    binary_grid = np.load('../static/show/binary_grid.npy')
    result =''''''
    for i in binary_grid:
        for j in i:
            result+=str(j)
            result+=' '
        result+='\n'
    imgs1 = ['show/binaryimage.png','show/plotimage.png']
    return render_template("hurdle_cord.html", sample_imgs=True, result=result.replace('\n', '<br>').replace(' ', '&nbsp;'), imgs=imgs1, table_res_cnn=True, loader=True)

@app.route('/pathresults', methods=['GET', 'POST'])
def pathresults():
    imgs1 = ['show/shortest_path.jpeg','show/best_routes.jpeg']
    return render_template("user_ui/path_cords.html", sample_imgs=True, imgs=imgs1, table_res_cnn=True, loader=True)

@app.route('/fire_pathresults', methods=['GET', 'POST'])
def fire_pathresults():
    imgs1 = ['show/shortest_path_fire.png']
    return render_template("user_ui/path_for_fire.html", sample_imgs=True, imgs=imgs1, table_res_cnn=True, loader=True)

@app.route('/safe_pathresults', methods=['GET', 'POST'])
def safe_pathresults():
    imgs1 = ['show/safe_path.png', 'show/safe_best.png']
    return render_template("user_ui/safe_path.html", sample_imgs=True, imgs=imgs1, table_res_cnn=True, loader=True)

@app.route('/a_star_predict', methods=['GET', 'POST'])
def a_star_predict():
    a_star_final()
    return render_template("a_star.html", loader=True)

@app.route('/sfm_a_predict', methods=['GET', 'POST'])
def sfm_a_predict():
    result=sfm_a_final()
    return render_template("sfm_a.html", loader=True, result=result.replace('\n', '<br>').replace(' ', '&nbsp;'),)

@app.route('/sfm_a_abm_predict', methods=['GET', 'POST'])
def sfm_a_abm_predict():
    result=abm_predict()
    return render_template("user_ui/sfm_a_abm.html", loader=True, result=result.replace('\n', '<br>').replace(' ', '&nbsp;'),)

@app.route('/evac_time', methods=['GET', 'POST'])
def evac_time():
    imgs=['/show/SFM.mp4']
    return render_template("user_ui/sfm.html", loader=True, sample_imgs=True, imgs=imgs)


@app.route('/safety_measures', methods=['GET', 'POST'])
def safety_measures():
    return render_template("user_ui/safety.html", loader=True)

# @app.route('/evac_time', methods=['GET', 'POST'])
# def evac_time():
#     access_thunderhead_pathfinder(r'../model\GF-1/GF-1.fds')
#     return render_template("user_ui/home1.html", loader=True)

@app.route('/fire_params', methods=['GET', 'POST'])
def fire_params():
    result=fire_params_test('../model/GF-1/GF-1_hrr.csv')
    imgs1=['show/fire_params.png']
    return render_template("user_ui/fire_parameters.html", loader=True, sample_imgs=True, imgs=imgs1)

@app.route('/path_for_fire', methods=['GET', 'POST'])
def path_for_fire():
    return render_template("user_ui/path_for_fire.html", loader=True)

@app.route('/rescue_contact', methods=['GET', 'POST'])
def rescue_contact():
    return render_template("user_ui/rescue_contact.html", loader=True)

@app.route('/rescue_contact_final', methods=['GET', 'POST'])
def rescue_contact_final():
    station_name = request.form['category'].upper()
    result=''
    if station_name.upper() in station_data.keys():
      result = f"Station Name : {station_name}\nTelephone: {station_data[station_name]['STD']}-{station_data[station_name]['phone_no']}\nMobile: {station_data[station_name]['mobile_no']}"
    else:
      result+= f'No Data Available for {station_name}'
    return render_template("user_ui/rescue_contact.html", loader=True, result=result.replace('\n', '<br>'))

@app.route('/path_predict', methods=['GET', 'POST'])
def path_predict():
        coordinates = request.form.get('coordinates')
        
        # Check if both coordinates and count are provided
        if not coordinates:
            return {"error": "Missing coordinates or count data"}, 400

        # Parse the coordinates if it's a JSON string
        coordinates = json.loads(coordinates)
        
        # Initialize the lists to store the coordinates based on person type
        adult_input = []
        child_input = []
        elderly_input = []
        width, height = 0,0

        # Handle file data (test-file)
        test_file = request.files['test-file']
        #test_file = request.files.get('test-file')

        if test_file:
            # Save the file in a specific location (uploads folder)
            file_path = './uploads/1.png'
            test_file.save(file_path)
            print(f"Received file: {test_file.filename}")
            with Image.open(file_path) as img:
                width, height = img.size  # Get width and height in pixels
                print(f"Image Dimensions: Width: {width} pixels, Height: {height} pixels")
        else:
            print("No file received")

        # Iterate through the coordinates and classify based on personType
        for item in coordinates:
            x = item.get('x')
            y = item.get('y')  # Not used
            z = item.get('z')
            person_type = item.get('personType')

            # Create a tuple (x, z) for coordinates
            x_min, x_max = 60, 110
            z_min, z_max = -35, -76
            
            scaled_x = scale_coordinate(x, x_min, x_max, 0, height)
            scaled_z = scale_coordinate(z, z_min, z_max, 0, width)

            # Create a tuple (scaled_x, scaled_z) for scaled coordinates
            coord = (round(scaled_z), round(scaled_x))
            print(coord)

            # Classify the coordinates based on the person type
            if person_type == 'Adult':
                adult_input.append(coord)
            elif person_type == 'Child':
                child_input.append(coord)
            elif person_type == 'Elderly':
                elderly_input.append(coord)

        # Print the parsed coordinates
        print(f"Received coordinates: {coordinates}")
        print({
            "adult_input": adult_input,
            "child_input": child_input,
            "elderly_input": elderly_input,
        })

        result=path_final(adult_input, child_input, elderly_input)
        
        print(result)
        imgs1=['show/best_routes.png', 'show/shortest_path.png']
        
        return redirect(url_for('pathresults'))
    
@app.route('/safe_path_predict', methods=['GET', 'POST'])
def safe_path_predict():
        coordinates = request.form.get('coordinates')
        
        # Check if both coordinates and count are provided
        if not coordinates:
            return {"error": "Missing coordinates or count data"}, 400

        # Parse the coordinates if it's a JSON string
        coordinates = json.loads(coordinates)
        
        # Initialize the lists to store the coordinates based on person type
        adult_input = []
        child_input = []
        elderly_input = []
        width, height = 0,0

        # Handle file data (test-file)
        test_file = request.files['test-file']
        #test_file = request.files.get('test-file')

        if test_file:
            # Save the file in a specific location (uploads folder)
            file_path = './uploads/1.png'
            test_file.save(file_path)
            print(f"Received file: {test_file.filename}")
            with Image.open(file_path) as img:
                width, height = img.size  # Get width and height in pixels
                print(f"Image Dimensions: Width: {width} pixels, Height: {height} pixels")
        else:
            print("No file received")

        # Iterate through the coordinates and classify based on personType
        for item in coordinates:
            x = item.get('x')
            y = item.get('y')  # Not used
            z = item.get('z')
            person_type = item.get('personType')

            # Create a tuple (x, z) for coordinates
            x_min, x_max = 60, 110
            z_min, z_max = -35, -76
            
            scaled_x = scale_coordinate(x, x_min, x_max, 0, height)
            scaled_z = scale_coordinate(z, z_min, z_max, 0, width)

            # Create a tuple (scaled_x, scaled_z) for scaled coordinates
            coord = (round(scaled_z), round(scaled_x))
            print(coord)

            # Classify the coordinates based on the person type
            if person_type == 'Adult':
                adult_input.append(coord)
            elif person_type == 'Child':
                child_input.append(coord)
            elif person_type == 'Elderly':
                elderly_input.append(coord)

        # Print the parsed coordinates
        print(f"Received coordinates: {coordinates}")
        print({
            "adult_input": adult_input,
            "child_input": child_input,
            "elderly_input": elderly_input,
        })

        result=safe_path_res(adult_input, child_input, elderly_input)
        imgs1=['show/best_routes.png', 'show/shortest_path.png']
        print(result)
        
        return redirect(url_for('safe_pathresults'))
    # tf = request.files['test-file']
    # file_path = "../samples/1.png"
    # tf.save(file_path)
    # adult = request.form['adult-input'].splitlines()
    # print(adult)
    # adult = [[int(i.split(',')[0].strip()), int(i.split(',')[1].strip())] for i in adult]
    # child = request.form['child-input'].splitlines()
    # child = [[int(i.split(',')[0].strip()), int(i.split(',')[1].strip())] for i in child]
    # elderly = request.form['elderly-input'].splitlines()
    # elderly = [[int(i.split(',')[0].strip()), int(i.split(',')[1].strip())] for i in elderly]
    
    # #result = hurd_convert(file_path)
    # result=safe_path_res(adult, child, elderly)
    # imgs1=['show/best_routes.png', 'show/shortest_path.png']
    # return render_template("user_ui/safe_path.html", sample_imgs=True, imgs=imgs1, loader=True)


def scale_coordinate(value, min_value, max_value, new_min, new_max):
    return (value - min_value) / (max_value - min_value) * (new_max - new_min) + new_min

@app.route('/fire_path_predict', methods=['GET', 'POST'])
def fire_path_predict():
        coordinates = request.form.get('coordinates')
        count = int(request.form.get('count'))
        
        # Check if both coordinates and count are provided
        if not coordinates or not count:
            return {"error": "Missing coordinates or count data"}, 400

        # Parse the coordinates if it's a JSON string
        coordinates = json.loads(coordinates)
        
        # Initialize the lists to store the coordinates based on person type
        adult_input = []
        child_input = []
        elderly_input = []
        width, height = 0,0

        # Handle file data (test-file)
        test_file = request.files['test-file']
        #test_file = request.files.get('test-file')

        if test_file:
            # Save the file in a specific location (uploads folder)
            file_path = './uploads/1.png'
            test_file.save(file_path)
            print(f"Received file: {test_file.filename}")
            with Image.open(file_path) as img:
                width, height = img.size  # Get width and height in pixels
                print(f"Image Dimensions: Width: {width} pixels, Height: {height} pixels")
        else:
            print("No file received")

        # Iterate through the coordinates and classify based on personType
        for item in coordinates:
            x = item.get('x')
            y = item.get('y')  # Not used
            z = item.get('z')
            person_type = item.get('personType')

            # Create a tuple (x, z) for coordinates
            x_min, x_max = 60, 110
            z_min, z_max = -35, -76
            
            scaled_x = scale_coordinate(x, x_min, x_max, 0, height)
            scaled_z = scale_coordinate(z, z_min, z_max, 0, width)

            # Create a tuple (scaled_x, scaled_z) for scaled coordinates
            coord = (round(scaled_z), round(scaled_x))
            print(coord)

            # Classify the coordinates based on the person type
            if person_type == 'Adult':
                adult_input.append(coord)
            elif person_type == 'Child':
                child_input.append(coord)
            elif person_type == 'Elderly':
                elderly_input.append(coord)

        # Print the parsed coordinates and fighter count
        print(f"Received coordinates: {coordinates}")
        print(f"Fighter count: {count}")
        print({
            "adult_input": adult_input,
            "child_input": child_input,
            "elderly_input": elderly_input,
            "fighter_count": count
        })

        result=fire_path_calc(count, adult_input,child_input,elderly_input)
        print(result)
        imgs1=['show/shortest_path_fire.png']
        
        return redirect(url_for('fire_pathresults'))

@app.route('/pyrosimresults', methods=['GET', 'POST'])
def pyrosimpredict():
    imgs1=['show/dept_pyro.jpeg', 'show/room_pyro.jpeg']
    return render_template("pyrosim.html", result=result.replace('\n', '<br>').replace(' ', '&nbsp;'), sample_imgs=True, imgs=imgs1, loader=True)

@app.route('/pic3dresults', methods=['GET', 'POST'])
def picfrom3dpredict():
    imgs1=[f'show/extracted_images/{i}' for i in os.listdir('../static/show/extracted_images')]
    return render_template("pic_from_3D.html", result=result.replace('\n', '<br>').replace(' ', '&nbsp;'), sample_imgs=True, imgs=imgs1, loader=True)

detail, category, classied_rooms=[], [], []

@app.route('/room_split_convert', methods=['GET', 'POST'])
def room_split_convert():
    global detail
    tf = request.files['test-file']
    ratio = request.form['meter-input']
    file_path = "../samples/1.png"
    tf.save(file_path)
    
    #result = hurd_convert(file_path)
    details, result=room_split_func(file_path, '../static/show/rooms/', ratio)
    detail = details
    imgs1=['show/rooms/annotated_image.png']
    return render_template("user_ui/room_split.html", result=result.replace('\n', '<br>').replace(' ', '&nbsp;'), sample_imgs=True, imgs=imgs1, loader=True)

@app.route('/cat_classify', methods=['GET', 'POST'])
def cat_classify():
    global detail, category
    ratio = request.form['category']
    ratio_one = ratio.split('\r\n')
    category=ratio_one
    #result = hurd_convert(file_path)
    print(detail, ratio_one)
    return render_template("user_ui/room_split.html", result=result.replace('\n', '<br>').replace(' ', '&nbsp;'), sample_details=detail, sample_list=ratio_one, loader=True)

@app.route('/cat_classify_final', methods=['GET', 'POST'])
def cat_classify_final():
    global detail, category, classied_rooms
    classified_room = {}
    for category1 in category:
      selected_details = []
      for detail1 in detail:
        checkbox_name = f"{category1}_{detail1}"
        if checkbox_name in request.form:
          selected_details.append(detail1)
          if selected_details:
            classified_room[category1] = selected_details
    print('Classified')
    classied_rooms=classified_room
    print(classified_room)
    return render_template("user_ui/room_split.html", result=result.replace('\n', '<br>').replace(' ', '&nbsp;'),material=True,sample_list=category, loader=True)

@app.route('/materials_classify', methods=['GET', 'POST'])
def materials_classify():
    global detail, category, classied_rooms
    materials_dict = {}
    for category in category:
      materials_input = request.form.get(category).lower()
      if materials_input:
        materials_list = [material.strip() for material in materials_input.split('\n') if material.strip()]
        materials_dict[category] = materials_list
    print(materials_dict, detail)
    results = []
    for category, materials_ in materials_dict.items():
        print(category)
        for room in classied_rooms[category]:
          print(room)
          area = detail.get(room, {}).get('area_m2', 0)

          if area > 0:
                  # Determine the materials present in the room
                  materials = materials_dict.get(category, [])

                  # Call the fire_extinguisher_calculator for each room
                  print(materials, area)
                  result = fire_extinguisher_calculator(materials, area, room)
                  print(result)
                  results.append(result)
    print(results)
    result='\n'.join(results)
    return render_template("user_ui/room_split.html", result=result.replace('\n', '<br>').replace(' ', '&nbsp;'), loader=True)

if __name__ == "__main__":
    app.run()#debug=True)