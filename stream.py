# -*- coding: utf-8 -*-

from fastapi import FastAPI, Request, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import os
import json
from PIL import Image
import io
import asyncio
import base64
import threading
import time
from datetime import datetime
from yolov5_tflite_inference import yolov5_tflite
from utils import letterbox_image, scale_coords
from sort import Sort
from queue import Queue
from datetime import datetime

app = FastAPI()

templates = Jinja2Templates(directory="templates")

templates.env.globals.update({
    'encoding': 'utf-8'
})

TEMPLATE_DIR = "templates"  # ou o diretório onde os seus arquivos HTML estão

def load_html_template(filename):
    filepath = os.path.join(TEMPLATE_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

index_html = load_html_template("index.html")
html_template = load_html_template("html_template.html")
html_main = load_html_template("html_main.html")

frame_data = {}
active_users = set()  # Conjunto para rastrear usuários ativos

# Configuração do modelo TFLite
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [5, 195, 281, 195]
totalCount = []


class VideoCapture:
    tipo_piso = 'None'  # Atributo de classe
    count = 0  # Atributo de classe
    update = '--.--'
    
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                self.q.get_nowait()
            self.q.put(frame)

    def read(self):
        return self.q.get()
        
    def get_frame_rate(self):
        frame_rate = self.cap.get(5)
        print('Camera FPS: {}'.format(frame_rate))



def detect_video(weights, img_size, conf_thres, iou_thres):
    start_time = time.time()
    last_modified_time = 0  # Timestamp da última modificação do arquivo
    
    video = None  # Inicializando o objeto VideoCapture
    yolov5_tflite_obj = yolov5_tflite(weights, img_size, conf_thres, iou_thres)

    size = (img_size, img_size)
    no_of_frames = 0
    last_post_time = time.time()

    # Inicia o loop principal
    while True:
        # Verifica se o arquivo foi modificado
        if os.path.exists('dados.json'):
            current_modified_time = os.path.getmtime('dados.json')

            # Se o arquivo foi modificado, recarrega o conteúdo
            if current_modified_time != last_modified_time:
                last_modified_time = current_modified_time
                print("Atualizando dados do arquivo dados.json.")
                
                # Carrega o arquivo dados.json
                with open("dados.json", "r") as json_file:
                    dados = json.load(json_file)

                url = f"rtsp://{dados['user']}:{dados['senha']}@{dados['ip']}/cam/realmonitor?channel={dados['canal']}&subtype={dados['tipo']}"
                
                # Reinicializa o objeto VideoCapture se já estiver em uso
                if video is not None:
                    video.cap.release()  # Libera o vídeo atual
                
                video = VideoCapture(url)  # Inicia o novo stream com os dados atualizados
        else:
            print("Arquivo dados.json não encontrado.")
            time.sleep(1)
            continue

        # Certifica-se de que o vídeo foi inicializado corretamente
        if video is None:
            print("Aguardando configuração inicial do vídeo...")
            time.sleep(1)
            continue

        frame = video.read()
        if frame is None:
            print("Nenhum frame capturado, saindo.")
            break
        
        # Processamento do frame
        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        no_of_frames += 1
        
        image_resized = letterbox_image(Image.fromarray(frame), size)
        image_array = np.asarray(image_resized)
        normalized_image_array = image_array.astype(np.float32) / 255.0
        
        # Detecta objetos no frame
        result_boxes, result_scores, result_class_names = yolov5_tflite_obj.detect(normalized_image_array)
        
        detections = np.empty((0, 5))
        if len(result_boxes) > 0:
            result_boxes = scale_coords(size, np.array(result_boxes), size)
            
            for i, r in enumerate(result_boxes):
                x1, y1 = int(r[0]), int(r[1])
                x2, y2 = int(r[2]), int(r[3])
                
                currentArray = np.array([x1, y1, x2, y2, int(100 * result_scores[i])])
                detections = np.vstack((detections, currentArray))
                
                org = (int(r[0]), int(r[1]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{int(100 * result_scores[i])}%  {str(result_class_names[i])}", org,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
                
                VideoCapture.tipo_piso = str(result_class_names[i])
            
            resultsTracker = tracker.update(detections)
            limits = [100, 200, 300, 400]  # Presumindo valores fixos de limites, substitua conforme necessário
            cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)
            
            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                wb, hb = int(x2) - int(x1), int(y2) - int(y1)
                cx, cy = int(x1 + wb // 2), int(y1 + hb // 2)
                cv2.circle(frame, (cx, cy), 2, (255, 0, 255), cv2.FILLED)
                
                if limits[0] < cx < limits[2] and limits[1] - 5 < cy < limits[1] + 5:
                    if id not in totalCount:
                        totalCount.append(id)
                        cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), cv2.FILLED)
            
            cv2.putText(frame, str(len(totalCount)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (50, 50, 255), 2)
            VideoCapture.count = len(totalCount)
            
            if len(totalCount) > 65535:
                totalCount.clear()
            
            if time.time() - last_post_time >= 60:
                data_hora_atual = datetime.now()
                data_hora_formatada = data_hora_atual.strftime("%Y-%m-%d %H:%M:%S.%f")
                VideoCapture.update = data_hora_formatada
                
                # Chama a função post_api se necessário
                # post_api(len(totalCount), str(result_class_names[i]))  
                last_post_time = time.time()

        # Codifica o frame no formato JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_data['video'] = buffer.tobytes()
        
        # Calcula o FPS
        end_time = time.time()
        print('FPS:', 1 / (end_time - start_time))
        start_time = end_time


'''@app.get("/home", response_class=HTMLResponse)
async def index():
    return HTMLResponse(index_html)'''
    
templates = Jinja2Templates(directory="templates")


@app.get("/login")
async def login_get(request: Request):
    if os.path.exists('dados.json'):
        
        with open("dados.json", "r") as json_file:
            dados = json.load(json_file)
            
            api_value = dados['api']
            ip_value = dados['ip']
            canal_value = dados['canal']
            user_value = dados['user']
    
    else:
        api_value = ''
        ip_value = ''
        canal_value = ''
        user_value = ''

    return templates.TemplateResponse("login.html", {"request": request, "api_value": api_value, "ip_value": ip_value, 
                                                    "canal_value": canal_value, "user_value": user_value})

@app.post("/login")
async def login_post(
    request: Request,
    user: str = Form(...),
    password: str = Form(...),
    tipo: str = Form(None),
    rtsp: str = Form(...),
    chanel: str = Form(...),
    api: str = Form(...)
):
    tipo = tipo if tipo is not None else 'Padrao'
    tipo = 0 if tipo == "Padrao" else 1
    
    print(api)
    dados = {
        "user": user,
        "senha": password,
        "tipo": tipo,
        "ip": rtsp,
        "canal": chanel,
        "api": api
    }

    with open('dados.json', 'w') as json_file:
        json.dump(dados, json_file, indent=5)
        
        
    time.sleep(2.5)
    return RedirectResponse(url="/")
     

@app.post("/log_enter")
async def log_enter(request: Request):
    data = await request.json()
    frame_name = data.get("frame_name", "")
    if frame_name:
        active_users.add(frame_name)
    print(f"User entered the page for {frame_name} at {datetime.now()}")
    return {}

@app.post("/log_exit")
async def log_exit(request: Request):
    data = await request.json()
    frame_name = data.get("frame_name", "")
    if frame_name:
        active_users.discard(frame_name)
    print(f"User left the page for {frame_name} at {datetime.now()}")
    return {}

@app.get("/video/{filename:path}")
async def get_video(filename: str):
    html_content = html_template.replace("{frame_name}", filename)
    return HTMLResponse(html_content)

@app.get("/stream/{filename:path}")
async def stream(filename: str):
    async def new_frame():
        while True:
            if filename in active_users:
                if 'video' in frame_data:
                    # Converte os bytes do frame para uma imagem PIL
                    image = Image.open(io.BytesIO(frame_data['video']))
                    
                    # Salva a imagem em um buffer de bytes
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='JPEG')
                    img_byte_arr.seek(0)
                    
                    # Codifica a imagem em base64
                    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                    
                    yield f"data: {encoded_img}\n\n"
            await asyncio.sleep(1/30)  # Ajuste o intervalo conforme necessário

    return StreamingResponse(new_frame(), media_type='text/event-stream')



@app.get("/info_stream")
async def info_stream():
    async def event_generator():
        while True:
            # Construa um dicionário com as informações desejadas
            data = {
                "piso": VideoCapture.tipo_piso,
                "count": VideoCapture.count,
                "update": VideoCapture.update
            }
            # Converta o dicionário para JSON
            json_data = json.dumps(data)
            # Envie os dados JSON como uma string
            yield f"data: {json_data}\n\n"
            await asyncio.sleep(1)  # Intervalo de 1 segundo entre as atualizações

    return StreamingResponse(event_generator(), media_type="text/event-stream")




@app.get("/")
@app.post("/")
async def index():
    if not os.path.exists('dados.json'):
        return RedirectResponse(url="/login")
        
    video_files = list(frame_data.keys())
    buttons = ''.join([f'<button onclick="window.location.href=\'/video/{file}\'">{file}</button><br/>' for file in video_files])
    
    html_content = html_main.replace("{buttons}", buttons)\
                        .replace("{frame_name}", "")

    return HTMLResponse(html_content)

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, default='288-2.tflite', help='model.tflite path(s)')  
    parser.add_argument('--img_size', type=int, default=288, help='image size') 
    parser.add_argument('--conf_thres', type=float, default=0.75, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.75, help='IOU threshold for NMS')

    opt = parser.parse_args()
    
    # Inicializa a detecção em uma thread separada
    detection_thread = threading.Thread(target=detect_video, args=(opt.weights, opt.img_size, opt.conf_thres, opt.iou_thres))
    detection_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    detection_thread.join()
