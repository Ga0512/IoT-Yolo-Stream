# IoT-Yolo-Stream

**IoT-Yolo-Stream** é um sistema de detecção de objetos em tempo real projetado para rodar em dispositivos Raspberry Pi com arquitetura **aarch64**. Ele utiliza o modelo YOLOv5 convertido para TFLite, integrando rastreamento de objetos e comunicação com APIs externas.

---

## **Pré-requisitos**

1. **Dispositivo compatível:**
   - Raspberry Pi com arquitetura **aarch64** (Raspberry Pi 4 recomendado).
   
2. **Sistema Operacional:**
   - Raspberry Pi OS ou outro sistema baseado em Linux com suporte a aarch64.

3. **Dependências do Sistema:**
   - Python 3.8+
   - OpenCV
   - FastAPI
   - Uvicorn

4. **Instale as dependências necessárias:**
   ```bash
   sudo apt install libssl-dev

   wget https://www.python.org/ftp/python/3.8.18/Python-3.8.18.tgz
   tar zxf Python-3.8.18.tgz
   cd Python-3.8.18/
   ./configure
   make
   sudo make install
   sudo apt update && sudo apt install -y libatlas-base-dev python3.8-pip
   ```

5. **Certifique-se de ter o modelo TFLite:**
   - **`288-2.tflite`**: O modelo YOLOv5 otimizado (incluído no repositório).

---

## **Configuração do Ambiente**

1. Clone o repositório no seu Raspberry Pi:
   ```bash
   git clone https://github.com/Ga0512/IoT-Yolo-Stream.git
   cd IoT-Yolo-Stream
   ```

2. Crie um ambiente virtual (opcional):
   ```bash
   python3.8 -m venv venv
   source venv/bin/activate
   ```

3. Instale as dependências do projeto:
   ```bash
   cd storage
   sh setup.sh
   cd ..
   ```

---

## **Como Executar**

1. Inicie o servidor:
   ```bash
   python3 stream.py
   ```

2. Acesse a interface web no navegador, usando o IP do Raspberry Pi:
   ```
   http://<IP_DO_RASPBERRY>:8000
   ```

---

## **Rotas Disponíveis**

### **`/login`**
- Exibe um formulário para configurar o RTSP e a API.
- Aceita métodos **GET** e **POST**.

### **`/stream`**
- Exibe o streaming de vídeo com detecções e contagem de objetos em tempo real.

---

## **Problemas Comuns**

1. **Erro de dependência**:
   - Certifique-se de que o OpenCV está instalado corretamente:
     ```bash
     pip install opencv-python-headless
     ```

2. **RTSP não funciona**:
   - Verifique as credenciais no `dados.json`.
   - Teste a conexão RTSP usando o VLC ou outra ferramenta.

3. **Modelo ausente:**
   - Certifique-se de que o arquivo `288-2.tflite` está na pasta do projeto.
