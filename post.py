import requests
import json
from datetime import datetime

# Formatando a data e hora no formato desejado
url = "http://192.168.0.101/api/tag_historic"

def post_api(n_pisos, formato):
    data_hora_atual = datetime.now()
    data_hora_formatada = data_hora_atual.strftime("%Y-%m-%d %H:%M:%S.%f")
    
    data = [
       {
        "tag_id": "e87b2702-0f7e-46bd-a8ae-35ce180ac550",
        "parameter": "",
        "description": "Linha 3 - Descarga do Forno",
        "orig_dbl_v": n_pisos,
        "formato": formato,
        "dbl_v": 0,
        "created_at": str(data_hora_formatada)
      }
    ]

    # Enviar a solicitação POST com os dados JSON
    response = requests.post("http://192.168.0.101/api/tag_historic", json=data)

    '''# Verificar o status da resposta
    if response.status_code == 200:
        print("POST BEM SUCEDIDO 200")
    else:
        print("Erro ao enviar pedido POST:", response.status_code)'''

    return response