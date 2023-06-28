## Descrição
Esse trabalho é resultado de um projeto de extensão realizado no Instituto Federal de Educação, Ciências e Tecnologia do Rio Grande do Norte (IFRN), Campus Pau dos Ferros, produzido em conjunto pelos alunos Luiz Henrique Bessa Régis e Michael César Fernandes Lopes, sobre orientação do Prof. Demétrios Coutinho e do Prof. Thiago de Oliveira. A aplicação tem como base o projeto Deep Learning and Applications, de Lucas Resck, cujo link do repositório está no final deste documento, e a YOLO, uma rede neural convolucional utilizada na detecção de objetos em imagens. Desse modo, foi possível fazer a inferência de aglomerações tanto em vídeos ao vivo quanto em gravados, além de oferecer arquivos que permitem a plotagem de informações cruciais para a plotagem de gráficos a fim de facilitar a análise dos dados.

## Instalação
#### Pré-requisitos
Você irá precisar, em sua máquina, de:
- [Python 3.10](https://www.python.org/downloads/release/python-3100/)
- [Anaconda](https://www.anaconda.com/download/)
- [YOLOV5](https://github.com/ultralytics/yolov5)

Na instalação da YOLO, você poderá optar pelo modelo do tamanho de sua preferência, mas se sua máquina possui especificações simples ou não possui placa de vídeo, opte pela YOLOV5s. 
Se você possui uma placa de vídeo que [suporta CUDA](https://developer.nvidia.com/cuda-gpus), recomendamos que você a instale a [CUDA Toolkit 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive), pois recrutará processamento da GPU para auxiliar na computação das pessoas sendo detectadas pela YOLO durante a inferência. 

Após a clonagem do repositório, instale as bibliotecas utilizadas usando o seguinte comando:
#### Dependências
##### Bibliotecas
```
pip install -r requirements.txt
```

##### PyTorch Sem CUDA
Caso você não tenha instalado CUDA, sobrescreva a instalação do PyTorch com a [versão para CPU](https://pytorch.org/get-started/locally/) como plataforma de computação.

##### Conda
Atenção: para executar a versão do Crowd Detection do notebook Jupyter, instale os mesmos pacotes em `requirements.txt` usando `conda`.

## Referências
- [Deep Learning and Applications](https://github.com/lucasresck/deep-learning-and-applications)
