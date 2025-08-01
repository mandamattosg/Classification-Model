# Classificador de Imagens: Pessoa vs. Cavalo

Este projeto visa desenvolver um modelo de Machine Learning capaz de classificar imagens como "Pessoa" ou "Cavalo", utilizando técnicas avançadas de Visão Computacional a partir de Redes Neurais Convolucionais.

## 📦 Arquivos
  Neste repositório estão presentes os arquivos:
  * horse-or-human: Dataset para treino;
  * validation-horse-or-human: Dataset para validação;
  * model: Diretório contendo o modelo;
  * metrics_results: Gráficos para análise do modelo;
  * validation_results: Imagens de validação com a predição do modelo;
  * data_loader.py: Módulo para carregamento do dataset;
  * train.py: Código de treinamento do modelo;
  * validate_model.py: Analisa as predições do modelo com o dataset de validação e salva imagens em validation_results;
  * inference_api.py: API para realizar predições em tempo real.
    
## 📋 Justificativa das Escolhas Técnicas
1. Arquitetura do Modelo:
   
  Para a construção do modelo de classificação, a escolha foi baseada na técnica de Transfer Learning utilizando a arquitetura MobileNetV2. Essa técnica é muito eficiente pois o MobileNetV2 (modelo conhecido por sua capacidade de classificação) já foi pré-treinado em um vasto conjunto de dados (ImageNet), aprendendo a reconhecer características visuais genéricas. Ao "congelar" as camadas convolucionais desse modelo e adicionar novas camadas de classificação, o modelo é capaz de aprender a distinguir as classes de interesse de forma mais rápida e eficiente. Para a implementação do modelo foi usado o TensorFlow com sua API de alto nível, o Keras.

2. Camadas do Modelo: 
   ```
   x = base_model.output
   x = GlobalAveragePooling2D()(x)
   x = Dense(128, activation='relu')(x)
   x = Dropout(0.3)(x)
   predictions = Dense(1, activation='sigmoid')(x)
   ```
  A base do MobileNetV2 atua como um extrator de características. Em seguida foi criado a estrutura:
  * GlobalAveragePooling2D: Calcula a média de cada mapa de características da imagem até que cada dimensão espacial seja 1 (faz um resumo da imagem).
  * Dense: Recebe o resumo da imagem e procura por padrões específicos que ajudem a diferenciar entre as classes. O número 128 representa a quantidade de neurônios e a função 'relu' permite que o modelo aprenda relações não lineares e mais complexas na imagem.
  * Dropout: Técnica para evitar overfitting, desativa aleatoriamente 30% dos neurônios da camada anterior.
  * Dense: Camada de saída, recebe os padrões finais e toma a decisão. A função 'sigmoid' transforma a saída do neurônio em um valor entre 0 e 1 que significa a probabilidade de ser um humano na imagem.
3. Regra de Decisão: 
  * Se a saída for maior que 0.5, a classe predita é Pessoa.
  * Se a saída for menor ou igual a 0.5, a classe predita é Cavalo.
4. Função Objetivo e Otimizador
  * Função Objetivo (loss): 'binary_crossentropy' foi escolhida por ser a função de perda padrão e mais adequada para problemas de classificação binária.
  * Otimizador: Adam com um learning_rate de 1e-4 foi utilizado pois evita grandes alterações nos pesos das camadas, mantendo o conhecimento pré-adquirido.
5. Estratégia de Validação e Treinamento
  * Data Augmentation: Para evitar overfitting e aumentar a quantidade de dados de treino, a técnica de data augmentation foi empregada com ImageDataGenerator.
  * Early Stopping: Para garantir que o modelo não treine mais do que o necessário e comece a ter overfitting, um callback de EarlyStopping foi implementado. Ele monitora a perda de validação (val_loss) e para o treinamento se não houver melhora por 5 épocas.

## 💻 Instruções para Setup e Execução
  * Pré-requisitos: Python3.8+ (foi usado o Python3.10) e Pip
1. Clonar o repositório:
   ```
   git clone https://github.com/mandamattosg/Classification-Model.git
   cd Classification-Model
   ```
2. Instalar dependências:
    ```
    python -m venv venv
    source venv/bin/activate  # No Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
   ```
3. Treinamento do modelo:
    ```
    python train.py
   ```
4. Validação do modelo:
   
   Para visualizar as predições do modelo no dataset de validação:
    ```
    python validate_model.py
   ```
6. Execução da API de Inferência:
   A API expõe um único endpoint para realizar a inferência.
  * Endpoint: /predict
  * Método: POST
  * Formato da Imagem: A imagem deve ser enviada como um arquivo no corpo da requisição, utilizando o formato multipart/form-data.
    
   Para ativar o funcionamento da API REST:
   ```
   python -m uvicorn inference_api:app --reload
   ```
   Agora, você pode solicitar a inferência como preferir, por exemplo:
   * Usando CURL no CMD:
     ```
     curl -X POST "http://127.0.0.1:8000/predict" -F "file=@validation-horse-or-human/horses/horse1-000.png"
     ```
  * Usando CURL no Windows PowerShell:
      ```
     curl.exe -X POST "http://127.0.0.1:8000/predict" -F file=@'validation-horse-or-human/horses/horse1-000.png'
     ```
  Exemplo de resposta:

  A resposta é um JSON constendo o texto da classificação e a confiança da predição.
  ```
  {
  "prediction": "Is a horse",
  "confidence": 0.9987
  }
```

## 📈 Análise de Métricas
Para entender e analisar a performance do modelo foram analisadas a Loss, Accuracy e Confusion Matrix que podem ser vistas abaixo. \
    <img width="400" height="300" alt="loss_plot" src="https://github.com/user-attachments/assets/82674523-957c-4807-aa27-e697218bdf8a" />
    <img width="400" height="300" alt="accuracy_plot" src="https://github.com/user-attachments/assets/88f2ba4f-8d17-4fd2-b423-8f11704303de" />
    <img width="400" height="300" alt="confusion_matrix" src="https://github.com/user-attachments/assets/abed2158-ce86-418a-97a1-f8abe7662736" />
    
Percebe-se que o mecanismo de Early Stopping encerrou o treinamento na época número 22. A loss de treino e validação diminui rapidamente nas primeiras épocas. Ambas as curvas de accuracy se estabilizam em um patamar alto (98.83%). Já na matriz vemos que das 128 imagens de cavalos, o modelo acertou 127 e errou apenas 1, além de ter acertado todas as previsões de humanos.
Esses resultados são ótimos e comprovam que o modelo é capaz de diferenciar imagens de pessoas e cavalos.

Para ter ainda mais certeza da sua acurácia, através do código 'validate_model.py' é possível obter os resultados em uma pasta chamada 'validation_results' onde podemos analisar calmamente a predição de cada imagem do conjunto de dados de validação. Esse tipo de análise é interessante pois podemos identificar padrões nas imagens que o modelo teve mais dificuldade em classificar. Por exemplo,
nesse teste realizado, o modelo errou 3 classificações de cavalos, como pode ser visto nas imagens abaixo: \
<img width="281" height="366" alt="incorrect_horse6-161" src="https://github.com/user-attachments/assets/390efd7f-3b6d-4f3c-8814-01ff086d4aea" />
<img width="281" height="366" alt="incorrect_horse5-103" src="https://github.com/user-attachments/assets/44d2b2af-0108-4730-9d97-90ea4f1038ae" />
<img width="281" height="366" alt="incorrect_horse3-498" src="https://github.com/user-attachments/assets/766209ff-9ba5-4e88-bcce-d7f3272265c1" />

É perceptível que o modelo tem dificuldade para classificar imagens em que o cavalo está somente em 2 patas e/ou sua estrutura corporal não está clara. Ainda assim, vale notar que sua acurácia chega a 98.83% representando um ótimo desempenho geral, portanto, sim, o modelo atende e supera o problema proposto.

Para melhorias e próximos passos é possível considerar:
* Fine-Tuning: Em vez de manter todas as camadas da base do MobileNetV2 congeladas, poderíamos descongelar as últimas camadas e treiná-las com uma taxa de aprendizado muito baixa. Isso permitiria que o modelo se ajuste ainda mais às particularidades do dataset.
* Aumentar o Dataset: Adicionar mais dados ao conjunto de treino, provenientes de outras fontes, poderia tornar o modelo ainda mais robusto a variações de iluminação, pose e fundo.
* Deploy em Cloud: A API pode ser containerizada com Docker e implantada em serviços de nuvem (AWS, GCP, Azure) para escalabilidade e disponibilidade em produção.
* Otimização de Latência: Para aplicações em tempo real, seria possível otimizar o modelo com técnicas de quantização usando TensorFlow Lite.

