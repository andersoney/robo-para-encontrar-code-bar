# robo-para-encontrar-code-bar
Robô para encontrar codebar em imagens, podendo assim ser incorporado a um drone para localização de objetos em um estoque ou similar.

## Comando de execução
Para o do zero é ultilizado o comando:
```
python dozero.py -i dump.jpg -w yoloDados\yolov3.weights -c yoloDados\yolov3.cfg -cl yoloDados\YoloNames.names
```
Para o antigo é usado o comando:
```
python main.py
```

Foi inserido mais uma forma de executar o codigo para tratar um conjunto maior de imagens, este modo é executado do mesmo modo que o algoritmo feito do zero, porem sem o parametro imagem pois deve ser inserido no codigo. Como escrito abaixo.
```
python alImage.py -w yoloDados\yolov3.weights -c yoloDados\yolov3.cfg -cl yoloDados\YoloNames.names
```


### Caixas tem 14 digitos de ean.


## Artigos visitados.

https://storage.googleapis.com/tfjs-examples/simple-object-detection/dist/index.html

para o arquivo main.py

https://www.youtube.com/watch?v=PdErmEf-FCs
