# UrbanSound8k

>Python, Convolutional Neural Networks, UrbanSound8k, FreeSound

Paper:

* Sajjad Abdoli; Patrick Cardinal; Alessandro Lameiras Koerich (2019). ***End-to-End Environmental Sound Classification using a 1D Convolutional Neural Network.*** arXiv: [https://doi.org/10.48550/arXiv.1904.08990](https://doi.org/10.48550/arXiv.1904.08990)


Creado por:

- **V. D. Betancourt**




## 📑 **Sobre el Artículo**

<details>
    <summary> Expandir </summary>

Este artículo describe un método para clasificar sonidos ambientales usando una Red Neuronal Convolucional (CNN) de 1 dimensión (1D).
Este método aprende directamente del audio, usando varias capas convolucionales para capturar detalles finos del tiempo en el sonido y aprender filtros diversos que ayudan en la tarea de clasificación. Puede manejar señales de audio de cualquier longitud, dividiéndolas en segmentos superpuestos mediante una ventana deslizante.
Se evaluaron diferentes estructuras, incluida una que inicia con un banco de filtros Gammatone, que imita cómo el oído humano procesa los sonidos en la cóclea.
El método se probó con el dataset UrbanSound8k, alcanzando un 89% de precisión media, superando a la mayoría de los métodos actuales que utilizan características diseñadas manualmente o representaciones en 2D. Además, este enfoque requiere menos parámetros que otros métodos, lo que significa que necesita menos datos para el entrenamiento.



</details>

----------------




## 💡 **Sobre el Dataset: UrbanSound8K**

<details>
    <summary> Expandir </summary>


### Descripción

Este conjunto de datos contiene 8732 fragmentos de sonidos urbanos etiquetados (<=4s) de 10 clases: aire acondicionado, bocina de carro, niños jugando, ladrido de perro, taladrado, ralentí del motor, disparo de arma, martillo neumático, sirena y música callejera. Las clases provienen de la taxonomía de sonidos urbanos.
Todos los fragmentos provienen de grabaciones de campo subidas a www.freesound.org. Los archivos están pre-organizados en diez grupos (carpetas nombradas de fold1 a fold10) para ayudar en la reproducción y comparación con los resultados de clasificación automática reportados en el artículo mencionado.

Además de los fragmentos de sonido, también se proporciona un archivo CSV que contiene metadatos sobre cada fragmento.



### Archivos de Audio Incluidos

8732 archivos de audio de sonidos urbanos (ver descripción arriba) en formato WAV. La tasa de muestreo, profundidad de bits y número de canales son los mismos que los del archivo original subido a Freesound (y por lo tanto pueden variar de archivo a archivo).


### Archivos de Meta-Datos Incluidos

Metadata: UrbanSound8k.csv

Columnas:

*	**slice_file_name**: El nombre del archivo de audio. El nombre sigue el siguiente formato: [fsID]-[classID]-[occurrenceID]-[sliceID].wav, donde: [fsID] = el ID de Freesound de la grabación de la cual se ha tomado este fragmento (slice) [classID] = un identificador numérico de la clase de sonido (ver descripción de classID abajo para más detalles) [occurrenceID] = un identificador numérico para distinguir diferentes ocurrencias del sonido dentro de la grabación original [sliceID] = un identificador numérico para distinguir diferentes fragmentos tomados de la misma ocurrencia.

* **fsID**: El ID de Freesound de la grabación de la cual se ha tomado este fragmento (slice).

*	**start**: El tiempo de inicio del fragmento en la grabación original de Freesound.

*	**end**: El tiempo de fin del fragmento en la grabación original de Freesound.

*	**salience**: Una calificación de saliencia (subjetiva) del sonido. 1 = primer plano, 2 = fondo.

*	**fold**: El número de grupo (1-10) al cual ha sido asignado este archivo.

*	**classID**: Un identificador numérico de la clase de sonido: 0 = air_conditioner 1 = car_horn 2 = children_playing 3 = dog_bark 4 = drilling 5 = engine_idling 6 = gun_shot 7 = jackhammer 8 = siren 9 = street_music.

*	**class**: El nombre de la clase: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music.





### Recomendaciones de UrbanSound8K

1.	**¡No reorganices los datos! Usa los 10 pliegues (folds) predefinidos y realiza validación cruzada de 10 pliegues (no de 5 pliegues).**

    Los experimentos realizados por la gran mayoría de publicaciones que usan UrbanSound8K, evalúan los modelos de clasificación mediante validación cruzada de 10 pliegues usando las divisiones predefinidas. Recomendamos encarecidamente seguir este procedimiento.
  
  * **¿Por qué?**

    - Si reorganizas los datos (por ejemplo, combinando los datos de todos los grupos y generando una división aleatoria de entrenamiento/prueba) estarás colocando   incorrectamente muestras relacionadas tanto en los conjuntos de entrenamiento como de prueba, lo que lleva a puntajes inflados que no representan el rendimiento de tu modelo en datos no vistos. En pocas palabras, tus resultados serán incorrectos. Tus resultados no serán comparables con resultados anteriores en la literatura, lo que significa que cualquier afirmación de una mejora sobre investigaciones anteriores será inválida. Incluso si no reorganizas los datos, evaluar usando divisiones diferentes (por ejemplo, validación cruzada de 5 pliegues) significará que tus resultados no son comparables con investigaciones anteriores.

2.	**¡No evalúes solo en una división! Usa validación cruzada de 10 pliegues (folds) (no de 5 pliegues) y promedia los puntajes.**

    Se han visto informes que solo proporcionan resultados para una única división de entrenamiento/prueba, por ejemplo, entrenar en los grupos 1-9, probar en el grupo 10 y reportar un único puntaje de precisión. Aconsejamos en contra de esto. En su lugar, realiza validación cruzada de 10 pliegues usando los grupos proporcionados e informa el puntaje promedio.

  * **¿Por qué?**

    - No todos los grupos son tan "fáciles". Es decir, los modelos tienden a obtener puntajes mucho más altos cuando se entrenan en los grupos 1-9 y se prueban en el grupo 10, comparado con (por ejemplo) entrenar en los grupos 2-10 y probar en el grupo 1. Por esta razón, es importante evaluar tu modelo en cada una de las 10 divisiones e informar la precisión promedio. De nuevo, tus resultados no serán comparables con resultados anteriores en la literatura.




</details>

----------------




## 📊 **Pre-Processing**

<details>
    <summary> Expandir </summary>

* **Dataset UrbanSound8k**: Utilizado para la evaluación de modelos con 8,732 clips de audio a diferentes tasas de muestreo, reducidos a 16 kHz.

* **Segmentación de Audio**: Archivos de audio segmentados en 16,000 muestras con marcos superpuestos en un 50%. Esto podría replicarse mediante el preprocesamiento de datos de audio para ajustarse a los requisitos de entrada del modelo.


</details>

----------------






##  🧬 **Arquitectura y Training**

<details>
    <summary> Expandir </summary>

* **Capas Convolucionales**: La experimentación mostró un aumento en la precisión con más capas convolucionales hasta cuatro, después de lo cual la dimensión del mapa de características se minimiza. Esto sugiere que un modelo con hasta cuatro capas convolucionales es óptimo para esta tarea.

* **Evaluación en Diferentes Longitudes de Audio**: El modelo se adaptó para varios tamaños de entrada, logrando la mayor precisión con entradas de 16,000 muestras. Esto enfatiza la importancia de considerar la longitud de entrada en el diseño de la arquitectura del modelo.


</details>

----------------




##  🔧 **Fine-Tuning y Mejoras**

<details>
    <summary> Expandir </summary>

* **Solapamiento de Ventanas y Tamaño de Marco**: Se probaron diferentes configuraciones de solapamiento de ventanas y tamaños de marcos de audio, resaltando la importancia de estos parámetros en el rendimiento del modelo.

* **Mejoras en la Arquitectura**:

  - Reemplazar la ventana deslizante de Hamming por una rectangular para una posible mejora en la precisión.

  - Aumentar el solapamiento de ventanas durante la segmentación de audio puede mejorar el rendimiento.

  - Inicializar la primera capa convolucional con un banco de filtros Gammatone y hacerla no entrenable para mejorar la precisión.


</details>

----------------




##  🎚️**Optimización**

<details>
    <summary> Expandir </summary>

* **Optimización**: Se utilizó el optimizador Adadelta con una tasa de aprendizaje predeterminada de 1.0, adaptando dinámicamente la tasa de aprendizaje durante la optimización.

* **Agregación de Predicciones al Estilo de Conjunto**: Combinar predicciones de marcos de audio segmentados puede mejorar el rendimiento general de la clasificación.



</details>

----------------




## 📜 **Conclusiones**

<details>
    <summary> Expandir </summary>

* **Eficiencia de la Arquitectura**: La CNN 1D propuesta, con tres a cinco capas convolucionales, aprende de manera eficiente representaciones de filtros directamente de la forma de onda de audio, superando a las CNNs 2D de última generación utilizadas para la clasificación de sonidos ambientales en términos de precisión y complejidad del modelo.

* **Reducción de la Complejidad del Modelo**: A pesar de tener menos parámetros que muchas arquitecturas de CNN 2D, la CNN 1D logra una precisión media significativamente mayor, mostrando la eficiencia y efectividad del modelo.

* **Potencial para la Integración de Representaciones 1D y 2D**: La observación de que puede haber aspectos complementarios entre los filtros 1D (aprendidos directamente de las formas de onda de audio) y los filtros 2D (aprendidos de los espectrogramas) sugiere una dirección futura de investigación. Integrar representaciones 1D y 2D podría potencialmente mejorar el rendimiento de clasificación para ciertas clases de sonido.

* **Necesidad de más Investigación**: El artículo indica la necesidad de investigar más sobre los filtros ruidosos de las capas convolucionales intermedias sin frecuencias dominantes. Esto apunta a posibles mejoras en el modelo al refinar estos filtros.


</details>

----------------




## ㊙️ **Código**

<details>
    <summary> Expandir </summary>

* [Modelo](https://github.com/vbleal/UrbanSound8k/blob/main/GH_UrbanSound8k_Keras.ipynb)
  

</details>

----------------



