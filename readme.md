# Reto: Store sales: Time series forecasting.

## Descripción del reto:

Este reto consistió en aplicar Machine Learning para predecir
las ventas de Corporación Favorita, una tienda minorista de Ecuador.

## Contexto de normatividades:

La información (datasets) utlizados para este reto se encuentran bajo la siguiente licencia por la plataforma Kaggle:

COMPETITION DATA.
"Competition Data" means the data or datasets available from the Competition Website for the purpose of use in the Competition, including any prototype or executable code provided on the Competition Website. The Competition Data will contain private and public test sets. Which data belongs to which set will not be made available to participants.

A. Data Access and Use. You may access and use the Competition Data for non-commercial purposes only, including for participating in the Competition and on Kaggle.com forums, and for academic research and education. The Competition Sponsor reserves the right to disqualify any participant who uses the Competition Data other than as permitted by the Competition Website and these Rules.

B. Data Security. You agree to use reasonable and suitable measures to prevent persons who have not formally agreed to these Rules from gaining access to the Competition Data. You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition. You agree to notify Kaggle immediately upon learning of any possible unauthorized transmission of or unauthorized access to the Competition Data and agree to work with Kaggle to rectify any unauthorized transmission or access.

C. External Data. You may use data other than the Competition Data (“External Data”) to develop and test your Submissions. However, you will ensure the External Data is publicly available and equally accessible to use by all participants of the Competition for purposes of the competition at no cost to the other participants. The ability to use External Data under this Section 7.C (External Data) does not limit your other obligations under these Competition Rules, including but not limited to Section 11 (Winners Obligations).

## Licencias de las herramientas utilizadas

Todas las herramientas utilizadas para la elaboración de la solución son de uso libre. Entre ellas se encuentran:

- Python
- Pandas
- Sci-kit Learn
- React
- Flask

## Solución:

Para la solución de este reto, se llevó a cabo un proceso completo de
limpieza, transformación y análisis de los datos. Esto nos permitió enriquecer
la modelación para tener una mayor precisión en las predicciones.

Como parte del proceso de hallar la solución final utilizamos un algoritmo de
regresión lineal. Este algoritmo nos dio un resultado con una precisión baja, pero
nos ayudó a comprender un poco mejor los datos y la dirección que debiamos tomar.

Para la solución final utilizamos el algoritmo de Random Forest con nuestras
variables generadas a partir de análisis de datos.

## Aplicación para la solución:

Una vez que se hizo la solución, hicimos una aplicación en React que te permite
hacer queries al modelo, pudiendo entrenar los modelos de regresión lineal y
random forest desde el front end. Para hacer esta conexión con el modelo, elaboramos
un API REST en Flask que entrena el modelo que se le solicite a través de peticiones HTTP.

El código fuente de front end y back end se encuentra en este repositorio en las carpetas
con sus respectivos nombres.

A continuación se encuentra una demostración de la aplicación funcionando:
https://youtu.be/Qa5e5sz4ryw
