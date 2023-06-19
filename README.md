# NN-architecture-optimizer

## Etap - 0 - Data Preprocessing
- stworzenie jednego folderu z danymi
- transformacja obrazków do jednego rozmiaru
- poprawienie zbilansowania danych
- transformacja do skali szarości
- wyrównanie histogramu ???
- podział na zbiór testowy i treningowy 
- podział na mniejszy zbiór treningowy dla algorytmu genetycznego ???
- przygotowanie danych do treningu (tak aby łatwo je było wgrać, etykiety)
- ???
- 
## Etap - 1 - Genetic Algorithm
Algorytm genetyczny ma za zadanie dobrać hiperparametry sieci neuronowej:  
- ilość warstw
- ilość neuronów w warstwie  

TODO:
- zaimplementować logikę algorytmu
- operator mutacji
  - operator krzyżowania
- 
- tworzenie sieci z genotypu
- ocena fenotypu

## Etap - 2 - Hierarchic Memetic Search
Algorytm HMS ma za zadanie wytrenować sieć neuronową
- Usage:
  -  clone [repo](https://github.com/WojtAcht/hms/tree/main)
  -  run `model_optimalization.R` (configure minoconda environment)
  -  TODO dla szybszego działania trzeba dodać obsługę GPU dla kerasa w minicondzie w R :(
