# TODO Jose, Valerio:
# - Aggiungere documentazione a tutte le classi/funzioni
# - Capire se vogliamo tenere un seed globale (in teoria bad practice) o vogliamo prendere il seed e propagarlo a chi di dovere
# - Creare dei diagrammi e delle spiegazioni sul funzionamento di tutto, idealmente Jose: ottenimento dati, Valerio: Trans One

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
