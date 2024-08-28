# Megatron_DeepFake

- [x] Terminare download Dataset
  - [x] effettuare un download di tutti i video specificando original_youtube_videos come dataset
  - [x] effettuare un download di tutte le info specificando original_youtube_videos_info come dataset
  - [x] effettuare un download di tutti i video specificando original-DeepFakeDetection_original come dataset
  - [x] Scaricare 100 video di ogni tecnica di manipolazione
- [x] Creare Dataloader per estrazione frame con comportamento lazy iterator e preparazione funzioni di estrazione crop del viso e calcolo DepthMask.
- [x] Impostare primo step pipeline di face detection ed extraction sfruttando Dlib library.
- [x] Impostare secondo step pipeline estrazione Depth Mask con DepthAnything, così da avere maschera e RGB pronti da dare in pasto alle due RepVit networks.
- [x] Impostare terzo step pipeline: 2 RepVit Networks, una che lavora sulla DepthMask e un'altra sull'RGB.
- [x] Impostare logica di output delle due RepVit: Vettore di tuple (ogni tupla contiene un embedding per l'RGB e un embedding per la Mask)
- [x] Implementazione Transformer con cross attention e successiva classificazione
- [x] Classificazione con Softmax
- [ ] aaaaa
- [x] Implementare classe e logiche di training
- [ ] Implementare classe e logiche di inferenza
- [ ] Ottimizzazione degli Iperparametri per il transformer. 
- [ ] Ottimizzazione iperparametri Training
- [ ] Addestramento
<!-- - [ ] Classificazione Softmax vs Classificazione Geometrica.  -->
<!-- - [ ] Ricerca miglior approccio per concatenazione di depth_embedding e rgb_embedding -->


**Filenames form:**
Original sequences: 
- All original sequences saved in the youtube folder to integers between 0 and 999
- The original DeepFakeDetection sequences are stored in the actors folder. The sequence filenames are of the form "actor number__scene name".

Manipulated sequences:
- FaceForensics++: All filenames are of the form "target sequence_source sequence".
- DeepFakeDetection: "target actor_source actor__sequence name__8 charactor long experiment id".

