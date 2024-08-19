# Megatron_DeepFake

- [ ] Terminare download Dataset
  - [ ] effettuare un download di tutti i video specificando original_youtube_videos come dataset
  - [ ] effettuare un download di tutte le info specificando original_youtube_videos_info come dataset
  - [ ] effettuare un download di tutti i video specificando original-DeepFakeDetection_original come dataset
  - [ ] Scaricare 100 video di ogni tecnica di manipolazione
- [ ] Effettuare Pre-processing


**Filenames form:**
Original sequences: 
- All original sequences saved in the youtube folder to integers between 0 and 999
- The original DeepFakeDetection sequences are stored in the actors folder. The sequence filenames are of the form "actor number__scene name".

Manipulated sequences:
- FaceForensics++: All filenames are of the form "target sequence_source sequence".
- DeepFakeDetection: "target actor_source actor__sequence name__8 charactor long experiment id".

