# SoundSense trained model

soundsense_saved_model includes frozen pretrained YAMNet plus the trained head. Load with tf.saved_model.load, then call signatures["serving_default"] with waveform=<one-dimensional float32 Tensor>. Input must be mono, 16 kHz, with values in -1..1. It is cropped/padded to the first five seconds. Use the notebook load_audio function to preprocess WAV files consistently.

soundsense_head.keras takes a batch of mean 1024-dimensional YAMNet embeddings; it cannot classify raw WAV files by itself.

Class order: class_names.json. Scores are not calibrated probabilities. The system always selects a supported class; it has no unknown-class detector. This is one fixed holdout on a small benchmark, not five-fold cross-validation. The ESC source notes possible class-dependent preprocessing cues. YAMNet was pretrained on external AudioSet data; possible pretraining overlap has not been exhaustively audited. Use independent field recordings for wider evaluation.

Dataset: ESC-10, Karol J. Piczak, 2015. https://github.com/karolpiczak/ESC-50. Retain the included source attribution and license files. YAMNet model: https://tfhub.dev/google/yamnet/1. Tutorial: https://www.tensorflow.org/tutorials/audio/transfer_learning_audio.
