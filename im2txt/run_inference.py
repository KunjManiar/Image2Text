

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os
sys.path.append(os.getcwd())

import math
import os


import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)




def rename_model_ckpt():
# C:\Users\kunjm\Desktop\studies\SEM 6\AI\Project\Git\Im2txt
    BASE_DIR = 'C:/Users/kunjm/Desktop/studies/SEM 6/AI/Project/Git/Im2txt/'
    
    vars_to_rename = {
    "lstm/BasicLSTMCell/Linear/Matrix": "lstm/basic_lstm_cell/kernel",
    "lstm/BasicLSTMCell/Linear/Bias": "lstm/basic_lstm_cell/bias",
    }
    
    new_checkpoint_vars = {}
    
    reader = tf.train.NewCheckpointReader(FLAGS.checkpoint_path)
    for old_name in reader.get_variable_to_shape_map():
      if old_name in vars_to_rename:
        new_name = vars_to_rename[old_name]
      else:
        new_name = old_name

      # print(old_name)
      new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(new_checkpoint_vars)

    # print("BASE_DIR = ", BASE_DIR)

    with tf.Session() as sess:
      sess.run(init)
      saver.save(sess, os.path.join(BASE_DIR, "im2txt/model/Hugh/train/newmodel.ckpt-2000000"))
    # print("checkpoint file rename successful... ")

def main(_):

  # Change tensor name
  #rename_model_ckpt()

  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    for filename in filenames:
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      print("Captions for image %s:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        # print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
        print("  %d) %s" % (i, sentence))


if __name__ == "__main__":
  tf.app.run()
