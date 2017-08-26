import tensorflow as tf
import tempfile

sequences = [[1, 2, 3], [4, 5, 1], [1, 2]]
label_sequences = [[0, 1, 0], [1, 0, 0], [1, 1]]
def make_seq(sequence, labels):
    example = tf.train.SequenceExample()
    sequencd_length = len(sequence)
    example.context.feature["length"].int64_list.value.append(sequencd_length)
    feature_length_tokens = example.feature_lists.feature_list["tokens"]
    feature_length_labels = example.feature_lists.feature_list["labels"]
    for token, label in zip(sequence, labels):
        feature_length_tokens.feature.add().int64_list.value.append(token)
        feature_length_labels.feature.add().int64_list.value.append(label)
    return example

with tempfile.NamedTemporaryFile() as fp:
    writer = tf.python_io.TFRecordWriter(fp.name)
    for sequence, label_sequence in zip(sequences, label_sequences):
        example = make_seq(sequence, label_sequence)
        writer.write(example.SerializeToString())
    writer.close()