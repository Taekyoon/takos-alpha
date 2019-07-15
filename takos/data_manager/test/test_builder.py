from pathlib import Path
from takos.data_manager.builder import TagSequenceDatasetBuilder
from torch.utils.data import DataLoader


def test_ner_dataset_builder_build_dataloader_as_default():
    input_path = './data_manager/test/test_dataset/ner/input.txt'
    label_path = './data_manager/test/test_dataset/ner/output.txt'

    dataset_dir = Path('./data_manager/test/test_dataset/ner/train_dataset')

    batch_size = 2
    sequence_length = 10

    ner_builder = TagSequenceDatasetBuilder(input_path, label_path, dataset_dir=dataset_dir)

    ner_builder.build_vocabulary()
    ner_builder.build_trainable_dataset()
    train_data_loader, valid_data_loader = ner_builder.build_data_loader(batch_size, sequence_length)

    assert isinstance(train_data_loader, DataLoader)
    assert isinstance(valid_data_loader, DataLoader)


def test_ner_dataset_test_lodaer_iterate():
    input_path = './data_manager/test/test_dataset/ner/input.txt'
    label_path = './data_manager/test/test_dataset/ner/output.txt'

    dataset_dir = Path('./data_manager/test/test_dataset/ner/train_dataset')

    batch_size = 2
    sequence_length = 10

    ner_builder = TagSequenceDatasetBuilder(input_path, label_path, dataset_dir=dataset_dir)

    ner_builder.build_vocabulary()
    ner_builder.build_trainable_dataset()
    train_data_loader, _ = ner_builder.build_data_loader(batch_size, sequence_length)

    for batch in train_data_loader:
        train_batch = batch
        break

    assert isinstance(train_batch, dict)
    assert len(train_batch['inputs']) == batch_size
    assert train_batch['inputs']['length'][0] <= sequence_length


def test_ner_dataset_valid_lodaer_iterate():
    input_path = './data_manager/test/test_dataset/ner/input.txt'
    label_path = './data_manager/test/test_dataset/ner/output.txt'

    dataset_dir = Path('./data_manager/test/test_dataset/ner/train_dataset')

    batch_size = 2
    sequence_length = 10

    ner_builder = TagSequenceDatasetBuilder(input_path, label_path, dataset_dir=dataset_dir)

    ner_builder.build_vocabulary()
    ner_builder.build_trainable_dataset()
    _, valid_data_loader = ner_builder.build_data_loader(batch_size, sequence_length)

    for batch in valid_data_loader:
        valid_batch = batch
        break

    assert isinstance(valid_batch, dict)
    assert len(valid_batch['inputs']['value']) == 1
