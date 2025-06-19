import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from polymon.data.dataset import PolymerDataset
from polymon.exp.utils import get_logger
from polymon.model.seq_model import CNNClassifier, GRUClassifier, TransformerClassifier
from polymon.exp.train import Trainer
from polymon.exp.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default='debug')
    parser.add_argument('--out-dir', type=str, default='./results')
    
    # Dataset
    parser.add_argument('--max-seq-len', type=int, default=300)
    parser.add_argument('--max-families', type=int, default=1000)
    parser.add_argument('--use-aligned-seq', action='store_true')
    parser.add_argument('--save-processed', action='store_true')
    parser.add_argument('--force-reprocess', action='store_true')
    parser.add_argument('--batch-size', type=int, default=128)
    
    # Model
    parser.add_argument(
        '--model-type', 
        choices=['transformer', 'gru', 'cnn'], 
        default='transformer'
    )
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dim-feedforward', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument(
        '--pos-encoding-type', 
        choices=['sin', 'learnable', 'rope'], 
        default='sin'
    )
    parser.add_argument('--padding-idx', type=int, default=0)

    # Training
    parser.add_argument('--num-epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--report-every', type=int, default=500)
    parser.add_argument('--ema-decay', type=float, default=0.0)
    parser.add_argument('--early-stopping-patience', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--apply-class-weight', action='store_true')

    return parser.parse_args()


def main():
    seed_everything(42)
    
    config = parse_args().__dict__
    out_dir = os.path.join(config['out_dir'], config['desc'])
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(out_dir, 'training')
    
    # 1. Load dataset
    logger.info('Loading dataset...')
    raw_dataset = PfamRawDataset.load()
    train_dataset, dev_dataset, test_dataset, data_info = raw_dataset.process(
        max_seq_len=config['max_seq_len'],
        max_families=config['max_families'],
        use_aligned_seq=config['use_aligned_seq'],
        save=config['save_processed'],
        force_reprocess=config['force_reprocess']
    )
    config.update(data_info)
    logger.info(
        f'Train: {len(train_dataset)}, '
        f'Dev: {len(dev_dataset)}, '
        f'Test: {len(test_dataset)}'
    )
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        save_config = config.copy()
        save_config.pop('family_counts')
        save_config.pop('family_names')
        yaml.dump(save_config, f)
    
    # 2. Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],)
    
    # 3. Create model
    logger.info('Creating model...')
    if config['model_type'].lower() == 'transformer':
        model = TransformerClassifier(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            num_classes=config['num_classes'],
            max_seq_len=config['max_seq_len'],
            pos_encoding_type=config['pos_encoding_type'],
            padding_idx=config['padding_idx']
        )
    elif config['model_type'].lower() == 'gru':
        model = GRUClassifier(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            num_classes=config['num_classes']
        )
    elif config['model_type'].lower() == 'cnn':
        model = CNNClassifier(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            num_classes=config['num_classes'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len'],
        )
    else:
        raise NotImplementedError(f"Model type {config['model_type']} not implemented")
    params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model parameters: {params / 1e6:.2f}M')
    
    # 4. Train model
    logger.info('Training model...')
    if config['apply_class_weight']:
        class_weights = torch.tensor(config['family_counts'])
        class_weights = class_weights / class_weights.sum()
    else:
        class_weights = None
    trainer = Trainer(
        out_dir=out_dir,
        model=model,
        lr=config['lr'],
        num_epochs=config['num_epochs'],
        logger=logger,
        num_classes=config['num_classes'],
        report_every=config['report_every'],
        ema_decay=config['ema_decay'],
        device=config['device'],
        early_stopping_patience=config['early_stopping_patience'],
        class_weights=class_weights,
    )
    trainer.train(train_loader, dev_loader, test_loader)


if __name__ == '__main__':
    main()