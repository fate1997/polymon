from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem
from sklearn.linear_model import LinearRegression
from polymon.setting import REPO_DIR
from loguru import logger


class Dedup:
    def __init__(
        self, 
        df: pd.DataFrame,
        label: str, 
        must_keep: List[str] = ['internal'],
        rtol: float = 0.05,
    ):
        self.label = label
        self.rtol = rtol
        # Remove other columns
        df = df[['SMILES', 'Source', 'Uncertainty', label, 'id']]
        self.df = df
        self.must_keep = must_keep
    
    def add_df(
        self,
        file_path: str,
        smiles_col: str,
        label_col: str,
        source: str,
        uncertainty: int,
    ):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f'Unsupported file type: {file_path}')
        assert label_col in df.columns and smiles_col in df.columns,\
            f'Check the columns {df.columns.tolist()}'
        df = df[df[label_col].notna()]
        df = df[[smiles_col, label_col]].assign(
            Source=source,
            Uncertainty=uncertainty,
        )
        df.rename(columns={
            label_col: self.label, 
            smiles_col: 'SMILES'
        }, inplace=True)
        df['SMILES'] = df['SMILES'].apply(lambda x: Chem.CanonSmiles(x))
        self.df = pd.concat([self.df, df], ignore_index=True)
    
    def show_distribution(
        self, 
        sources: List[str],
        bins: int = 50, 
        alpha: float = 0.5,
        density: bool = False,
    ):
        # Plot the distribution of the label for each source
        for source in sources:
            df = self.df[self.df['Source'] == source]
            plt.hist(
                df[self.label], 
                bins=bins, 
                alpha=alpha, 
                label=source, 
                density=density,
            )
        plt.legend()
        plt.xlabel(self.label)
        plt.ylabel('Density' if density else 'Count')
        plt.show()
    
    def run(self, sources: Optional[List[str]] = None, save: bool = False):
        if sources is None:
            sources = self.df['Source'].unique()
        sources = list(set(sources + self.must_keep))
        source_info = [
            f'{source}: {len(self.df[self.df["Source"] == source])}' \
            for source in sources
        ]
        logger.info(f'Sources: {", ".join(source_info)}')
        df = self.df[self.df['Source'].isin(sources)]
        
        logger.info(f'Number of rows before deduplication: {len(df)}')
        mask = df['SMILES'].duplicated(keep=False)
        duplicates = df[mask]
        smiles_groups = duplicates.groupby('SMILES').groups
        drop_indices = []
        for smiles, indices in smiles_groups.items():
            rows = df.loc[indices]
            # 1. Remove rows with lower confidence.
            min_uncertainty = rows['Uncertainty'].min()
            ids_low_uncertainty = rows[rows['Uncertainty'] > min_uncertainty].index
            drop_indices.extend(ids_low_uncertainty)
            rows = rows[rows['Uncertainty'] == min_uncertainty]
            
            # 2. Remove rows with higher relative difference.
            if len(rows) > 1:
                v = rows[self.label]
                if np.abs(v.max() - v.min()) / (v.max() + 1e-6) > self.rtol:
                    drop_indices.extend(rows.index)
                else:
                    df.loc[rows.index[0], self.label] = v.mean()
                    drop_indices.extend(rows.index[1:])
        
        # 3. Remove indices if the source is in must_keep
        for source in self.must_keep:
            drop_indices = list(set(drop_indices) - set(df[df['Source'] == source].index))
        
        df = df.drop(drop_indices)
        logger.info(f'Number of rows after deduplication: {len(df)}')
        
        if save:
            path = str(REPO_DIR / 'database' / 'merged' / f'{self.label}_{"_".join(sources)}.csv')
            df['id'] = np.arange(len(df))
            df.to_csv(path, index=False)
        
        return df
    
    def compare(
        self,
        source1: str,
        source2: str,
        fitting: bool = False,
    ):
        # Compare the label of the two sources with the same smiles
        df1 = self.df[self.df['Source'] == source1]
        df2 = self.df[self.df['Source'] == source2]
        df = pd.merge(df1, df2, on='SMILES', how='inner')
        x = df[self.label + '_x'].values
        y = df[self.label + '_y'].values
        # Swap x and y if x is the label with higher confidence
        if df1['Uncertainty'].min() > df2['Uncertainty'].min():
            x, y = y, x
            source1, source2 = source2, source1
        
        # Plot the comparison
        plt.scatter(x, y, color='blue', alpha=0.5, label='Data')
        plt.xlabel(f'{self.label} ({source1})')
        plt.ylabel(f'{self.label} ({source2})')
        plt.gca().set_aspect('equal')
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        if fitting:
            # Remove outliers
            mask = np.abs((y - x) / (x + 1e-6)) < 0.05
            x = x[mask]
            y = y[mask]
            bias = np.mean(y - x)
            y_pred = x + bias
            # Learn linear relationship y = ax + b to fit the data
            # model = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
            # y_pred = model.predict(x.reshape(-1, 1))
            # logger.info(f'Coefficient: {model.coef_[0][0]:.4f}, Intercept: {model.intercept_[0]:.4f}')
            plt.scatter(y_pred, y, color='red', alpha=0.5, marker='x', label='Fitted')
            # Write the equation of the fitted line
            plt.text(
                min_val, max_val, 
                # f'y = {model.coef_[0][0]:.4f}x + {model.intercept_[0]:.4f}',
                f'y = x + {bias:.4f}',
                fontsize=12,
                ha='left',
            )
            plt.legend()
        
        plt.show()
        if fitting:
            # Add the fitted points as a new source
            df = self.df[self.df['Source'] == source2]
            # y_fitted = model.predict(df[self.label].values.reshape(-1, 1))
            y_fitted = df[self.label].values.reshape(-1, 1) + bias
            df_fitted = pd.DataFrame({
                'SMILES': df['SMILES'],
                self.label: y_fitted.squeeze(-1),
                'Source': f'{source2}_fitted',
                'Uncertainty': 4,
            })
            self.df = pd.concat([self.df, df_fitted], ignore_index=True)
        return None