import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pickle
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration settings for the XGBoost model."""
    
    xgb_params: Dict = None
    amazon_brands: List[str] = None
    
    def __post_init__(self):
        """Initialize default configuration if none provided."""
        if self.xgb_params is None:
            self.xgb_params = {
                'eta': 0.5, 
                'max_depth': 2,
                'min_child_weight': 1,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'nthread': 8,
                'seed': 1,
                'verbosity': 1,
            }
            
        if self.amazon_brands is None:
            self.amazon_brands = [
                'Pinzon', 'Amazon Basics', 'AmazonBasics', 'Solimo',
                'Amazon Elements', 'AmazonElements', 'Amazon Brand',
                'Mama Bear', 'Wickedly Prime', 'Whole Foods',
                'AmazonFresh', 'Vedaka', 'Goodthreads', '206 Collective',
                'Amazon Essentials', 'AmazonEssentials', 'Core 10'
            ]

class DataPreprocessor:
    """Handles all data preprocessing steps for Amazon products classification."""
    
    def __init__(self, file_path: str):
        """
        Initialize the preprocessor.
        
        Args:
            file_path: Path to the raw data CSV file
        """
        self.file_path = file_path
        self._data = None
        
    def load_data(self) -> None:
        """Load and perform initial cleaning of the raw data."""
        self._data = pd.read_csv(self.file_path)
        self._data = self._data.dropna()
        self._data = self._data[self._data['price'] != 0]
        
    def create_features(self) -> None:
        """Create additional features from raw data."""
        if self._data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Calculate discount features
        mask = (self._data['listPrice'] != 0)
        self._data['discount'] = 0.0
        self._data.loc[mask, 'discount'] = (
            self._data['listPrice'] - self._data['price']
        )
        self._data['perc_discount'] = self._data['discount'] / self._data['price']
        
        # Create other features
        self._data['high_rating'] = (self._data['stars'] >= 4.5)
        self._data['vol_purchase'] = (
            self._data['boughtInLastMonth'] * self._data['price']
        )
        
    def mark_amazon_products(self, brands: List[str]) -> None:
        """
        Mark Amazon-owned products.
        
        Args:
            brands: List of Amazon brand names to identify
        """
        if self._data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        pattern = '|'.join(brands)
        self._data['amazon_owned'] = (
            self._data['title'].str.contains(pattern, case=False).astype(int)
        )
        
    def aggregate_by_category(self) -> pd.DataFrame:
        """
        Aggregate data at category level.
        
        Returns:
            DataFrame with category-level aggregations
        """
        if self._data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        df = self._data.groupby(['category_id']).agg(
            item_count=('title', 'count'),
            amazon_owned_total=('amazon_owned', 'sum'),
            vol_purchase_total=('vol_purchase', 'sum'),
            high_rating_perc=('high_rating', 'mean'),
            rating_mean=('stars', 'mean'),
            rating_std=('stars', 'std'),
            best_seller_count=('isBestSeller', 'sum')
        ).reset_index()
        
        df['amazon_owned'] = (df.amazon_owned_total > 0).astype(int)
        df.drop('amazon_owned_total', axis=1, inplace=True)
        return df.set_index('category_id')

class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.model = None
        self.vectorizer = None
        
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2
                    ) -> Tuple[xgb.DMatrix, xgb.DMatrix, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data to use for testing
            
        Returns:
            Training DMatrix, testing DMatrix, and test labels
        """
        # Split data
        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=50
        )
        
        # Extract target variable
        y_train = df_train.pop('amazon_owned').values
        y_test = df_test.pop('amazon_owned').values
        
        # Convert to DMatrix
        self.vectorizer = DictVectorizer(sparse=False)
        X_train = self.vectorizer.fit_transform(
            df_train.to_dict(orient='records')
        )
        X_test = self.vectorizer.transform(df_test.to_dict(orient='records'))
        
        feature_names = self.vectorizer.get_feature_names_out().tolist()
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, feature_names=feature_names)
        
        return dtrain, dtest, y_test
    
    def train(self, dtrain: xgb.DMatrix) -> None:
        """
        Train the model.
        
        Args:
            dtrain: Training data in DMatrix format
        """
        self.model = xgb.train(
            self.config.xgb_params, dtrain, num_boost_round=100
        )
    
    def evaluate(self, dtest: xgb.DMatrix, y_test: np.ndarray) -> float:
        """
        Evaluate model performance.
        
        Args:
            dtest: Test data in DMatrix format
            y_test: True test labels
            
        Returns:
            AUC score
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        y_pred = self.model.predict(dtest)
        return roc_auc_score(y_test, y_pred)
    
    def save_model(self, output_file: str) -> None:
        """
        Save model and vectorizer to file.
        
        Args:
            output_file: Path to save the model
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model or vectorizer not initialized.")
            
        with open(output_file, 'wb') as f_out:
            pickle.dump((self.vectorizer, self.model), f_out)

class AmazonProductClassifier:
    """Main class orchestrating the entire training process."""
    
    def __init__(self, data_path: str, output_path: str, config: ModelConfig = None):
        """
        Initialize the classifier.
        
        Args:
            data_path: Path to input data
            output_path: Path to save model
            config: Optional configuration object
        """
        self.data_path = data_path
        self.output_path = output_path
        self.config = config or ModelConfig()
        
        self.preprocessor = DataPreprocessor(data_path)
        self.trainer = ModelTrainer(self.config)
        
    def train(self) -> float:
        """
        Execute the full training pipeline.
        
        Returns:
            Model AUC score
        """
        print('Importing and preprocessing data...')
        # Preprocess data
        self.preprocessor.load_data()
        self.preprocessor.create_features()
        self.preprocessor.mark_amazon_products(self.config.amazon_brands)
        final_df = self.preprocessor.aggregate_by_category()
        
        print('Training the model...')
        # Train and evaluate
        dtrain, dtest, y_test = self.trainer.prepare_data(final_df)
        self.trainer.train(dtrain)
        auc = self.trainer.evaluate(dtest, y_test)
        
        # Save model
        self.trainer.save_model(self.output_path)
        print(f'Model saved to {self.output_path}')
        
        return auc

def main():
    """Main function to run the training pipeline."""
    # Initialize classifier
    classifier = AmazonProductClassifier(
        data_path='data/amazon_products.csv',
        output_path='model_v1.bin'
    )
    
    # Train model
    auc = classifier.train()
    print(f'Final model AUC: {auc:.4f}')

if __name__ == '__main__':
    main()