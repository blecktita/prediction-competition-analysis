"""
Test scenarios for Amazon Product Category Prediction Service.
"""

import requests
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class TestScenario:
    """Test scenario container with description and expected behavior."""
    
    name: str
    description: str
    data: Dict
    expected_behavior: str


class TestCases:
    """Collection of test scenarios for the prediction service."""
    
    @staticmethod
    def get_test_scenarios() -> List[TestScenario]:
        """Return list of test scenarios."""
        return [
            # Large, established category
            TestScenario(
                name="Large Electronics Category",
                description="High volume electronics category with strong ratings",
                data={
                    "item_count": 5000,
                    "vol_purchase_total": 2500000.0,
                    "high_rating_perc": 0.75,
                    "rating_mean": 4.2,
                    "rating_std": 0.8,
                    "best_seller_count": 45
                },
                expected_behavior="High probability of Amazon presence due to large market size and good ratings"
            ),
            
            # Small niche category
            TestScenario(
                name="Niche Craft Category",
                description="Small, specialized craft supplies category",
                data={
                    "item_count": 150,
                    "vol_purchase_total": 25000.0,
                    "high_rating_perc": 0.82,
                    "rating_mean": 4.5,
                    "rating_std": 0.4,
                    "best_seller_count": 3
                },
                expected_behavior="Lower probability of Amazon presence due to small market size"
            ),
            
            # Emerging category
            TestScenario(
                name="Emerging Tech Category",
                description="New technology category with mixed ratings",
                data={
                    "item_count": 800,
                    "vol_purchase_total": 750000.0,
                    "high_rating_perc": 0.45,
                    "rating_mean": 3.8,
                    "rating_std": 1.2,
                    "best_seller_count": 12
                },
                expected_behavior="Moderate probability due to growth potential but mixed ratings"
            ),
            
            # High-value luxury category
            TestScenario(
                name="Luxury Category",
                description="High-value items with fewer sales",
                data={
                    "item_count": 300,
                    "vol_purchase_total": 1500000.0,
                    "high_rating_perc": 0.68,
                    "rating_mean": 4.0,
                    "rating_std": 0.9,
                    "best_seller_count": 8
                },
                expected_behavior="Moderate probability due to high value but lower volume"
            ),
            
            # Low-performing category
            TestScenario(
                name="Struggling Category",
                description="Category with poor ratings and low sales",
                data={
                    "item_count": 1200,
                    "vol_purchase_total": 180000.0,
                    "high_rating_perc": 0.15,
                    "rating_mean": 2.8,
                    "rating_std": 1.5,
                    "best_seller_count": 5
                },
                expected_behavior="Low probability due to poor performance metrics"
            ),
            
            # High-volume, low-margin category
            TestScenario(
                name="Bulk Items Category",
                description="High volume, low-cost items",
                data={
                    "item_count": 3500,
                    "vol_purchase_total": 850000.0,
                    "high_rating_perc": 0.55,
                    "rating_mean": 3.9,
                    "rating_std": 0.7,
                    "best_seller_count": 28
                },
                expected_behavior="High probability due to volume despite moderate ratings"
            ),
            
            # Seasonal category
            TestScenario(
                name="Seasonal Category",
                description="Holiday/seasonal items with variable sales",
                data={
                    "item_count": 950,
                    "vol_purchase_total": 425000.0,
                    "high_rating_perc": 0.62,
                    "rating_mean": 3.7,
                    "rating_std": 1.1,
                    "best_seller_count": 15
                },
                expected_behavior="Moderate probability due to seasonal nature"
            ),
            
            # Highly competitive category
            TestScenario(
                name="Competitive Category",
                description="Many sellers with varied performance",
                data={
                    "item_count": 4200,
                    "vol_purchase_total": 1750000.0,
                    "high_rating_perc": 0.48,
                    "rating_mean": 3.5,
                    "rating_std": 1.3,
                    "best_seller_count": 38
                },
                expected_behavior="High probability due to market size despite competition"
            )
        ]


def test_prediction_service(url: str = 'http://localhost:9696/predict') -> None:
    """
    Test the prediction service with all scenarios.
    
    Args:
        url: Service endpoint URL
    """
    scenarios = TestCases.get_test_scenarios()
    
    for scenario in scenarios:
        print(f"\n===== Testing: {scenario.name} =====")
        print(f"Description: {scenario.description}")
        print(f"Expected: {scenario.expected_behavior}")
        
        try:
            response = requests.post(url, json=scenario.data, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            print("\nResults:")
            print(f"Amazon Probability: {result['amazon_probability']:.2%}")
            print(f"Prediction: {'Amazon Present' if result['amazon_guess'] else 'Amazon-free'}")
            
        except requests.RequestException as e:
            print(f"Error making request: {str(e)}")
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("=" * 50)


if __name__ == "__main__":
    test_prediction_service()
